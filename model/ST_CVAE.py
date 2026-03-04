import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler, QuantileTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from utils import seed_everything
import numpy as np
import pandas as pd
import copy

# --- 1. Modules ---

class ReGLU(nn.Module):
    def forward(self, x):
        chunks = x.chunk(2, dim=-1)
        return chunks[0] * F.relu(chunks[1])

class ResBlock(nn.Module):
    def __init__(self, d_hidden, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_hidden)
        self.linear1 = nn.Linear(d_hidden, d_hidden * 2)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_hidden * 2)
        self.act = ReGLU()

    def forward(self, x):
        h = self.norm1(x)
        h = self.linear1(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.norm2(h)
        h = self.linear2(h)
        h = self.act(h)
        return x + h 

class TrafficLightEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.T = 84.0

    def forward(self, tlt_c, tls_c_flag, tlt_n, tls_n_flag):
        tlt_c = torch.clamp(tlt_c, 0.0, 41.0)
        tlt_n = torch.clamp(tlt_n, 0.0, 41.0)

        def get_phase(tlt, flg):
            phi = torch.zeros_like(tlt)
            mask_r = (flg == -1); phi[mask_r] = (41.0 - tlt[mask_r])
            mask_g = (flg == 1); phi[mask_g] = 41.0 + (41.0 - tlt[mask_g])
            mask_y = (flg == 0); phi[mask_y] = 82.0 + (2.0 - tlt[mask_y])
            return phi

        phi_c = get_phase(tlt_c, tls_c_flag)
        phi_n = get_phase(tlt_n, tls_n_flag)

        return torch.stack([
            torch.sin(2*np.pi*phi_c/self.T), torch.cos(2*np.pi*phi_c/self.T),
            torch.sin(2*np.pi*phi_n/self.T), torch.cos(2*np.pi*phi_n/self.T)
        ], dim=1)

# --- 2. ST-CVAE Components (Traffic: With TLE) ---

class Encoder(nn.Module):
    def __init__(self, d_in, d_out, d_latent, d_hidden, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in + d_out + 4, d_hidden), 
            ResBlock(d_hidden, dropout),
            ResBlock(d_hidden, dropout),
            nn.LayerNorm(d_hidden)
        )
        self.mu = nn.Linear(d_hidden, d_latent)
        self.logvar = nn.Linear(d_hidden, d_latent)

    def forward(self, x, y):
        h = self.net(torch.cat([x, y], dim=-1))
        return self.mu(h), self.logvar(h)

class Prior(nn.Module):
    def __init__(self, d_in, d_latent, d_hidden, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in + 4, d_hidden), 
            ResBlock(d_hidden, dropout),
            ResBlock(d_hidden, dropout),
            nn.LayerNorm(d_hidden)
        )
        self.mu = nn.Linear(d_hidden, d_latent)
        self.logvar = nn.Linear(d_hidden, d_latent)

    def forward(self, x):
        h = self.net(x)
        return self.mu(h), self.logvar(h)

class Decoder(nn.Module):
    def __init__(self, d_in, d_out, d_latent, d_hidden, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(d_in + 4 + d_latent, d_hidden)
        self.blocks = nn.ModuleList([ResBlock(d_hidden, dropout) for _ in range(3)])
        self.head = nn.Linear(d_hidden, d_out)

    def forward(self, x, z):
        h = self.input_proj(torch.cat([x, z], dim=-1))
        for block in self.blocks:
            h = block(h)
        return self.head(h)

# --- 3. ST-CVAE Core (Traffic) ---

class ST_CVAECore(nn.Module):
    def __init__(self, d_in, d_out, d_latent, d_hidden, dropout=0.1, kl_weight=0.001):
        super().__init__()
        self.d_latent = d_latent
        self.kl_weight = kl_weight
        
        self.tl_encoder = TrafficLightEncoder()
        
        self.encoder = Encoder(d_in, d_out, d_latent, d_hidden, dropout)
        self.prior = Prior(d_in, d_latent, d_hidden, dropout)
        self.decoder = Decoder(d_in, d_out, d_latent, d_hidden, dropout)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y):
        # 1. Train Step: Encode Y -> Z
        mu_post, logvar_post = self.encoder(x, y)
        z = self.reparameterize(mu_post, logvar_post)
        
        # 2. Decode Z -> Y_pred
        y_recon = self.decoder(x, z)
        
        # 3. Prior Network
        mu_prior, logvar_prior = self.prior(x)
        
        # Loss Calculation
        recon_loss = F.mse_loss(y_recon, y)
        # KL Divergence
        kl_loss = -0.5 * torch.mean(1 + logvar_post - logvar_prior - 
                                    (logvar_post.exp() + (mu_post - mu_prior).pow(2)) / logvar_prior.exp())
        
        return recon_loss, kl_loss

    @torch.no_grad()
    def predict(self, x, n_samples=1, method='mean'):
        mu_prior, logvar_prior = self.prior(x)
        
        if method == 'mean':
            z = mu_prior
            return self.decoder(x, z)
        else:
            batch_samples = []
            for _ in range(n_samples):
                z = self.reparameterize(mu_prior, logvar_prior)
                y_gen = self.decoder(x, z)
                batch_samples.append(y_gen)
            return torch.stack(batch_samples, dim=1)

# --- 4. Wrapper (Traffic) ---

class ST_CVAE_Wrapper(BaseEstimator, RegressorMixin):
    def __init__(self, numerical_cols, categorical_cols, spatial_cols, d_out, 
                 d_hidden=256, d_latent=16, epochs=100, batch_size=512, lr=1e-3, 
                 weight_decay=1e-5, kl_weight=0.01, dropout=0.1, patience=20, 
                 verbose=False, seed=77, device='cuda', 
                 calibration_ratio=0.2, quantiles=[0.05, 0.95], **kwargs):
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.d_out = d_out
        self.d_hidden = d_hidden
        self.d_latent = d_latent
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.kl_weight = kl_weight
        self.dropout = dropout
        self.patience = patience
        self.verbose = verbose
        self.seed = seed
        self.device = device
        self.calibration_ratio = calibration_ratio
        self.quantiles = sorted(quantiles)
        
        self.model = None
        self.tl_cols = ['tlt_c', 'tls_c', 'tlt_n', 'tls_n']
        self.history = {'train_loss': [], 'val_rmse': []}

    def _fit_preprocessor(self, X, y):
        self.preprocessor_X = ColumnTransformer(
            transformers=[
                ('num', QuantileTransformer(output_distribution='normal', n_quantiles=1000, random_state=self.seed), self.numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), self.categorical_cols)
            ], remainder='drop')
        self.preprocessor_X.fit(X)
        self.preprocessor_y = StandardScaler()
        self.preprocessor_y.fit(y)

    def _enhance_features(self, X):
        for c in self.tl_cols: 
            if c not in X.columns: X[c] = 0.0
        tlt_c = torch.tensor(X['tlt_c'].values, dtype=torch.float32)
        tls_c = torch.tensor(X['tls_c'].values, dtype=torch.float32)
        tlt_n = torch.tensor(X['tlt_n'].values, dtype=torch.float32)
        tls_n = torch.tensor(X['tls_n'].values, dtype=torch.float32)
        enc = TrafficLightEncoder()
        return enc(tlt_c, tls_c, tlt_n, tls_n).numpy()

    def _transform_X(self, X):
        X_base = self.preprocessor_X.transform(X).astype(np.float32)
        X_tl = self._enhance_features(X).astype(np.float32)
        return np.hstack([X_base, X_tl])

    def _transform_y(self, y):
        return self.preprocessor_y.transform(y).astype(np.float32)

    def _inverse_transform_y(self, y):
        return self.preprocessor_y.inverse_transform(y)
    
    def fit(self, X, y, eval_set=None):
        seed_everything(self.seed)
        y_np = y.values if isinstance(y, pd.DataFrame) else y
        
        # No CQR Split
        
        self._fit_preprocessor(X, y_np)
        X_tr = self._transform_X(X)
        y_tr = self._transform_y(y_np)

        d_in = X_tr.shape[1] - 4
        
        self.model = ST_CVAECore(d_in, self.d_out, self.d_latent, self.d_hidden, 
                              self.dropout, self.kl_weight).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        train_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        
        if eval_set:
            X_va_tr = self._transform_X(eval_set[0])
            val_ds = TensorDataset(torch.tensor(X_va_tr))
            val_dl = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
        else:
            val_dl = None
        
        best_rmse = float('inf')
        patience_cnt = 0
        best_state = None
        
        for epoch in range(self.epochs):
            self.model.train()
            tr_loss = 0
            for bx, by in train_dl:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                recon, kl = self.model(bx, by)
                loss = recon + self.kl_weight * kl
                loss.backward()
                optimizer.step()
                tr_loss += loss.item()
            
            val_rmse_real = 0
            if val_dl:
                self.model.eval()
                val_preds = []
                with torch.no_grad():
                    for bx, in val_dl:
                        bx = bx.to(self.device)
                        pred = self.model.predict(bx, method='mean')
                        val_preds.append(pred.cpu().numpy())
            
                val_preds_orig = self._inverse_transform_y(np.concatenate(val_preds))
                val_mse_real = mean_squared_error(eval_set[1], val_preds_orig)
                val_rmse_real = np.sqrt(val_mse_real)
            
            if self.verbose:
                print(f"Epoch {epoch}: Train Loss {tr_loss/len(train_dl):.4f} | Val RMSE: {val_rmse_real:.4f} | Best RMSE: {best_rmse:.4f}")
            
            self.history['train_loss'].append(tr_loss/len(train_dl))
            self.history['val_rmse'].append(val_rmse_real)
            
            if val_rmse_real < best_rmse:
                best_rmse = val_rmse_real
                patience_cnt = 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                patience_cnt += 1
                if patience_cnt >= self.patience: break
                
        if best_state: self.model.load_state_dict(best_state)
        
        # No CQR
        
        return self

    def predict(self, X):
        self.model.eval()
        X_p = self._transform_X(X)
        ds = TensorDataset(torch.tensor(X_p))
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
        preds = []
        with torch.no_grad():
            for bx, in dl:
                bx = bx.to(self.device)
                p = self.model.predict(bx, method='mean')
                preds.append(p.cpu().numpy())
        return self._inverse_transform_y(np.concatenate(preds))

    def predict_interval(self, X):
        self.model.eval()
        X_p = self._transform_X(X)
        ds = TensorDataset(torch.tensor(X_p))
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
        low, upp = [], []
        n_samples = 50
        with torch.no_grad():
            for bx, in dl:
                bx = bx.to(self.device)
                p = self.model.predict(bx, n_samples=n_samples, method='sample')
                B, N, D = p.shape
                flat = p.reshape(-1, D).cpu().numpy()
                p_inv = self._inverse_transform_y(flat).reshape(B, N, D)
                low.append(np.quantile(p_inv, self.quantiles[0], axis=1))
                upp.append(np.quantile(p_inv, self.quantiles[1], axis=1))
                
        lo = np.concatenate(low)
        up = np.concatenate(upp)
        
        # No CQR
            
        return np.stack([lo, up], axis=2)

    def evaluate_objective(self, X, y):
        preds = self.predict(X)
        y_val = y.values if isinstance(y, pd.DataFrame) else y
        return mean_squared_error(y_val, preds)
    
    def get_dist_params(self, X):
        self.model.eval()
        X_p = self._transform_X(X)
        ds = TensorDataset(torch.tensor(X_p))
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
        all_samples = []
        n_samples = 50
        with torch.no_grad():
            for bx, in dl:
                bx = bx.to(self.device)
                p = self.model.predict(bx, n_samples=n_samples, method='sample')
                B, N, D = p.shape
                flat = p.reshape(-1, D).cpu().numpy()
                p_inv = self._inverse_transform_y(flat).reshape(B, N, D)
                all_samples.append(p_inv)
                
        return np.concatenate(all_samples, axis=0)
