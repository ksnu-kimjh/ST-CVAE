import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from utils import seed_everything
import numpy as np
import pandas as pd
import typing as ty
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

# --- 2. ST-CVAE Components ---

class Encoder(nn.Module):
    """
    Posterior Network q(z|x, y).
    학습 시에만 사용됩니다. 정답 Y를 보고 Z를 생성합니다.
    """
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
    """
    Prior Network p(z|x).
    추론 시에 사용됩니다. X만 보고 Z의 분포를 예측합니다.
    """
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
    """
    Generator Network p(y|x, z).
    X와 Latent Z를 받아 Y를 예측합니다.
    """
    def __init__(self, n_layers, d_in, d_out, d_latent, d_hidden, dropout=0.1):
        super().__init__()
        # Input: X features + 4 TL features + Latent Z
        self.input_proj = nn.Linear(d_in + 4 + d_latent, d_hidden)
        self.blocks = nn.ModuleList([ResBlock(d_hidden, dropout) for _ in range(n_layers)])
        self.head = nn.Linear(d_hidden, d_out)

    def forward(self, x, z):
        h = self.input_proj(torch.cat([x, z], dim=-1))
        for block in self.blocks:
            h = block(h)
        return self.head(h)

# --- 3. ST-CVAE Core ---

class ST_CVAECore(nn.Module):
    def __init__(self, d_in, d_out, n_layers, d_latent, d_hidden, dropout=0.1, kl_weight=0.001):
        super().__init__()
        self.d_latent = d_latent
        self.kl_weight = kl_weight
        
        self.tl_encoder = TrafficLightEncoder()
        
        # Modules
        self.encoder = Encoder(d_in, d_out, d_latent, d_hidden, dropout) # q(z|x,y)
        self.prior = Prior(d_in, d_latent, d_hidden, dropout) # p(z|x)
        self.decoder = Decoder(n_layers, d_in, d_out, d_latent, d_hidden, dropout) # p(y|x,z)

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
        
        # 3. Prior Network (for KL loss)
        mu_prior, logvar_prior = self.prior(x)
        
        # Loss Calculation
        recon_loss = F.mse_loss(y_recon, y)
        
        # KL Divergence between Posterior and Prior
        # KL(q(z|x,y) || p(z|x)) -> We want Prior to match Posterior
        # This forces the Prior network to learn to guess Z as well as possible from X
        kl_loss = -0.5 * torch.mean(1 + logvar_post - logvar_prior - 
                                    (logvar_post.exp() + (mu_post - mu_prior).pow(2)) / logvar_prior.exp())
        
        return recon_loss, kl_loss

    @torch.no_grad()
    def predict(self, x, n_samples=1, method='mean'):
        # Inference Step: Only X is available
        mu_prior, logvar_prior = self.prior(x)
        
        if method == 'mean':
            # Deterministic: Mean of Prior
            z = mu_prior
            return self.decoder(x, z)
        else:
            # Stochastic: Monte Carlo Sampling from Prior
            batch_samples = []
            for _ in range(n_samples):
                z = self.reparameterize(mu_prior, logvar_prior)
                y_gen = self.decoder(x, z)
                batch_samples.append(y_gen)
            
            # (B, N, D)
            return torch.stack(batch_samples, dim=1)

# --- 4. Wrapper ---

class ST_CVAE_CQR_Wrapper(BaseEstimator, RegressorMixin):
    def __init__(self, numerical_cols, categorical_cols, spatial_cols, d_out, 
                 n_layers=3, d_hidden=256, d_latent=16, epochs=100, batch_size=512, lr=1e-3, 
                 weight_decay=1e-5, kl_weight=0.01, dropout=0.1, patience=20, 
                 verbose=False, seed=77, device='cuda', 
                 calibration_ratio=0.2, quantiles=[0.05, 0.95], **kwargs):
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.d_out = d_out
        self.n_layers = n_layers
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
        self.cqr_correction_ = None
        
        self.model = None
        self.tl_cols = ['tlt_c', 'tls_c', 'tlt_n', 'tls_n']
        self.history = {'train_loss': [], 'val_rmse': []}

    def _fit_preprocessor(self, X, y):
        # X: QuantileTransformer
        self.preprocessor_X = ColumnTransformer(
            transformers=[
                ('num', QuantileTransformer(output_distribution='normal', n_quantiles=1000, random_state=self.seed), self.numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), self.categorical_cols)
            ], remainder='drop')
        self.preprocessor_X.fit(X)
        
        # Y: StandardScaler
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
        
        # Calib split
        X_train, X_cal, y_train, y_cal = train_test_split(X, y_np, test_size=self.calibration_ratio, random_state=self.seed)
        
        self._fit_preprocessor(X_train, y_train)
        X_tr = self._transform_X(X_train); y_tr = self._transform_y(y_train)
        
        # Base dim (excluding TL cols which are +4 inside)
        d_in = X_tr.shape[1] - 4
        
        self.model = ST_CVAECore(d_in, self.d_out, self.n_layers, self.d_latent, self.d_hidden, 
                              self.dropout, self.kl_weight).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        train_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        
        if eval_set:
            X_va = self._transform_X(eval_set[0])
            y_va = self._transform_y(eval_set[1])
            y_va_raw = eval_set[1]
            val_ds = TensorDataset(torch.tensor(X_va), torch.tensor(y_va))
            val_dl = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
        else:
            val_dl = None
        
        best_rmse = float('inf')
        patience_cnt = 0
        
        for epoch in range(self.epochs):
            self.model.train()
            tr_loss = 0
            for bx, by in train_dl:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                recon, kl = self.model(bx, by)
                loss = recon + self.kl_weight * kl
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                tr_loss += loss.item()
            
            sval_rmse_real = 0
            if val_dl:
                self.model.eval()
                val_preds = []
                with torch.no_grad():
                    for bx, by in val_dl:
                        bx = bx.to(self.device)
                        # Validation using Prior Mean (Deterministic)
                        pred = self.model.predict(bx, method='mean')
                        val_preds.append(pred.cpu().numpy())
            
                val_preds_orig = self._inverse_transform_y(np.concatenate(val_preds))
                val_mse_real = mean_squared_error(y_va_raw, val_preds_orig)
                val_rmse_real = np.sqrt(val_mse_real)
            
            if self.verbose:
                print(f"Epoch {epoch}: Train Loss {tr_loss/len(train_dl):.4f} | Val RMSE: {val_rmse_real:.4f} | Best RMSE: {best_rmse:.4f} (Patience: {patience_cnt})")
            
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
        
        # CQR Calibration
        if X_cal is not None:
            self.model.eval()
            # Simplified CQR:
            intervals = self.predict_interval(X_cal)
            lo, up = intervals[:,:,0], intervals[:,:,1]
            scores = np.maximum(lo - y_cal, y_cal - up)

            alpha = 1.0 - (self.quantiles[1] - self.quantiles[0])
            n_cal = len(scores)
            q_level = min(1.0, np.ceil((n_cal + 1) * (1 - alpha)) / n_cal)
            self.cqr_correction_ = np.quantile(scores, q_level, axis=0)
        
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

    @torch.no_grad()
    def predict_interval(self, X):
        self.model.eval()
        X_p = self._transform_X(X)
        ds = TensorDataset(torch.tensor(X_p))
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False)

        low_list, upp_list = [], []
        n_samples = 50

        for bx, in dl:
            bx = bx.to(self.device)
            B = bx.size(0)

            mu_prior, logvar_prior = self.model.prior(bx)

            # Reparameterization
            std = torch.exp(0.5 * logvar_prior)
            eps = torch.randn(n_samples, B, self.d_latent, device=self.device)
            z = mu_prior + eps * std

            bx_expanded = bx.unsqueeze(0).expand(n_samples, -1, -1)
            decoder_input = torch.cat([bx_expanded, z], dim=-1)

            decoder_input_flat = decoder_input.reshape(-1, decoder_input.size(-1))
            y_gen_flat = self.model.decoder(decoder_input_flat[:, :-self.d_latent], decoder_input_flat[:, -self.d_latent:])

            # Inverse transform
            y_gen_np_flat = y_gen_flat.cpu().numpy()
            y_inv_flat = self._inverse_transform_y(y_gen_np_flat)
            y_inv = y_inv_flat.reshape(n_samples, B, self.d_out)

            low_list.append(np.quantile(y_inv, self.quantiles[0], axis=0))
            upp_list.append(np.quantile(y_inv, self.quantiles[1], axis=0))

        lo = np.concatenate(low_list, axis=0)
        up = np.concatenate(upp_list, axis=0)

        if self.cqr_correction_ is not None:
            lo -= self.cqr_correction_
            up += self.cqr_correction_

        return np.stack([lo, up], axis=2)

    # def predict_interval(self, X):
    #     self.model.eval()
    #     X_p = self._transform_X(X)
    #     ds = TensorDataset(torch.tensor(X_p))
    #     dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
    #     low, upp = [], []
    #     n_samples = 1000
    #     with torch.no_grad():
    #         for bx, in dl:
    #             bx = bx.to(self.device)
    #             # Sample form Prior
    #             p = self.model.predict(bx, n_samples=n_samples, method='sample') # (B, N, D)
                
    #             B, N, D = p.shape
    #             flat = p.reshape(-1, D).cpu().numpy()
    #             p_inv = self._inverse_transform_y(flat).reshape(B, N, D)
                
    #             low.append(np.quantile(p_inv, self.quantiles[0], axis=1))
    #             upp.append(np.quantile(p_inv, self.quantiles[1], axis=1))
                
    #     lo = np.concatenate(low)
    #     up = np.concatenate(upp)
        
    #     if self.cqr_correction_ is not None:
    #         lo -= self.cqr_correction_
    #         up += self.cqr_correction_
            
    #     return np.stack([lo, up], axis=2)

    def evaluate_objective(self, X, y):
        # Optuna Metric: MSE of Mean Prediction
        preds = self.predict(X)
        y_val = y.values if isinstance(y, pd.DataFrame) else y
        return mean_squared_error(y_val, preds)
    
    def get_dist_params(self, X):
        # [CRITICAL FIX] Returns samples for Energy Score Calculation
        self.model.eval()
        X_p = self._transform_X(X)
        ds = TensorDataset(torch.tensor(X_p))
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
        
        all_samples = []
        n_samples = 50 # Number of samples for ES calculation
        
        with torch.no_grad():
            for bx, in dl:
                bx = bx.to(self.device)
                # Predict samples using Prior p(z|x)
                p = self.model.predict(bx, n_samples=n_samples, method='sample') # (B, N, D)
                
                B, N, D = p.shape
                flat = p.reshape(-1, D).cpu().numpy()
                p_inv = self._inverse_transform_y(flat).reshape(B, N, D)
                all_samples.append(p_inv)
                
        return np.concatenate(all_samples, axis=0)

    def get_latent_representations(self, X):
        self.model.eval()
        X_p = self._transform_X(X)
        ds = TensorDataset(torch.tensor(X_p))
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
        
        mus = []
        with torch.no_grad():
            for bx, in dl:
                bx = bx.to(self.device)
                mu, _ = self.model.prior(bx)
                mus.append(mu.cpu().numpy())
                
        return np.concatenate(mus, axis=0)