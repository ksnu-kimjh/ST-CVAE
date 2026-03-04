import os
import glob
import yaml
import random
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def seed_everything(seed:int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path="settings.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
    
def load_data(config):
    data_frames = []
    # 데이터 경로 처리 (List or String)
    dirs = config['project']['data_dirs']
    if isinstance(dirs, str):
        dirs = [dirs]

    for d in dirs:
        path = os.path.join(d, config['project']['file_pattern'])
        files = glob.glob(path)
        for f in files:
            try:
                df = pd.read_csv(f)
                data_frames.append(df)
            except Exception as e:
                print(f"[Warning] Failed to load {f}: {e}")

    if not data_frames:
        raise FileNotFoundError("No data files loaded. Check 'data_dirs' and 'file_pattern' in settings.yaml")
    
    full_df = pd.concat(data_frames, ignore_index=True)
    print(f"Data Loaded: {len(full_df)} samples from {len(data_frames)} files.")
    return full_df

def feature_engineering(df:pd.DataFrame):
    _df = df.copy()

    # Type 2 Targets 생성
    if 'next_entry_time' in _df.columns and 'dwell_time' in _df.columns:
        _df['delta_entry_time'] = _df['next_entry_time'] - _df['dwell_time']
    if 'exit_time' in _df.columns and 'next_entry_time' in _df.columns:
        _df['delta_exit_time'] = _df['exit_time'] - _df['next_entry_time']
    
    # Naive Features
    epsilon = 1e-5
    v_c_a = _df['v_c_a'].replace(0, epsilon)
    v_n_a = _df['v_n_a'].replace(0, epsilon)

    _df['dist_to_exit'] = _df['r_cov'] + (_df['dirct'].astype(int) * _df['d_t_c'])
    _df['naive_dwell_time'] = _df['dist_to_exit'] / v_c_a
    _df['naive_next_entry_time'] = _df['d_n_c'] / v_c_a
    _df['naive_exit_time'] = _df['naive_next_entry_time'] + (2 * _df['r_cov'] / v_n_a)

    return _df

def strat_split_data(df:pd.DataFrame, test_size=0.3, random_state=77):
    strata_cols = ['dirct', 'tls_c']
    valid_strata = [c for c in strata_cols if c in df.columns]
    
    if not valid_strata:
        return train_test_split(df, test_size=test_size, random_state=random_state)
    
    temp_df = df[valid_strata].fillna('unknown')
    if len(valid_strata) > 1:
        stratify_col = temp_df.astype(str).agg('-'.join, axis=1)
    else:
        stratify_col = temp_df[valid_strata[0]]
    
    # 빈도가 너무 낮은 클래스 제외
    vc = stratify_col.value_counts()
    valid_indices = stratify_col.isin(vc[vc > 1].index)
    
    train_df, test_df = train_test_split(
        df[valid_indices],
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col[valid_indices]
    )

    if (~valid_indices).sum() > 0:
        train_df = pd.concat([train_df, df[~valid_indices]], axis=0)
    
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

def sampling_stratified(df:pd.DataFrame, n_samples=50000, random_seed=77):
    if len(df) <= n_samples:
        return df
    
    strata_cols = ['dirct', 'tls_c', 'cur_rsu']
    valid_strata = [c for c in strata_cols if c in df.columns]

    if not valid_strata:
        return df.sample(n=n_samples, random_state=random_seed).reset_index(drop=True)
    
    temp_df = df[valid_strata].fillna('unknown')
    stratify_col = temp_df.astype(str).agg('-'.join, axis=1)

    vc = stratify_col.value_counts()
    valid_indices = stratify_col.isin(vc[vc > 1].index)
    
    df_subset, _ = train_test_split(
        df[valid_indices],
        train_size=n_samples,
        stratify=stratify_col[valid_indices],
        random_state=random_seed
    )
    return df_subset.reset_index(drop=True)

# --- Metrics Calculation ---

def calculate_gaussian_nll(y_true, mu, sigma):
    sigma = np.maximum(sigma, 1e-6)
    nll = 0.5 * np.log(2 * np.pi * sigma**2) + ((y_true - mu)**2) / (2 * sigma**2)
    return np.mean(nll)

def calculate_gmm_nll(y_true, pi, mu, sigma):
    if y_true.ndim == 1:
        y_true = y_true[:, np.newaxis]
    
    sigma = np.maximum(sigma, 1e-6)
    var = sigma**2
    log_scale = -0.5 * np.log(2 * np.pi * var)
    log_exp = -0.5 * (y_true - mu)**2 / var

    log_prob_comp = log_scale + log_exp
    log_weighted = np.log(pi + 1e-9) + log_prob_comp

    max_log = np.max(log_weighted, axis=1, keepdims=True)
    log_likelihood = max_log + np.log(np.sum(np.exp(log_weighted - max_log), axis=1, keepdims=True))
    return -np.mean(log_likelihood)

def calculate_gaussian_crps(y_true, mu, sigma):
    sigma = np.maximum(sigma, 1e-6)
    z = (y_true - mu) / sigma
    pdf = norm.pdf(z)
    cdf = norm.cdf(z)
    crps = sigma * (z * (2 * cdf - 1) + 2 * pdf - 1 / np.sqrt(np.pi))
    return np.mean(crps)

def calculate_gmm_crps(y_true, pi, mu, sigma):
    if y_true.ndim == 1:
        y_true = y_true[:, np.newaxis]
    
    sigma = np.maximum(sigma, 1e-6)

    def A(u, s):
        z = u / s
        return 2 * s * norm.pdf(z) + u * (2 * norm.cdf(z) - 1)
    
    term1 = np.sum(pi * A(y_true - mu, sigma), axis=1)

    mu_expand_k = mu[:, :, np.newaxis]
    sigma_sq_expand_k = sigma[:, :, np.newaxis]**2
    pi_expand_k = pi[:, :, np.newaxis]

    mu_expand_l = mu[:, np.newaxis, :]
    sigma_sq_expand_l = sigma[:, np.newaxis, :]**2
    pi_expand_l = pi[:, np.newaxis, :]

    u_diff = mu_expand_k - mu_expand_l
    s_comb = np.sqrt(sigma_sq_expand_k + sigma_sq_expand_l)
    w_comb = pi_expand_k * pi_expand_l
    
    term2_matrix = w_comb * A(u_diff, s_comb)
    term2 = np.sum(term2_matrix, axis=(1, 2))
    
    crps = term1 - 0.5 * term2
    return np.mean(crps)

def calculate_energy_score(y_true, y_samples):
    """
    Multivariate Energy Score (ES). Lower is better.
    y_true: (N, D)
    y_samples: (N, M, D)
    """
    N, M, D = y_samples.shape
    if len(y_true.shape) == 1:
        y_true = y_true[:, np.newaxis]
    
    # Term 1: E ||X - y||
    diff_truth = y_samples - y_true[:, np.newaxis, :] # (N, M, D)
    norm_truth = np.linalg.norm(diff_truth, axis=2) # (N, M)
    term1 = np.mean(norm_truth, axis=1) # (N,)

    # Term 2: -0.5 * E ||X - X'|| (Diversity)
    s1 = y_samples[:, :, np.newaxis, :]
    s2 = y_samples[:, np.newaxis, :, :]
    diff_samples = s1 - s2 # (N, M, M, D)
    norm_samples = np.linalg.norm(diff_samples, axis=3) # (N, M, M)
    term2 = np.sum(norm_samples, axis=(1, 2)) / (2 * (M**2)) # Expectation
    
    return np.mean(term1 - term2)

def calculate_interval_score(y_true, y_lower, y_upper, alpha=0.1):
    """
    Winkler Score (Interval Score). Lower is better.
    S = (U - L) + (2/alpha) * (L - y) * I(y < L) + (2/alpha) * (y - U) * I(y > U)
    """
    # Width penalty
    width = y_upper - y_lower
    
    # Coverage penalty
    # If y < Lower
    below = (y_lower - y_true) * (y_true < y_lower)
    # If y > Upper
    above = (y_true - y_upper) * (y_true > y_upper)
    
    score = width + (2.0 / alpha) * (below + above)
    return np.mean(score)

def calculate_metrics(y_true, y_pred, y_lower, y_upper, dist_params=None, y_train_range=None, y_samples=None, alph=0.1):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    mpiw = np.mean(y_upper - y_lower)
    covered = (y_true >= y_lower) & (y_true <= y_upper)
    picp = np.mean(covered)

    nmpiw = np.nan
    if y_train_range is not None and y_train_range > 0:
        nmpiw = mpiw / y_train_range
    
    nll = np.nan
    crps = np.nan
    energy_score = np.nan

    if dist_params is not None:
        try:
            dtype = dist_params.get('type')
            if dtype == 'gaussian':
                mu, sigma = dist_params['mu'], dist_params['sigma']
                nll = calculate_gaussian_nll(y_true, mu, sigma)
                crps = calculate_gaussian_crps(y_true, mu, sigma)
            elif dtype == 'gmm':
                pi, mu, sigma = dist_params['pi'], dist_params['mu'], dist_params['sigma']
                nll = calculate_gmm_nll(y_true, pi, mu, sigma)
                crps = calculate_gmm_crps(y_true, pi, mu, sigma)
        except Exception:
            pass
            
    if y_samples is not None:
        try:
            energy_score = calculate_energy_score(y_true, y_samples)
        except Exception as e:
            print(f"[Warning] Energy Score calc failed: {e}")

    # Interval Score (Winkler Score)
    interval_score = np.nan
    if alph is not None:
        try:
             interval_score = calculate_interval_score(y_true, y_lower, y_upper, alph)
        except Exception as e:
             pass

    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'PICP': picp,
        'MPIW': mpiw,
        'NMPIW': nmpiw,
        'NLL': nll,
        'CRPS': crps,
        'ES': energy_score,
        'IS': interval_score
    }
