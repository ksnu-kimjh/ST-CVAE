import argparse
import os
import datetime
import optuna
import pandas as pd
import torch
import numpy as np
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

from utils import (seed_everything, load_config, load_data, feature_engineering,
                   strat_split_data, sampling_stratified, calculate_metrics)
from model_factory import create_model_instance

def objective(trial, model_name, config, X_t, y_t, X_v, y_v):
    model_conf = config['models'][model_name]
    search_space = model_conf.get('search_space', {})

    params = {}
    for k, v in search_space.items():
        if v['type'] == 'int':
            step = v.get('step', 1)
            params[k] = trial.suggest_int(k, int(v['low']), int(v['high']), step=int(step))
        elif v['type'] == 'float':
            params[k] = trial.suggest_float(k, float(v['low']), float(v['high']), log=v.get('log', False))
        elif v['type'] == 'categorical':
            params[k] = trial.suggest_categorical(k, v['choices'])
    
    full_params = config['common_model_params'].copy()
    full_params.update(model_conf.get('fixed_params', {}))
    full_params.update(params)

    full_params['use_native_multitarget'] = model_conf.get('use_native_multitarget', False)
    full_params['numerical_cols'] = config['data']['numerical_cols']

    full_params['categorical_cols'] = config['data']['categorical_cols']
    full_params['spatial_cols'] = config['data'].get('spatial_cols', [])

    n_targets = y_t.shape[1]

    try:
        model = create_model_instance(model_name, full_params, n_targets)
        model.fit(X_t, y_t, eval_set=(X_v, y_v))
        score = model.evaluate_objective(X_v, y_v)
        return score
    except Exception as e:
        print(f"[Optuna Error] Trial failed: {e}")
        return float('inf')

def generate_samples_from_dist(dist_params, n_samples=50):
    if dist_params is None:
        return None
        
    if isinstance(dist_params, np.ndarray):
        return dist_params
    
    if isinstance(dist_params, tuple):
        pi, sigma, mu = dist_params
        # Convert to numpy if tensor
        def to_np(x): return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
        pi = to_np(pi)
        sigma = to_np(sigma)
        mu = to_np(mu)
        
        B, N, K = pi.shape
        y_samples = np.zeros((B, n_samples, N))
        
        for i in range(B):
            for j in range(N):
                # Select component for each sample
                k_indices = np.random.choice(K, size=n_samples, p=pi[i, j])
                # Sample from Normal
                means = mu[i, j, k_indices]
                stds = sigma[i, j, k_indices]
                y_samples[i, :, j] = np.random.normal(means, stds)
                
        return y_samples

    elif isinstance(dist_params, dict) and dist_params.get('type') == 'gaussian':
        mu = dist_params['mu']
        sigma = dist_params['sigma']
        B, N = mu.shape
        
        mu_exp = mu[:, np.newaxis, :]
        sigma_exp = sigma[:, np.newaxis, :]
        
        y_samples = np.random.normal(mu_exp, sigma_exp, size=(B, n_samples, N))
        return y_samples

    elif isinstance(dist_params, list):
        samples_list = []
        for dp in dist_params:
            if dp is None: return None
            if isinstance(dp, dict) and dp.get('type') == 'gaussian':
                mu = dp['mu']
                sigma = dp['sigma']
                if hasattr(mu, 'values'): mu = mu.values
                if hasattr(sigma, 'values'): sigma = sigma.values

                s = np.random.normal(mu[:, np.newaxis], sigma[:, np.newaxis], size=(len(mu), n_samples))
                samples_list.append(s)
            else:
                return None
        
        return np.stack(samples_list, axis=2)

    return None

def extract_history(model):
    if hasattr(model, 'history'):
        return model.history

    if hasattr(model, 'model') and hasattr(model.model, 'history'):
        return model.model.history

    if hasattr(model, 'models'):
        combined_history = {}
        for i, m in enumerate(model.models):
            if hasattr(m, 'history'):
                combined_history[f'target_{i}'] = m.history
        return combined_history if combined_history else None
    return None

def save_model_weights(model, save_path_pt, save_path_pkl):
    try:
        if hasattr(model, 'model') and isinstance(model.model, torch.nn.Module):
            torch.save(model.model.state_dict(), save_path_pt)
            print(f"[Saved] Best model (Torch) saved to: {save_path_pt}")
            return

        if hasattr(model, 'model') and hasattr(model.model, 'model') and isinstance(model.model.model, torch.nn.Module):
            torch.save(model.model.model.state_dict(), save_path_pt)
            print(f"[Saved] Best model (Torch Native) saved to: {save_path_pt}")
            return
            
        if hasattr(model, 'models') and len(model.models) > 0:
            # Check first model kind
            first_m = model.models[0]
            if hasattr(first_m, 'model') and isinstance(first_m.model, torch.nn.Module):
                # Save dict of state_dicts
                full_state = {f'target_{i}': m.model.state_dict() for i, m in enumerate(model.models)}
                torch.save(full_state, save_path_pt)
                print(f"[Saved] Best model (Multi-Target Torch) saved to: {save_path_pt}")
                return

        joblib.dump(model, save_path_pkl)
        print(f"[Saved] Best model (Pickle) saved to: {save_path_pkl}")

    except Exception as e:
        print(f"[Error] Failed to save model: {e}")

def run_experiment(model_name, config_path="settings.yaml"):
    config = load_config(config_path)
    seed_everything(config['pipeline']['seed'])

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    df = load_data(config)
    df = feature_engineering(df)

    target_cols = config['data'][f"targets_{config['pipeline']['target_type'].lower()}"]
    print(f"Targets: {target_cols}")

    tr_val_df, test_df = strat_split_data(df, config['pipeline']['test_size'])

    best_params = {}
    base_fixed = config['models'][model_name].get('fixed_params', {}).copy()

    if config['pipeline']['use_optuna']:
        print("Starting Optuna Optimization...")

        optuna_df = sampling_stratified(tr_val_df, config['pipeline']['sampling_size'])
        tr_opt, va_opt = strat_split_data(optuna_df, config['pipeline']['optuna_test_size'])

        X_t_opt = tr_opt[config['data']['numerical_cols'] + config['data']['categorical_cols']]
        y_t_opt = tr_opt[target_cols]
        X_v_opt = va_opt[config['data']['numerical_cols'] + config['data']['categorical_cols']]
        y_v_opt = va_opt[target_cols]

        study = optuna.create_study(direction='minimize')
        study.optimize(
            lambda t: objective(t, model_name, config, X_t_opt, y_t_opt.values, X_v_opt, y_v_opt.values), n_trials=config['pipeline']['n_optuna_trials'])

        print(f"Best Optuna Params: {study.best_params}")
        best_params = study.best_params
    else:
        print("Optuna disabled. Using fixed parameters from settings.yaml only.")

    print("Training Final Model...")
    X_train = tr_val_df[config['data']['numerical_cols'] + config['data']['categorical_cols']]
    y_train = tr_val_df[target_cols]
    X_test = test_df[config['data']['numerical_cols'] + config['data']['categorical_cols']]
    y_test = test_df[target_cols]

    final_params = config['common_model_params'].copy()
    final_params.update(base_fixed)
    final_params.update(best_params)
    final_params['use_native_multitarget'] = config['models'][model_name].get('use_native_multitarget', False)
    final_params['mdn_pred_method'] = 'mean'
    final_params['numerical_cols'] = config['data']['numerical_cols']
    final_params['categorical_cols'] = config['data']['categorical_cols']
    final_params['spatial_cols'] = config['data'].get('spatial_cols', [])

    n_targets = y_train.shape[1]
    model = create_model_instance(model_name, final_params, n_targets)
    model.fit(X_train, y_train.values, eval_set=(X_test, y_test.values))

    print("Evaluating...")
    preds = model.predict(X_test)
    intervals = model.predict_interval(X_test)

    if intervals.ndim == 2 and n_targets == 1:
        intervals = intervals[:, np.newaxis, :]
    
    if intervals.ndim == 3 and intervals.shape[1] != n_targets:
        print(f"[Warning] Interval shape mismatch! Expected {n_targets} targets, got {intervals.shape[1]}. Check model implementation.")
        n_targets = min(n_targets, intervals.shape[1])

    dist_params_list = model.get_dist_params(X_test)

    y_train_ranges = y_train.max() - y_train.min()
    results = {
        'Model': model_name,
        'Date': timestamp
    }

    # Save hyperparameters
    for k, v in final_params.items():
        if isinstance(v, (int, float, str, bool)):
            results[f"Param_{k}"] = v
        else:
            results[f"Param_{k}"] = str(v)
    
    loop_limit = min(len(target_cols), n_targets)
    
    y_samples_all = generate_samples_from_dist(dist_params_list, n_samples=50) # (B, M, N)
    
    for i in range(loop_limit):
        tgt = target_cols[i]
        
        dp = None
        if dist_params_list is not None:
             if isinstance(dist_params_list, tuple):
                  # GMM
                  try:
                        pi, sigma, mu = [x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x for x in dist_params_list]
                        dp = {'type': 'gmm', 'pi': pi[:, i, :], 'mu': mu[:, i, :], 'sigma': sigma[:, i, :]}
                  except: dp = None
             elif isinstance(dist_params_list, dict) and dist_params_list.get('type') == 'gaussian':
                  # Gaussian
                  try:
                        dp = {'type': 'gaussian', 'mu': dist_params_list['mu'][:, i], 'sigma': dist_params_list['sigma'][:, i]}
                  except: dp = None
             elif isinstance(dist_params_list, list):
                  try:
                      dp = dist_params_list[i]
                  except: dp = None

        y_samp_tgt = None
        if y_samples_all is not None:
            y_samp_tgt = y_samples_all[:, :, i] # (B, M)
            y_samp_tgt = y_samp_tgt[:, :, np.newaxis]

        # Calculate metrics
        try:
            metrics = calculate_metrics(
                y_true=y_test.iloc[:, i].values,
                y_pred=preds[:, i],
                y_lower=intervals[:, i, 0],
                y_upper=intervals[:, i, 1],
                dist_params=dp,
                y_train_range=y_train_ranges[tgt],
                y_samples=y_samp_tgt,
                alph=1.0 - (final_params['quantiles'][1] - final_params['quantiles'][0])
            )

            for m_k, m_v in metrics.items():
                results[f"{tgt}_{m_k}"] = m_v
            
            print(f"[{tgt}] RMSE: {metrics['RMSE']:.4f}, R2: {metrics['R2']:.4f}, PICP: {metrics['PICP']:.4f}, NMPIW: {metrics['NMPIW']:.4f}, ES: {metrics['ES']:.4f}, IS: {metrics['IS']:.4f}")
        except Exception as e:
            print(f"[Error] Failed metric calculation for target {tgt}: {e}")

    # Save to .csv
    out_dir = os.path.join(config['project']['output_dir'], datetime.datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(out_dir, exist_ok=True)

    filename = f"{model_name}_{timestamp}.csv"
    save_path = os.path.join(out_dir, filename)

    res_df = pd.DataFrame([results])
    res_df.to_csv(save_path, index=False)
    print(f"Results saved to: {save_path}")

    # --- Save Best Model & History & Predictions for Visualization ---
    model_save_name = f"{model_name}_{datetime.datetime.now().strftime('%Y%m%d')}_{timestamp.split('_')[1]}_best" # suffix added below
    model_save_path_pt = os.path.join(out_dir, model_save_name + ".pt")
    model_save_path_pkl = os.path.join(out_dir, model_save_name + ".pkl")
    
    # Save Model Weights
    save_model_weights(model, model_save_path_pt, model_save_path_pkl)

    history_save_name = f"{model_name}_{timestamp}_history.json"
    history_save_path = os.path.join(out_dir, history_save_name)
    try:
        history = extract_history(model)
        if history:
            def default_json(obj):
                if hasattr(obj, 'tolist'): return obj.tolist()
                return str(obj)
                
            with open(history_save_path, 'w') as f:
                json.dump(history, f, default=default_json)
            print(f"[Saved] Training history saved to: {history_save_path}")
        else:
            print("[Warning] No history found to save.")
    except Exception as e:
        print(f"[Error] Failed to save history: {e}")

    pred_save_name = f"{model_name}_{timestamp}_preds.npz"
    pred_save_path = os.path.join(out_dir, pred_save_name)
    
    try:
        data_to_save = {}
        for i in range(min(n_targets, 3)):
            data_to_save[f'target_{i}_true'] = y_test.iloc[:, i].values
            data_to_save[f'target_{i}_pred'] = preds[:, i]
            data_to_save[f'target_{i}_lower'] = intervals[:, i, 0]
            data_to_save[f'target_{i}_upper'] = intervals[:, i, 1]
            if y_samples_all is not None:
                data_to_save[f'target_{i}_samples'] = y_samples_all[:, :, i]

        np.savez(pred_save_path, **data_to_save)
        print(f"[Saved] Predictions for visualization saved to: {pred_save_path}")
    except Exception as e:
         print(f"[Error] Failed to save predictions: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model Name (e.g., ST_CVAE, ST_CVAE_CQR)")
    parser.add_argument("--config", default="settings.yaml", help="Path to config file")

    args = parser.parse_args()

    run_experiment(args.model, args.config)
