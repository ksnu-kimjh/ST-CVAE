from model.ST_CVAE import ST_CVAE_Wrapper
from model.ST_CVAE_CQR import ST_CVAE_CQR_Wrapper

# Factory function
def create_model_instance(model_name, params, n_targets):
    mapping = {
        "ST_CVAE": ST_CVAE_Wrapper, "ST_CVAE_CQR": ST_CVAE_CQR_Wrapper
    }

    if model_name not in mapping:
        raise ValueError(f"Unknown model name: {model_name}")
    
    model_cls = mapping[model_name]
    
    native_generative_models = ["ST_CVAE", "ST_CVAE_CQR"]

    if model_name in native_generative_models:
        params['d_out'] = n_targets
        return model_cls(**params)
