"""
Model registry for energy demand forecasting architectures.

Usage:
    from models import create_model, MODEL_REGISTRY
    model = create_model("cnn_transformer", history_len=24)
"""

from .cnn_transformer import CNNTransformerForecaster
from .cnn_encoder_decoder import CNNEncoderDecoderForecaster

MODEL_REGISTRY = {
    "cnn_transformer": CNNTransformerForecaster,
    "cnn_encoder_decoder": CNNEncoderDecoderForecaster,
}

MODEL_DEFAULTS = {
    "cnn_transformer": {
        "history_len": 24,
        "embed_dim": 128,
        "grid_size": 8,
        "n_layers": 4,
        "n_heads": 4,
    },
    "cnn_encoder_decoder": {
        "history_len": 24,
        "embed_dim": 128,
        "grid_size": 8,
        "n_encoder_layers": 4,
        "n_decoder_layers": 2,
        "n_heads": 4,
    },
}


def create_model(name, **kwargs):
    """Instantiate a model by name with given kwargs."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)


def get_model_defaults(name):
    """Return default hyperparameters for a model."""
    return MODEL_DEFAULTS.get(name, {})
