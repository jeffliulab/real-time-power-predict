"""Slim model registry for the HF Space (baseline only)."""

from .cnn_transformer_baseline import CNNTransformerBaselineForecaster

MODEL_REGISTRY = {
    "cnn_transformer_baseline": CNNTransformerBaselineForecaster,
    "cnn_transformer": CNNTransformerBaselineForecaster,  # legacy alias
}


def create_model(name, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)
