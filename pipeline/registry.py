"""AutoPARK Model & Task Registry.

Provides a pluggable model registry enabling ML engineers to register custom
PyTorch architectures, Hugging Face models, or TorchVision backbones via a simple
decorator (@register_model) or factory function, eliminating hardcoded model dictionaries.
"""
from __future__ import annotations

from typing import Any, Callable
import torch
import torch.nn as nn


_MODEL_REGISTRY: dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str) -> Callable[[Callable[..., nn.Module]], Callable[..., nn.Module]]:
    """Decorator to register a model factory function with AutoPARK.
    
    Usage:
        @register_model("my_custom_vit")
        def build_my_vit(num_classes=10, **kwargs):
            return MyViT(num_classes=num_classes)
    """
    def decorator(fn: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        canonical_name = name.lower().strip()
        _MODEL_REGISTRY[canonical_name] = fn
        return fn
    return decorator


def get_model(name: str, **kwargs: Any) -> nn.Module:
    """Retrieve and instantiate a registered model by name.
    
    Args:
        name: Canonical model name (case-insensitive).
        **kwargs: Arguments forwarded to the builder (e.g. num_classes).
        
    Returns:
        Instantiated nn.Module instance.
    """
    canonical_name = name.lower().strip()
    if canonical_name not in _MODEL_REGISTRY:
        # Attempt deferred import of built-in models
        _register_builtin_models()

    if canonical_name not in _MODEL_REGISTRY:
        available = list_models()
        raise KeyError(
            f"Model '{name}' not found in AutoPARK model registry. "
            f"Available models: {available}. Register custom models with @register_model('{name}')."
        )

    builder = _MODEL_REGISTRY[canonical_name]
    return builder(**kwargs)


def list_models() -> list[str]:
    """Return all registered model names."""
    _register_builtin_models()
    return sorted(list(_MODEL_REGISTRY.keys()))


def _register_builtin_models() -> None:
    """Lazily registers built-in models to avoid import cycles."""
    if "resnet18" not in _MODEL_REGISTRY:
        try:
            from models.resnet import build_resnet18
            _MODEL_REGISTRY["resnet18"] = build_resnet18
        except ImportError:
            pass

    if "tiny" not in _MODEL_REGISTRY:
        try:
            from models.tiny_cnn import build_tiny_cnn
            _MODEL_REGISTRY["tiny"] = build_tiny_cnn
        except ImportError:
            pass

    if "mlp" not in _MODEL_REGISTRY:
        def build_simple_mlp(num_classes: int = 10, in_features: int = 3072, hidden: int = 256, **kwargs) -> nn.Module:
            return nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features, hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(hidden, num_classes),
            )
        _MODEL_REGISTRY["mlp"] = build_simple_mlp

    if "simple_cnn" not in _MODEL_REGISTRY:
        def build_simple_cnn(num_classes: int = 10, in_channels: int = 3, **kwargs) -> nn.Module:
            return nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(32 * 4 * 4, num_classes),
            )
        _MODEL_REGISTRY["simple_cnn"] = build_simple_cnn


# Initialize built-in registrations
_register_builtin_models()
