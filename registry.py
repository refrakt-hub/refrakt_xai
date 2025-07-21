"""
XAI method registry for refrakt_xai.
"""

from typing import Any, Callable, Dict, Type

XAI_REGISTRY: Dict[str, Type[Any]] = {}


def register_xai(name: str) -> Callable[[Type[Any]], Type[Any]]:
    """Decorator to register an XAI method class or function under a given name."""
    def decorator(cls: Type[Any]) -> Type[Any]:
        if name in XAI_REGISTRY:
            print(f"Warning: XAI method '{name}' already registered. Overwriting.")
        XAI_REGISTRY[name] = cls
        return cls
    return decorator


def get_xai(name: str) -> Type[Any]:
    """Retrieve an XAI method class or function by name."""
    if name not in XAI_REGISTRY:
        available = list(XAI_REGISTRY.keys())
        raise ValueError(f"XAI method '{name}' not found. Available: {available}")
    return XAI_REGISTRY[name] 