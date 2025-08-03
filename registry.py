"""
XAI method registry for refrakt_xai.

This module provides a registry and decorators for registering and retrieving XAI
methods by name. It enables dynamic lookup and extension of available XAI methods
in the refrakt_xai package.

Typical usage:
    @register_xai("my_method")
    class MyXAI(BaseXAI):
        ...
    method_cls = get_xai("my_method")
"""

from typing import Any, Callable, Dict, Type

XAI_REGISTRY: Dict[str, Type[Any]] = {}
"""
Global registry mapping XAI method names to their corresponding classes.
"""


def register_xai(name: str) -> Callable[[Type[Any]], Type[Any]]:
    """
    Decorator to register an XAI method class or function under a given name.

    Args:
        name: The name to register the XAI method under.

    Returns:
        A decorator that registers the class/function in the XAI_REGISTRY.
    """

    def decorator(cls: Type[Any]) -> Type[Any]:
        if name in XAI_REGISTRY:
            print(f"Warning: XAI method '{name}' already registered. Overwriting.")
        XAI_REGISTRY[name] = cls
        return cls

    return decorator


def get_xai(name: str) -> Type[Any]:
    """
    Retrieve an XAI method class or function by name.

    Args:
        name: The name of the registered XAI method.

    Returns:
        The XAI method class or function associated with the given name.

    Raises:
        ValueError: If the name is not found in the registry.
    """
    if name not in XAI_REGISTRY:
        available = list(XAI_REGISTRY.keys())
        raise ValueError(f"XAI method '{name}' not found. Available: {available}")
    return XAI_REGISTRY[name]
