"""
Base class for XAI (eXplainable AI) methods in refrakt_xai.

This module defines the abstract interface for all XAI method implementations.
All custom XAI methods should inherit from BaseXAI and implement the explain method.

The base class stores the model and any additional parameters required for explanation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class BaseXAI:
    """
    Abstract base class for XAI methods.

    This class provides a standard interface for all XAI methods in refrakt_xai.
    Subclasses must implement the explain method to provide model explanations.

    Attributes:
        model: The model to be explained.
        params: Additional parameters for the XAI method.
    """

    model: Any
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Post-initialization hook for any setup needed after
        dataclass initialization.
        """
        # Override in subclasses if needed

    def explain(self, input_tensor: Any, target: Any = None, **kwargs: Any) -> Any:
        """
        Generate an explanation for the given input.

        Args:
            input_tensor: The input data for which to generate an explanation.
            target: Optional target label or index for explanation.
            **kwargs: Additional method-specific parameters.

        Returns:
            Explanation result (type depends on the specific XAI method).

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def get_model(self) -> Any:
        """
        Get the model being explained.

        Returns:
            The model object.
        """
        return self.model
