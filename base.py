"""
Base class for XAI (eXplainable AI) methods in refrakt_xai.

This module defines the abstract interface for all XAI method implementations.
All custom XAI methods should inherit from BaseXAI and implement the explain method.

The base class stores the model and any additional parameters required for explanation.
"""

from typing import Any, Optional


class BaseXAI:
    """
    Abstract base class for XAI methods.

    This class provides a standard interface for all XAI methods in refrakt_xai.
    Subclasses must implement the explain method to provide model explanations.

    Attributes:
        model: The model to be explained.
        params: Additional parameters for the XAI method.
    """

    def __init__(self, model: Any, **kwargs: Any) -> None:
        """
        Initialize the XAI method with a model and optional parameters.

        Args:
            model: The model to be explained.
            **kwargs: Additional parameters for the XAI method.
        """
        self.model = model
        self.params = kwargs

    def explain(
        self, input_tensor: Any, target: Optional[Any] = None, **kwargs: Any
    ) -> Any:
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
