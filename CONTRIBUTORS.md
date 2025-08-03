# CONTRIBUTORS.md

Thank you for your interest in contributing to Refrakt XAI! This document provides comprehensive guidelines for contributing to the project.

## 🛠️ Development Setup

### Prerequisites

- Python 3.10 or higher
- Git
- pip (latest version)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/refrakt/refrakt.git
   cd refrakt/external/refrakt_xai
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode with all dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

### Development Dependencies

The following dependencies are required for development:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",           # Testing framework
    "pytest-cov>=6.0.0",       # Coverage reporting
    "coverage>=7.0.0",         # Coverage measurement
    "isort>=5.0.0",           # Import sorting
    "black>=24.0.0",          # Code formatting
    "pylint>=3.0.0",          # Code linting
    "ruff>=0.4.0",            # Fast Python linter
    "radon>=6.0.0",           # Code complexity analysis
    "lizard>=1.17.0",         # Code complexity analysis
    "mypy>=1.0.0",            # Type checking
    "pre-commit>=3.0.0",      # Git hooks
]
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=refrakt_xai --cov-report=term-missing

# Run specific test categories
pytest tests/methods/
pytest tests/utils/

# Run with verbose output
pytest -v

# Run with parallel execution
pytest -n auto
```

### Coverage Requirements

- **Minimum coverage**: 80%
- **Current coverage**: 80.25%
- **Coverage reports**: Generated in `htmlcov/` directory

### Test Structure

```
tests/
├── methods/                 # XAI method tests
│   ├── test_saliency.py
│   ├── test_integrated_gradients.py
│   ├── test_layer_gradcam.py
│   ├── test_occlusion.py
│   ├── test_deeplift.py
│   ├── test_tcav.py
│   └── test_reconstruction_attribution.py
└── utils/                   # Utility function tests
    ├── test_model_utils.py
    ├── test_layer_detection.py
    ├── test_layer_resolvers.py
    └── test_concept_utils.py
```

## 🔍 Code Quality

### Automated Checks

The project uses several tools to maintain code quality:

```bash
# Type checking
mypy . --exclude tests/

# Linting
pylint . --ignore=tests/

# Formatting
black .
isort .

# Complexity analysis
radon cc refrakt_xai/
lizard refrakt_xai/
```

### Quality Standards

- **Pylint score**: Minimum 9.4/10 (current: 10.00/10)
- **Mypy**: 0 errors
- **Black**: Consistent code formatting
- **Isort**: Organized imports
- **Line length**: 88 characters maximum

### Pre-commit Hooks

The following hooks run automatically on commit:

- **black**: Code formatting
- **isort**: Import sorting
- **pylint**: Code linting
- **coverage**: Coverage checking (80% minimum)
- **radon**: Cyclomatic complexity
- **lizard**: Code complexity
- **pytest**: Test execution

## 📝 Adding New XAI Methods

### Method Structure

New XAI methods should follow this structure:

```python
"""
[Method Name] XAI method for refrakt_xai.

This module implements the [Method Name] method using [Library],
providing [description] for model predictions.

Typical usage:
    xai = [MethodName]XAI(model)
    attributions = xai.explain(input_tensor, target=target_class)
"""

from dataclasses import dataclass
from typing import Any, Optional

from torch import Tensor

from refrakt_xai.base import BaseXAI
from refrakt_xai.registry import register_xai


@register_xai("method_name")
@dataclass
class [MethodName]XAI(BaseXAI):
    """
    [Method Name] XAI method using [Library].

    [Description of what the method does].

    Attributes:
        model: The model to be explained.
        [other_attributes]: [descriptions]
    """

    # Add method-specific attributes here

    def __post_init__(self) -> None:
        """Initialize after dataclass initialization."""
        # Add initialization logic here
        pass

    def explain(
        self, input_tensor: Tensor, target: Optional[int] = None, **kwargs: Any
    ) -> Tensor:
        """
        Generate [method] attributions for the given input and target.

        Args:
            input_tensor: Input tensor for which to compute attributions.
            target: Optional target class index for explanation.
            **kwargs: Additional parameters.

        Returns:
            Tensor of attributions with the same shape as input_tensor.
        """
        # Implement the explanation logic here
        pass
```

### Testing Requirements

Each new method must include:

1. **Smoke tests**: Basic functionality verification
2. **Unit tests**: Individual component testing
3. **Edge case tests**: Error handling and boundary conditions
4. **Integration tests**: End-to-end functionality

### Pull Request Guidelines

1. **Title**: Clear, descriptive title
2. **Description**: Detailed description of changes
3. **Tests**: Include new tests for new functionality
4. **Documentation**: Update docstrings and README if needed
5. **Type annotations**: Ensure all functions are properly typed
6. **Coverage**: Maintain or improve test coverage

### Commit Message Format

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions or changes
- `chore`: Maintenance tasks

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Environment details**: Python version, OS, package versions
2. **Reproduction steps**: Clear steps to reproduce the issue
3. **Expected behavior**: What you expected to happen
4. **Actual behavior**: What actually happened
5. **Error messages**: Full error traceback
6. **Code example**: Minimal code to reproduce the issue

## 💡 Feature Requests

When requesting features, please include:

1. **Use case**: Why this feature is needed
2. **Proposed solution**: How you envision the feature working
3. **Alternatives**: Any alternative approaches considered
4. **Impact**: How this affects existing functionality

## 📚 Documentation

### Code Documentation

- **Docstrings**: Use Google-style docstrings
- **Type hints**: Include type annotations for all functions
- **Examples**: Provide usage examples in docstrings

### API Documentation

- **Method signatures**: Clear parameter descriptions
- **Return values**: Detailed return type information
- **Exceptions**: Document all possible exceptions

## 🤝 Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Help others learn and grow
- Follow the project's coding standards

### Communication

- **Issues**: Use GitHub issues for bug reports and feature requests
- **Discussions**: Use GitHub discussions for general questions
- **Pull Requests**: Use PRs for code contributions

## 🏆 Recognition

Contributors will be recognized in:

- **README.md**: For significant contributions
- **Release notes**: For each release
- **Contributors list**: GitHub contributors page

## 📞 Getting Help

- **Documentation**: Check the README.md and docstrings
- **Issues**: Search existing issues for similar problems
- **Discussions**: Start a discussion for general questions
- **Code**: Review existing code for examples

---