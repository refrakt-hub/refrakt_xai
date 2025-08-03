# Refrakt XAI

The explainability hook that powers Refrakt's powerful visualization and explainability component. `refrakt_xai` provides a unified interface for state-of-the-art Explainable AI (XAI) methods, enabling researchers and practitioners to understand and interpret their machine learning models.

## 🚀 Features

- **Unified XAI Interface**: Consistent API across all explanation methods
- **State-of-the-Art Methods**: Implementation of leading XAI techniques
- **PyTorch Integration**: Seamless integration with PyTorch models
- **Extensible Architecture**: Easy to add new explanation methods
- **Type Safety**: Full type annotations and mypy compliance
- **Comprehensive Testing**: 80%+ test coverage with 68 test cases

## 📦 Installation

Since `refrakt_xai` is part of the Refrakt ecosystem, you can install it in several ways:

### Step 1: Clone the repository. 
```bash
# Clone the repository
git clone https://github.com/refrakt-hub/refrakt_xai.git
cd refrakt_xai
```


### Step 2: Create a virtual environment
```bash
# Option A: Using uv (recommended)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Option B: Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Option C: Using conda
conda create -n refrakt_xai python=3.10
conda activate refrakt_xai
```

### Step 3: Install from Requirements
```bash
# Option A (with uv)
uv pip install -r pyproject.toml

# Option B (with pip)
pip install -r requirements.txt
```

### Dependencies

**Runtime Dependencies:**
- `torch` - PyTorch deep learning framework
- `captum` - Model interpretability library

**Development Dependencies:**
- `pytest` - Testing framework
- `pytest-cov` - Coverage reporting
- `coverage` - Coverage measurement
- `isort` - Import sorting
- `black` - Code formatting
- `pylint` - Code linting
- `ruff` - Fast Python linter
- `radon` - Code complexity analysis
- `lizard` - Code complexity analysis
- `mypy` - Type checking
- `pre-commit` - Git hooks

## 🏗️ Project Structure

```
refrakt_xai/
├── methods/                 # XAI method implementations
│   ├── saliency.py         # Gradient-based saliency maps
│   ├── integrated_gradients.py  # Integrated gradients
│   ├── layer_gradcam.py    # Layer-wise GradCAM
│   ├── occlusion.py        # Occlusion sensitivity
│   ├── deeplift.py         # DeepLift attribution
│   ├── tcav.py             # Testing with Concept Activation Vectors
│   └── reconstruction_attribution.py  # Reconstruction-based attribution
├── utils/                  # Utility functions
│   ├── model_utils.py      # Model validation and processing
│   ├── layer_detection.py  # Automatic layer detection
│   ├── layer_resolvers.py  # Layer path resolution
│   └── concept_utils.py    # Concept-based utilities
├── tests/                  # Comprehensive test suite
│   ├── methods/           # Method-specific tests
│   └── utils/             # Utility function tests
├── base.py                # Base XAI class interface
├── registry.py            # Method registration system
└── __init__.py           # Package initialization
```

## 🎯 Available XAI Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| **SaliencyXAI** | Gradient-based attribution maps | General model interpretation |
| **IntegratedGradientsXAI** | Path-integrated gradients | Robust attribution analysis |
| **LayerGradCAMXAI** | Layer-wise GradCAM | CNN visualization |
| **OcclusionXAI** | Occlusion sensitivity | Feature importance analysis |
| **DeepLiftXAI** | DeepLift attribution | Deep network interpretation |
| **TCAVXAI** | Concept activation vectors | Concept-based explanations |
| **ReconstructionAttributionXAI** | Reconstruction-based attribution | Autoencoder interpretation |

## 💻 Usage Examples

### Basic Usage

```python
import torch
import torch.nn as nn
from refrakt_xai import SaliencyXAI, IntegratedGradientsXAI, LayerGradCAMXAI

# Define a simple model
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 10)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Initialize model and input
model = SimpleCNN()
input_tensor = torch.randn(1, 3, 32, 32, requires_grad=True)

# Create XAI explanations
saliency = SaliencyXAI(model)
saliency_attributions = saliency.explain(input_tensor, target=0)

ig = IntegratedGradientsXAI(model)
ig_attributions = ig.explain(input_tensor, target=0)

gradcam = LayerGradCAMXAI(model, layer="conv")
gradcam_attributions = gradcam.explain(input_tensor, target=0)
```

### Advanced Usage with Custom Models

```python
from refrakt_xai import OcclusionXAI, DeepLiftXAI

# Occlusion analysis
occlusion = OcclusionXAI(model, window_size=8)
occlusion_attributions = occlusion.explain(input_tensor, target=0)

# DeepLift attribution
deeplift = DeepLiftXAI(model)
deeplift_attributions = deeplift.explain(input_tensor, target=0)

# Auto-detection of layers
auto_gradcam = LayerGradCAMXAI(model, layer="auto")
auto_attributions = auto_gradcam.explain(input_tensor, target=0)
```

### Batch Processing

```python
# Process multiple inputs
batch_input = torch.randn(4, 3, 32, 32, requires_grad=True)
batch_targets = [0, 1, 2, 3]

# Batch processing with individual targets
batch_attributions = saliency.explain(batch_input, target=batch_targets)

# Single target for entire batch
single_target_attributions = saliency.explain(batch_input, target=0)
```

### Custom Model Integration

```python
# Works with any PyTorch model
import torchvision.models as models

resnet = models.resnet18(pretrained=True)
resnet.eval()

# Layer-specific analysis
layer_gradcam = LayerGradCAMXAI(resnet, layer="layer4.1.conv2")
attributions = layer_gradcam.explain(input_tensor, target=0)
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTORS.md](CONTRIBUTORS.md) for detailed guidelines on:

- Setting up the development environment
- Code style and conventions
- Testing requirements
- Pull request process
- Adding new XAI methods

## 📚 Integration with Refrakt

`refrakt_xai` is designed as a core component of the Refrakt ecosystem, providing:

- **Natural Language Interface**: XAI methods can be invoked through Refrakt's NL orchestrator
- **Visualization Pipeline**: Attributions are automatically integrated with Refrakt's visualization system
- **Workflow Integration**: Seamless integration with Refrakt's ML/DL workflow orchestration
- **Scalability**: Methods are optimized for large-scale model analysis

## 📄 License

This project is licensed under the same license as the main Refrakt project. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Built on top of [Captum](https://captum.ai/) for robust XAI implementations
- Inspired by the XAI research community
- Part of the Refrakt ecosystem for scalable ML/DL workflows

---

**Part of the [Refrakt](https://github.com/refrakt-hub/refrakt) ecosystem** - Natural-language orchestrator for scalable ML/DL workflows. [COMING SOON]
