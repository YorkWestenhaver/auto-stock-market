import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseActivation(nn.Module):
    """
    Base class for activation functions to ensure consistent interface
    and documentation standards.
    """
    def __init__(self):
        super().__init__()

    def extra_repr(self) -> str:
        """Returns a string with extra information about the module."""
        return ""

class SigmoidActivation(BaseActivation):
    """
    Sigmoid Activation Function

    Equation:
    f(x) = 1 / (1 + e^-x)

    Note: While this implementation exists for completeness, consider using
    torch.nn.Sigmoid directly in production code.
    """
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(input)

class TanhActivation(BaseActivation):
    """
    Hyperbolic Tangent Activation Function

    Equation:
    f(x) = (1 - e^-2x) / (1 + e^-2x)

    Note: While this implementation exists for completeness, consider using
    torch.nn.Tanh directly in production code.
    """
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.tanh(input)

class ReLUActivation(BaseActivation):
    """
    Rectified Linear Unit Activation Function

    Equation:
    f(x) = max(0, x)

    Note: While this implementation exists for completeness, consider using
    torch.nn.ReLU directly in production code.
    """
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.relu(input)

class LeakyReLUActivation(BaseActivation):
    """
    Leaky ReLU Activation Function

    Equation:
    f(x) = 0.01x if x ≤ 0
           x     if x > 0

    Note: While this implementation exists for completeness, consider using
    torch.nn.LeakyReLU directly in production code.
    """
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(input, negative_slope=self.negative_slope)

    def extra_repr(self) -> str:
        return f'negative_slope={self.negative_slope}'

class PReLUActivation(nn.PReLU):
    """
    Parametric ReLU Activation Function

    Equation:
    f(x) = ax if x ≤ 0
           x  if x > 0

    where 'a' is a learnable parameter.

    Note: This is a thin wrapper around torch.nn.PReLU for consistency
    with our activation interface.
    """
    pass

class SwishActivation(BaseActivation):
    """
    Swish Activation Function

    Equation:
    f(x) = x * sigmoid(βx)

    where β is a learnable or fixed parameter (default=1)
    """
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * torch.sigmoid(self.beta * input)

    def extra_repr(self) -> str:
        return f'beta={self.beta}'

class MishActivation(BaseActivation):
    """
    Mish Activation Function

    Equation:
    f(x) = x * tanh(softplus(x))
    """
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * torch.tanh(F.softplus(input))

class GCUActivation(BaseActivation):
    """
    Growing Cosine Unit Activation Function

    Equation:
    f(x) = x * cos(x)
    """
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * torch.cos(input)

# Optional: Implementation of the biologically inspired oscillating activation functions
class NCUActivation(BaseActivation):
    """
    Neural Cosine Unit Activation Function
    A variant of GCU with normalized frequency
    """
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * torch.cos(torch.pi * input)

class SQUActivation(BaseActivation):
    """
    Squared Unit Activation Function
    Combines quadratic and oscillatory behaviors
    """
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * input * torch.cos(input)

class DSUActivation(BaseActivation):
    """
    Damped Sine Unit Activation Function
    Combines dampening with sinusoidal oscillation
    """
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.exp(-0.5 * input * input) * torch.sin(input)

class SSUActivation(BaseActivation):
    """
    Smooth Sine Unit Activation Function
    Combines smooth scaling with sinusoidal behavior
    """
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * torch.sin(torch.sigmoid(input))