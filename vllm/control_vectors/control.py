from typing import Dict, Callable
import numpy as np
import warnings
import dataclasses
import torch

## questions: look over the control.py forward pass and see if I'm implementing it correctly in MixtralDecoderLayer
## it seems that the original code is wrapping each layer in a ControlModule class, which is then used to apply the control vector
## but I'm not sure if I need to do that, or if I can just apply the control vector directly to the layer.
## if yes, then I'll prolly need to take an approach similar to Lora?
@dataclasses.dataclass
class BlockControlParams:
    control: torch.Tensor = None
    normalize: bool = False
    operator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = (
        lambda current, control: current + control
    )

    @classmethod
    def default(cls) -> "BlockControlParams":
        return cls()
    

class ControlVector:
    model_type: str
    directions: Dict[int, np.ndarray]
    
    def __init__(self, model_type: str, directions: Dict[int, np.ndarray]) -> None:
        self.model_type = model_type
        self.directions = directions

    def _helper_combine(
        self, other: "ControlVector", other_coeff: float
    ) -> "ControlVector":
        if self.model_type != other.model_type:
            warnings.warn(
                "Trying to add vectors with mismatched model_types together, this may produce unexpected results."
            )

        model_type = self.model_type
        directions: dict[int, np.ndarray] = {}
        for layer in self.directions:
            directions[layer] = self.directions[layer]
        for layer in other.directions:
            other_layer = other_coeff * other.directions[layer]
            if layer in directions:
                directions[layer] = directions[layer] + other_layer
            else:
                directions[layer] = other_layer
        return ControlVector(model_type=model_type, directions=directions)

    def __add__(self, other: "ControlVector") -> "ControlVector":
        if not isinstance(other, ControlVector):
            raise TypeError(
                f"Unsupported operand type(s) for +: 'ControlVector' and '{type(other).__name__}'"
            )
        return self._helper_combine(other, 1)

    def __sub__(self, other: "ControlVector") -> "ControlVector":
        if not isinstance(other, ControlVector):
            raise TypeError(
                f"Unsupported operand type(s) for -: 'ControlVector' and '{type(other).__name__}'"
            )
        return self._helper_combine(other, -1)

    def __neg__(self) -> "ControlVector":
        directions: dict[int, np.ndarray] = {}
        for layer in self.directions:
            directions[layer] = -self.directions[layer]
        return ControlVector(model_type=self.model_type, directions=directions)

    def __mul__(self, other) -> "ControlVector":
        directions: dict[int, np.ndarray] = {}
        for layer in self.directions:
            directions[layer] = other * self.directions[layer]
        return ControlVector(model_type=self.model_type, directions=directions)

    def __rmul__(self, other) -> "ControlVector":
        return self.__mul__(other)

    def __truediv__(self, other) -> "ControlVector":
        return self.__mul__(1 / other)

    