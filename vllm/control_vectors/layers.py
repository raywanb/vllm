import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.control_vectors.control import ControlVector
from vllm.config import ControlVectorConfig
        
def _apply_control_vector(
    x: torch.Tensor,
    control_vector: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
):
    """Applies a single array of control vectors to the input tensor.

    This method applies a single array of control vectors to each input. It uses the
    indices vector to determine which control vector within the array yields the
    correct output. An index of -1 means no control vector should be
    applied. This method adds the final control vector results to the
    output.

    Input shapes:
        x:                 (batch_size, hidden_dim)
        control_vector:    (num_vectors, hidden_dim, output_dim)
        indices:           (batch_size)
        output:            (batch_size, output_dim)
    """
    org_output = output
    x = x.view(-1, x.shape[-1])
    output = output.view(-1, output.shape[-1])
    indices = indices.view(-1)

    # Apply control vectors
    for i in range(len(indices)):
        if indices[i] >= 0:
            cv = control_vector[indices[i]]
            output[i] += F.linear(x[i].unsqueeze(0), cv).squeeze(0)

    return output.view_as(org_output)


class ControlModule(nn.Module):

    def __init__(self, control_vector: ControlVector):
        super().__init__()
        self.control_vector = control_vector


    def reset_control_vectors(self):
        """Reset control vectors to zero, effectively disabling them."""
        if self.control_vectors is not None:
            with torch.no_grad():
                self.control_vectors.fill_(0)

    def set_control_vectors(self, index: int, cv: torch.Tensor):
        """Directly set a specific control vector."""
        if 0 <= index < self.control_vectors.size(0):
            self.control_vectors[index] = cv

    def apply_control_vectors(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Applies control vectors to the input tensor according to indices."""
        output = torch.zeros_like(x)
        if self.control_vectors is not None:
            _apply_control_vector(x, self.control_vectors, indices, output)
        return output
    
