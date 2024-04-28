from vllm.control_vectors.control import ControlVector
from vllm.control_vectors.layers import MLPWithControlVector, BaseLayerWithControlVector
from vllm.config import ControlVectorConfig
from vllm.control_vectors.request import ControlVectorAdapter
from vllm.lora.utils import replace_submodule
from typing import Optional, Dict, Set, Type
import torch
import torch.nn as nn


_all_cv_classes: Set[Type[BaseLayerWithControlVector]] = {
    MLPWithControlVector
}

class ControlVectorManager:
    """A manager that manages multiple control vectors for a model."""

    def __init__(
        self,
        model: nn.Module,
        control_vector_config: ControlVectorConfig,
    ):
        self.model = model
        self.control_vector_config = control_vector_config
        self._registered_control_vectors = {}
        self._active_control_vectors = {}

    def _create_cv_modules(self):
        for module_name, module in self.model.named_modules():
            if not module_name.contains("mlp"):
                continue
            new_module = replace_submodule(self.model, module_name, MLPWithControlVector(module))
            self.register_module(module_name, new_module)
    
    def register_module(self, module_name: str, module: "BaseLayerWithControlVector"):
        assert isinstance(module, BaseLayerWithControlVector)
        self.modules[module_name] = module
    
    def add_control_vector_adapter(self, adapter):
        assert isinstance(adapter, ControlVectorAdapter), "Adapter must be an instance of ControlVectorAdapter"
        self._registered_control_vectors[adapter.name] = adapter

    def remove_control_vector_adapter(self, adapter_name):
        if adapter_name in self._registered_control_vectors:
            del self._registered_control_vectors[adapter_name]

    def get_control_vector_adapter(self, adapter_name):
        return self._registered_control_vectors.get(adapter_name)


def create_cv_manager(
        model: nn.Module,
        control_vector_config: ControlVectorConfig,
        **kwargs) -> ControlVectorManager:
    """Create a ControlVectorManager for a given model."""
    cv_manager = ControlVectorManager(
        model=model,
        control_vector_config=control_vector_config,
        **kwargs)
    return cv_manager
