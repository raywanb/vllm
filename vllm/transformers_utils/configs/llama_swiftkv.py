from typing import Dict, Optional

from transformers import LlamaConfig


class LlamaSwiftKVConfig(LlamaConfig):
    """
    Configuration class for Llama models with SwiftKV support.
    
    SwiftKV enables KV cache sharing across layers, reducing memory usage
    and computation. Layers specified in kv_sharing_map will reuse KV caches
    from other layers and use separate q_proj_swiftkv weights for queries.
    
    Args:
        swiftkv (bool, optional):
            Whether to enable SwiftKV mode. Defaults to False.
        num_key_value_layers (int, optional):
            The number of layers, from the first layer, that have keys and
            values. If None, all layers have keys and values.
        kv_sharing_map (Dict[int, int], optional):
            Mapping from layer index to the layer index whose KV cache should be used.
            If provided, this overrides the simple num_key_value_layers behavior.
            Example: {4: 0, 5: 1} means layer 4 uses KV from layer 0, layer 5 uses KV from layer 1.
    """

    model_type = "llama_swiftkv"

    def __init__(
        self,
        swiftkv: bool = False,
        num_key_value_layers: Optional[int] = None,
        kv_sharing_map: Optional[Dict[int, int]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.swiftkv = swiftkv
        self.num_key_value_layers = (num_key_value_layers
                                     or self.num_hidden_layers)
        self.kv_sharing_map = kv_sharing_map or {}


