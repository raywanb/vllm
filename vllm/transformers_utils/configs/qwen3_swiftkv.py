from typing import Dict, Optional

from transformers import Qwen3Config


class Qwen3SwiftKVConfig(Qwen3Config):
    """
    Args:
        num_key_value_layers (int, optional):
            The number of layers, from the first layer, that have keys and
            values. If None, all layers have keys and values.
        kv_sharing_map (Dict[int, int], optional):
            Mapping from layer index to the layer index whose KV cache should be used.
            If provided, this overrides the simple num_key_value_layers behavior.
    """

    model_type = "qwen3_swiftkv"

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
