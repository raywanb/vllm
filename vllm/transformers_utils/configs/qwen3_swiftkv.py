from typing import Optional

from transformers import Qwen3Config


class Qwen3SwiftKVConfig(Qwen3Config):
    """
    Args:
        num_key_value_layers (int, optional):
            The number of layers, from the first layer, that have keys and
            values. If None, all layers have keys and values.
    """

    model_type = "qwen3_swiftkv"

    def __init__(
        self,
        swiftkv: bool = False,
        num_key_value_layers: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.swiftkv = swiftkv
        self.num_key_value_layers = (num_key_value_layers
                                     or self.num_hidden_layers)
