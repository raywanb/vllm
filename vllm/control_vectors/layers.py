from vllm.control_vectors.control import ControlVector
from vllm.control_vectors.request import ControlVectorAdapter




class MLPWithControlVector(BaseLayerWithControlVector):
    def __init__(self, baselayer) -> None:
        super().__init__()
        self.base_layer = base_layer

    
    def forward(self):
        output, residual = self.base_layer.forward()

        