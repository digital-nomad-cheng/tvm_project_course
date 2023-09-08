When exporting the simple model from pytorch to onnx, the nn.bias_add operation will be replaced with add operation.
This will result in the relay failed to match dense + bias operation, while if we directly parse from pytorch traced 
graph this will not happen. 
