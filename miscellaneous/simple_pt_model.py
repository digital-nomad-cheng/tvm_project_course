import tvm
from tvm import relay
import torch
from torch import nn
import onnx

class ConvBNReLU(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        self.__init_weights()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x
    
    def __init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

torch.set_grad_enabled(False)
input_shape = [1, 3, 8, 8]
input_data = torch.rand(input_shape).float()
pt_model = ConvBNReLU().eval().float()
torch.onnx.export(pt_model, input_data,"simple_model.onnx", input_names=["input"], output_names=["output"])
print("Finish export onnx model.")
onnx_model = onnx.load("simple_model.onnx")
onnx.checker.check_model(onnx_model)

input_shapes = [("data", input_shape)]
scripted_model = torch.jit.trace(pt_model, input_data)
mod, params = relay.frontend.from_pytorch(scripted_model, input_shapes)
print("mod before optimizing...")
print(mod["main"])

seq = tvm.transform.Sequential(
        [
            relay.transform.SimplifyInference(),
            relay.transform.InferType(),
            relay.transform.FuseOps(),
            # relay.transform.FoldConstant(),
        ]
    )
with tvm.transform.PassContext(opt_level=3):
    mod_opt = seq(mod)
# mod_opt = mod
mod_opt = relay.quantize.prerequisite_optimize(mod_opt, params)
mod_opt = relay.quantize.partition()(mod_opt)
print("mod after optimizing...")
print(mod_opt["main"])

