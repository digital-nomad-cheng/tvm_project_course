import tvm
from tvm import relay
import torch
from torch import nn

class ConvBNReLU(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

torch.set_grad_enabled(False)
input_shape = [1, 3, 8, 8]
input_data = torch.rand(input_shape).float()
pt_model = ConvBNReLU().eval().float()
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

