import torch
from torch import nn 
import tvm
from tvm import relay 
from tvm.relay import transform, build
from tvm.contrib import graph_executor
import numpy as np 

class ConvBNReLUFC(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv = nn.Conv2d(3, 16, 3, 1, 1, bias=True)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(16*64*64, 10)

        self.__init_weights()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def __init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

numpy_input_tenosr = np.random.rand(1, 3, 64, 64)
torch_input_tensor = torch.rand([1, 3, 64, 64]).float()

simple_model = ConvBNReLUFC().eval().float()
scripted_model = torch.jit.trace(simple_model, torch_input_tensor)
input_shapes = [("input", [1, 3, 64, 64])]
mod, params = tvm.relay.frontend.from_pytorch(scripted_model, input_shapes)
print("============== traced model from pytorch===============")
mod.show()

print("============== partition_for_ncnn ==========")
mod = tvm.relay.op.contrib.ncnn.partition_for_ncnn(mod, params)
mod.show()
target = "llvm"
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)
lib.export_library("simple_model_ncnn_codegen.so")
print("Build graph executor...")
m = graph_executor.GraphModule(lib["default"](tvm.cpu()))
m.set_input("input", tvm.nd.array(numpy_input_tenosr.astype("float32")))
m.run()
output = m.get_output(0)
print(output)

print("=================parition using arm_compute_lib API=============")
from tvm.relay.op.contrib.arm_compute_lib import partition_for_arm_compute_lib
mod, params = tvm.relay.frontend.from_pytorch(scripted_model, input_shapes)
mod = partition_for_arm_compute_lib(mod, params)
mod.show()
target = "llvm"

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)
lib.export_library("simple_model_ncnn_codegen.so")

