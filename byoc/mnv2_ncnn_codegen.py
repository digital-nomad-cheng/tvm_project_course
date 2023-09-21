import numpy as np
import torch
import torchvision
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2 

import tvm
from tvm import (
    relay
)
from tvm.contrib import graph_executor
from tvm.relay.op.contrib.ncnn import partition_for_ncnn
from tvm.relay.op.contrib.arm_compute_lib import partition_for_arm_compute_lib
from tvm.relay.build_module import bind_params_by_name

# load pretrained pytorch model
def load_pretrained_model(model_name="mobilenet_v2"):
    model = getattr(torchvision.models, model_name)(weights=MobileNet_V2_Weights.DEFAULT)
    model = model.eval()

    input_shape = [1, 3, 224, 224]
    input_tensor = torch.randn(input_shape)
    scripted_model = torch.jit.trace(model, input_tensor).eval()
    return scripted_model

def import_pytorch_to_relay(scripted_model, input_name:str="input_tensor", input_shape=[1, 3, 224, 224], preprocess=True):

    shape_list = [(input_name, input_shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

    if preprocess:
        mod["main"] = bind_params_by_name(mod["main"], params)
        mod = relay.transform.InferType()(mod)
        mod = relay.transform.FoldConstant()(mod)

    return mod, params

def build_mnv2_lib(mod, params, codegen="default", target="llvm"):
    print("mod before transform...")
    mod.show()

    if codegen == "default":
        print("Use default codegen...")
    else:
        print("Use ncnn codegen...")
        mod = partition_for_ncnn(mod, params)
        print("mod after transform...")
    
    mod.show()

    with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
        lib = relay.build(mod, target=target, params=params)

    return lib

def run(lib, numpy_input_tensor, codegen="default"):
    print("Run with {} graph executor...".format(codegen))
    rt_mod = graph_executor.GraphModule(lib["default"](tvm.cpu()))
    rt_mod.set_input("input", tvm.nd.array(numpy_input_tensor.astype("float32")))
    rt_mod.run()
    output = rt_mod.get_output(0)
    return output

if __name__ == "__main__":
    numpy_input_tensor = np.random.rand(1, 3, 224, 224)
    scripted_model = load_pretrained_model()
    mod, params  = import_pytorch_to_relay(scripted_model, input_name="input")
    tvm_lib = build_mnv2_lib(mod, params)
    ncnn_lib = build_mnv2_lib(mod, params, codegen="ncnn")
    tvm_output = run(tvm_lib, numpy_input_tensor)
    ncnn_output = run(ncnn_lib, numpy_input_tensor, codegen="ncnn")
    print("tvm output is...\n", tvm_output.numpy()[0, :10])
    print("ncnn output is...\n", ncnn_output.numpy()[0, :10])
