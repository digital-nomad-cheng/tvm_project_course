import os
import time 
import json

import numpy as np
import torch
from torch import nn
import torchvision
from torchvision.models import MobileNet_V2_Weights, AlexNet_Weights

import tvm
from tvm import (
    relay
)
from tvm.contrib import graph_executor
from tvm.relay.op.contrib.ncnn import partition_for_ncnn
from tvm.relay.op.contrib.arm_compute_lib import partition_for_arm_compute_lib
from tvm.relay.build_module import bind_params_by_name

def extract_byoc_modules(module, codegen="ncnn"):
    """Get the BYOC module(s) from llvm module."""
    print("Extract BYOC modules from runtime...")
    return list(filter(lambda mod: mod.type_key == codegen,
                       module.get_lib().imported_modules))

# load pretrained pytorch model
def load_pretrained_model(model_name="alexnet"):
    model = getattr(torchvision.models, model_name)(weights=AlexNet_Weights.DEFAULT)
    model.eval()
    input_shape = [1, 3, 227, 227]
    input_tensor = torch.randn(input_shape)
    scripted_model = torch.jit.trace(model, input_tensor).eval()
    return scripted_model

def import_pytorch_to_relay(scripted_model, input_name:str="input_tensor", input_shape=[1, 3, 227, 227], preprocess=True):

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

    with tvm.transform.PassContext(opt_level=0, disabled_pass=["AlterOpLayout"]):
        lib = relay.build(mod, target=target, params=params)

    return lib

def run(lib, numpy_input_tensor, codegen="default"):
    print("Run with {} graph executor...".format(codegen))
    rt_mod = graph_executor.GraphModule(lib["default"](tvm.cpu()))
    rt_mod.set_input("input", tvm.nd.array(numpy_input_tensor.astype("float32")))
    rt_mod.run()
    output = rt_mod.get_output(0)
    return output

def dump_module_to_json(modules):
    print("Dump json to files...")
    os.mkdir("./tmp")
    with open('./tmp/ncnn_modules.json', 'w') as outfile, \
            open('./tmp/ncnn_modules_readable.json', 'w') as readable_outfile:
        for mod in modules:
            source = mod.get_source("json")
            node_info = json.loads(source)["nodes"]
            readable_outfile.write(json.dumps(node_info, sort_keys=True, indent=2))
            json.dump(node_info, outfile)

if __name__ == "__main__":
    numpy_input_tensor = np.random.rand(1, 3, 227, 227)
    scripted_model = load_pretrained_model()
    mod, params  = import_pytorch_to_relay(scripted_model, input_name="input")
    tvm_lib = build_mnv2_lib(mod, params)
    tvm_lib.export_library("alexnet_tvm_lib.so")
    tvm_lib = tvm.runtime.load_module("/home/tvm_project_course/byoc/alexnet_tvm_lib.so")
    t0 = time.time()
    ncnn_lib = build_mnv2_lib(mod, params, codegen="ncnn")
    t1 = time.time()
    print("Time for build runtime library using ncnn codegen for mnv2 is:", t1-t0)
    
    ncnn_modules = extract_byoc_modules(ncnn_lib)
    dump_module_to_json(ncnn_modules) 

    tvm_output = run(tvm_lib, numpy_input_tensor)
    ncnn_output = run(ncnn_lib, numpy_input_tensor, codegen="ncnn")
    print("tvm output is...\n", tvm_output.numpy()[0, :5])
    print("ncnn output is...\n", ncnn_output.numpy()[0, :5])

    benchmark = True
    if benchmark:
        t0 = time.time()
        for i in range(100):
            tvm_output = run(tvm_lib, numpy_input_tensor)
        t1 = time.time()
        tvm_default_time  = t1 - t0

        t0 = time.time()
        for i in range(100):
            ncnn_output = run(ncnn_lib, numpy_input_tensor, codegen="ncnn")
        t1 = time.time()
        ncnn_time = t1 - t0

        print("total time for tvm default runtime: ", tvm_default_time)
        print("total time for ncnn default runtime: ", ncnn_time)


