import onnx
import tvm
from tvm import relay
from tvm.relay import transform, build
from tvm.contrib import graph_executor
import numpy as np
from tvm.relay.op.contrib.ncnn import partition_for_ncnn 

numpy_input_tensor = np.random.rand(1, 3, 64, 64)

def build_lib(mod, params, codegen="default", target="llvm", verbose=True):
    if codegen == "default":
        print("Use default codegen...")
    elif codegen == "ncnn":
        print("Use ncnn codegen...")
        if verbose:
            mod["main"] = relay.build_module.bind_params_by_name(mod["main"], params)
            print("============= model bind by params ==========")
            mod.show()
            mod = relay.transform.AnnotateTarget(["ncnn"])(mod)
            print("============= model annotated by ncnn codgen ==========")
            mod.show()
            mod = relay.transform.PartitionGraph()(mod)
            print("============= model with graph partitioned ============")
            mod.show()
        else:
            with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
                mod = partition_for_ncnn(mod, params)
            print("============= model paritioned using parition_for_ncnn ==========")
            mod.show()
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)
    return lib

def export_lib(lib, codegen="default", model_name="simple"):
    print("Export runtime library...")
    lib.export_library("onnx_{}_{}_lib.so".format(codegen, model_name))

def run(lib, numpy_input_tensor):
    print("Run with graph executor...")
    rt_mod = graph_executor.GraphModule(lib["default"](tvm.cpu()))
    rt_mod.set_input("input", tvm.nd.array(numpy_input_tensor.astype("float32")))
    rt_mod.run()
    output = rt_mod.get_output(0)
    return output  
            
# load simple model composed of conv bn relu linear to tvm
simple_model = onnx.load("/home/work/tvm_project_course/miscellaneous/simple_model_sim.onnx")
mod, params = tvm.relay.frontend.from_onnx(simple_model)
print("============== traced model from onnx==============")
mod.show()

default_lib = build_lib(mod, params)

ncnn_lib = build_lib(mod, params, codegen="ncnn", verbose=False)
export_lib(ncnn_lib, codegen="ncnn")

print("Load runtime from library...")
loaded_ncnn_lib = tvm.runtime.load_module("onnx_ncnn_simple_lib.so")

tvm_output = run(default_lib, numpy_input_tensor)
print("tvm output is...\n", tvm_output)
ncnn_output = run(loaded_ncnn_lib, numpy_input_tensor)
print("ncnn output is...\n", ncnn_output)

