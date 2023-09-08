import onnx
import tvm
from tvm import relay
from tvm.relay import transform, build
from tvm.contrib import graph_executor
import numpy as np

numpy_input_tenosr = np.random.rand(1, 3, 64, 64)

# load simple model composed of conv bn relu linear to tvm
simple_model = onnx.load("/home/work/tvm_project_course/miscellaneous/simple_model_sim.onnx")
mod, params = tvm.relay.frontend.from_onnx(simple_model)
print("============== traced model from onnx==============")
mod.show()

mod["main"] = relay.build_module.bind_params_by_name(mod["main"], params)
print("============= model bind by params ==========")
mod.show()

target="llvm"
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)
lib.export_library('simple_model_default_codegen.so')

print("================Run inference with tvm default graph executor model...===========")

print("Build graph executor...")
m = graph_executor.GraphModule(lib["default"](tvm.cpu()))
m.set_input("input", tvm.nd.array(numpy_input_tenosr.astype("float32")))
m.run()
output = m.get_output(0)
print(output)

mod = relay.transform.AnnotateTarget(["ncnn"])(mod)
print("============= model annotated by ncnn codgen ==========")
mod.show()
mod = relay.transform.PartitionGraph()(mod)
print("============= model with graph partitioned ============")
mod.show()


print("begin to export library...")
target="llvm"
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)
lib.export_library('simple_model_ncnn_codegen.so')

print("Load library from file...")
lib = tvm.runtime.load_module("simple_model_ncnn_codegen.so")

print("============Run inference with ncnn codegen model...================")
print("Build graph executor...")
m = graph_executor.GraphModule(lib["default"](tvm.cpu()))
m.set_input("input", tvm.nd.array(numpy_input_tenosr.astype("float32")))
m.run()
output = m.get_output(0)
print(output)

#from tvm.relay.op.contrib.ncnn import partition_for_ncnn
from tvm.relay.op.contrib.arm_compute_lib import partition_for_arm_compute_lib
mod, params = tvm.relay.frontend.from_onnx(simple_model)
#mod = partition_for_ncnn(mod, params)
mod = partition_for_arm_compute_lib(mod, params)
print("Use parition for ncnn api...")
mod.show()

print("begin to export library...")
target="llvm"
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)
lib.export_library('simple_model_ncnn_codegen.so')
