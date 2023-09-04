import onnx
import tvm
from tvm import relay
from tvm.relay import transform, build
from tvm.contrib import graph_executor
import numpy as np

# load simple model composed of conv bn relu linear to tvm
simple_model = onnx.load("/home/work/tvm_project_course/miscellaneous/simple_model.onnx")
mod, params = tvm.relay.frontend.from_onnx(simple_model)

mod["main"] = relay.build_module.bind_params_by_name(mod["main"], params)
print("============= model bind by params ==========")
mod.show()
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

input_tensor = np.random.rand(1, 3, 64, 64)
m = graph_executor.GraphModule(lib["default"](tvm.cpu()))
m.set_input("input", tvm.nd.array(input_tensor.astype("float32")))
m.run()
output = m.get_output(0)
print(output)
print(output.numpy().shape)
lib.export_library('simple_model_ncnn_codegen.so')
