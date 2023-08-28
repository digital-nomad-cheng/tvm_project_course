import onnx
import tvm
from tvm import relay
from tvm.relay import transform, build

simple_model = onnx.load("simple_model.onnx")
mod, params = tvm.relay.frontend.from_onnx(simple_model)

mod["main"] = relay.build_module.bind_params_by_name(mod["main"], params)
mod = relay.transform.AnnotateTarget(["tvmcon23"])(mod)
mod = relay.transform.MergeCompilerRegions()(mod)
mod = relay.transform.PartitionGraph()(mod)
print("begin to export library...")
target="llvm"
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)


lib.export_library('simple_model_byoc.so')
