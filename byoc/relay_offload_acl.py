import tvm
from tvm import relay
import onnx

model_name = "simple"
if model_name == "max_pool2d":
    data_type = "float32"
    data_shape = (1, 14, 14, 512)
    strides = (2, 2)
    padding = (0, 0, 0, 0)
    pool_size = (2, 2)
    layout = "NHWC"
    output_shape = (1, 7, 7, 512)

    data = relay.var('data', shape=data_shape, dtype=data_type)
    out = relay.nn.max_pool2d(data, pool_size=pool_size, strides=strides, layout=layout, padding=padding)
    module = tvm.IRModule.from_expr(out)

    from tvm.relay.op.contrib.arm_compute_lib import partition_for_arm_compute_lib
    module = partition_for_arm_compute_lib(module)

    target = "llvm -mtriple=aarch64-linux-gnu -mattr=+neon"
    with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
        lib = relay.build(module, target=target)

    lib_path = './lib_acl.so'
    cross_compile = 'aarch64-linux-gnu-g++'
    lib.export_library(lib_path, cc=cross_compile)
elif model_name == "simple":
    onnx_model = onnx.load("/home/work/tvm_project_course/miscellaneous/simple_model.onnx")
    mod, params = relay.frontend.from_onnx(onnx_model)
    mod.show()
    mod = tvm.relay.transform.ConvertLayout({"nn.conv2d": ["NHWC", "default"]})(mod)
    mod.show()
    from tvm.relay.op.contrib.arm_compute_lib import partition_for_arm_compute_lib
    mod = partition_for_arm_compute_lib(mod, params)
    mod.show()
    target = "llvm -mtriple=aarch64-linux-gnu -mattr=+neon"
    with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
        lib = relay.build(mod, target=target)

    lib_path = './lib_simple_acl.so'
    cross_compile = 'aarch64-linux-gnu-g++'
    lib.export_library(lib_path, cc=cross_compile)
