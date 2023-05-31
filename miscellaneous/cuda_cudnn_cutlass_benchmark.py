# code copied from: https://github.com/umiswing/tvm-cutlass-eval
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import tvm
from tvm import relay
import tvm.relay.testing
from tvm.relay.op.contrib.cutlass import partition_for_cutlass
import torchvision
from tvm.relay.transform import ToMixedPrecision
from tvm.contrib.download import download_testdata
from tvm.relay.build_module import bind_params_by_name
from tvm.relay import transform
from tvm.contrib.cutlass import (
    has_cutlass,
    num_cutlass_partitions,
    finalize_modules,
    finalize_modules_vm,
)


def profile_and_build(
    mod,
    params,
    sm,
    split_k_slices=[1],
    tmp_dir="./tmp",
    use_fast_math=False,
    use_3xtf32=False,
    use_ansor=False,
    ansor_tuning=False,
    lib_path="compile.so",
    precompiled=False,
    use_cudnn=False,
):
    dev = tvm.device("cuda", 0)
    if precompiled:
        lib = tvm.runtime.load_module(lib_path)
        rt_mod = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
        return rt_mod, dev, 1
    elif use_cudnn:
        with tvm.transform.PassContext(opt_level=3):
            target = tvm.target.Target(
                {
                    "kind": "cuda",
                    "device": "nvidia/geforce-rtx-3070",
                    "libs": ["cudnn","cublas"],
                }
            )
            lib = relay.build(mod, target=target, params=params)
        lib.export_library(lib_path, workspace_dir=tmp_dir)
        rt_mod = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
        return rt_mod, dev, 1
    else:
        mod = partition_for_cutlass(mod)
        mod = convert_conv2d_layout(mod, {"nn.conv2d": ["NHWC", "default"]})
        num_cutlass_partition = num_cutlass_partitions(mod)
        host = tvm.target.Target("llvm")
        cuda = tvm.target.Target("nvidia/geforce-rtx-3070", host=host)
        cutlass = tvm.target.Target(
            {
                "kind": "cutlass",
                "sm": sm,
                "use_3xtf32": use_3xtf32,
                "split_k_slices": split_k_slices,
                "profile_all_alignments": False,
                "find_first_valid": True,
                "use_multiprocessing": True,
                "use_fast_math": False, # use_fast_math,
                "tmp_dir": tmp_dir,
            },
            host=host,
        )
        print("num of partiitons:", num_cutlass_partition)
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=[cuda, cutlass], params=params)
        lib = finalize_modules(lib, lib_path, tmp_dir)
        rt_mod = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
        return rt_mod, dev, num_cutlass_partition


def convert_conv2d_layout(mod, desired_layouts):
    with tvm.transform.PassContext(opt_level=3):
        seq = tvm.transform.Sequential([relay.transform.ConvertLayout(desired_layouts)])
        return seq(mod)

img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_path = download_testdata(img_url, "cat.png", module="data")
img = Image.open(img_path).resize((224, 224))

my_preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
img = my_preprocess(img)
img = np.expand_dims(img, 0)

batch_size = 1
img = np.tile(img, (batch_size, 1, 1, 1))

sm  = 80
# model = torchvision.models.resnet50(pretrained=True).eval()
model = torchvision.models.mobilenet_v2(pretrained=True).eval()
input_name = "input0"
input_data = torch.from_numpy(img)
scripted_model = torch.jit.trace(model, input_data).eval()

with torch.no_grad():
    torch_res = scripted_model(input_data).numpy()

shape_list = [(input_name, img.shape)]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

dev = tvm.device("cuda", 0)
# mod = ToMixedPrecision("float16")(mod)
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target="nvidia/geforce-rtx-3070", params=params)
rt_mod_cuda = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
mod = convert_conv2d_layout(mod, {"nn.conv2d": ["NHWC", "OHWI"]})

rt_mod_cutlass, dev, num_partition = profile_and_build(mod, params, sm, use_fast_math=True, lib_path="cutlass_compile.so")
rt_mod_cudnn, dev, num_partition = profile_and_build(mod, params, sm, use_fast_math=True, lib_path="cudnn_compile.so", use_cudnn=True)
rt_mod_cutlass_compile, dev, num_partition = profile_and_build(mod, params, sm, use_fast_math=True, lib_path="tmp/cutlass_compile.so", precompiled=True)
rt_mod_cudnn_compile, dev, num_partition = profile_and_build(mod, params, sm, use_fast_math=True, lib_path="cudnn_compile.so", use_cudnn=True, precompiled=True)
# print(rt_mod)
print(num_partition)
assert num_partition > 0

rt_mod_cutlass_compile.set_input(input_name, img)
rt_mod_cutlass_compile.run()
tvm_res = rt_mod_cutlass_compile.get_output(0).numpy()
print(tvm_res[0][:10])
print(torch_res[0][:10])


print("Evaluate inference time cost...")
print("cuda...")
print(rt_mod_cuda.benchmark(dev, number=1, repeat=100))
print("cutlass...")
print(rt_mod_cutlass.benchmark(dev, number=1, repeat=100))
print("cudnn...")
print(rt_mod_cudnn.benchmark(dev, number=1, repeat=100))
print("cutlass precompiled...")
print(rt_mod_cutlass_compile.benchmark(dev, number=1, repeat=100))
print("cudnn precompiled...")
print(rt_mod_cudnn_compile.benchmark(dev, number=1, repeat=100))

