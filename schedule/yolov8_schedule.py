import os, sys
import argparse
import time
from types import MappingProxyType

import numpy as np
from PIL import Image
import onnx 

import torch
import torchvision
from torchvision import transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

import tvm
from tvm import (
    relay, 
    auto_scheduler,
    meta_schedule as ms
)
from tvm.relay.op.contrib.tensorrt import partition_for_tensorrt
from tvm.relay.op.contrib.cutlass import partition_for_cutlass
from tvm.contrib.cutlass import (
    has_cutlass,
    num_cutlass_partitions,
    finalize_modules
)
from tvm.contrib.download import download_testdata
from tvm.contrib import graph_executor


# load pretrained onnx model
def load_pretrained_model(model_name="yolov8l", model_dir="onnx_model"):
    onnx_model = onnx.load_model(os.path.join(model_dir, model_name+".onnx"))

    return onnx_model 

#import onnx model into relay graph
def import_onnx_to_relay(scripted_model, input_name:str="images", input_shape=[1, 3, 640, 640]):

    shape_dict = {input_name: input_shape}
    mod, params = relay.frontend.from_onnx(scripted_model, shape_dict)
    return mod, params

def build_relay_graph(mod, params, target:str="cuda", codegen=None):
    host = tvm.target.Target("llvm")
    cutlass = tvm.target.Target(
        {
            "kind": "cutlass",
            "sm": 80,
            "use_3xtf32": True,
            "split_k_slices": [1],
            "profile_all_alignments": False,
            "find_first_valid": True,
            "use_multiprocessing": True,
            "use_fast_math": True,
            "tmp_dir": "./tmp",
        },
        host=host,
    )
    target = tvm.target.Target(target, host=host)
    if codegen == "tensorrt":
        print("Use tensorrt codegen...")
        mod = partition_for_tensorrt(mod, params)
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params=params)
    elif codegen == "cutlass":
        print("Use cutlass codegen...")
        def convert_conv2d_layout(mod, desired_layouts):
            with tvm.transform.PassContext(opt_level=3):
                seq = tvm.transform.Sequential([relay.transform.ConvertLayout(desired_layouts)])
                return seq(mod)
        print(mod)
        mod = convert_conv2d_layout(mod, {"nn.conv2d": ["NHWC", "OHWI"]})
        mod = partition_for_cutlass(mod, params)
        mod = convert_conv2d_layout(mod, {"nn.conv2d": ["NHWC", "default"]})
        num_partition = num_cutlass_partitions(mod)
        assert num_partition != 0, "Partition for cutlass failed!"
        print("num of partition using cutlass:", num_partition)
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=[target, cutlass], params=params)
        lib = finalize_modules(lib)
    else:
        print("Use default codegen...")
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params=params)

    return lib

def predict(input_tensor, lib, executor="tvm", input_name="input_tensor", benchmark=True):
    dtype = "float32"
    if executor == "tvm":
        dev = tvm.cuda(0)
        m = graph_executor.GraphModule(lib["default"](dev))
        m.set_input(input_name, tvm.nd.array(input_tensor.astype(dtype)))
        m.run()
        output = m.get_output(0)
        if benchmark:
            print(m.benchmark(dev, repeat=2, min_repeat_ms=500))
            print(m.benchmark(dev, repeat=5, min_repeat_ms=500))

    return output
    
def schedule(mod, params, strategy="auto", target:str="cuda", work_dir="./work_dir", 
             log_file="tune_log.json"):
    target = tvm.target.Target(target)
    if strategy == "auto":
        log_file = os.path.join(work_dir, "auto", log_file)
        tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
        print("total tasks:", len(tasks))
        for i, task in enumerate(tasks):
            print("================= Task %d (workload key: %s) ================" %(i, task.workload_key))
            print(task.compute_dag)
        measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, 
                timeout=10)

        tuner = auto_scheduler.TaskScheduler(tasks, task_weights, load_log_file=log_file) 
        tune_option = auto_scheduler.TuningOptions(
                num_measure_trials=10000,
                # runner=auto_scheduler.LocalRunner(repeat=10, enable_cpu_cache_flush=True),
                runner=measure_ctx.runner,
                measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
                )
        tuner.tune(tune_option)
        # apply the best schedule record to module and build library
        with auto_scheduler.ApplyHistoryBest(log_file):
            with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
                lib = relay.build(mod, target=target, params=params)
        
        return lib
    else:
        raise NotImplementedError("Strategy {} is not implemented!".format(strategy))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser("MobileNetv2 scheduler")
    parser.add_argument("-i", "--input_model", default="yolov8n", type=str, help="Path to pretrained pytorch model")
    parser.add_argument("-t", "--target", default="nvidia/geforce-rtx-3070", type=str, help="Target string defined in tvm tag.cc")
    parser.add_argument("-s", "--strategy", default="auto", type=str, help=r"Strategy used for sheduling, can be 'meta' or 'auto'")
    parser.add_argument("-c", "--codegen", default="cutlass", type=str, help=r"Codegen used for dispatch operators, can be 'tensorrt', 'cutlass', or 'None'")
    parser.add_argument("--use_scheduler", default=False, type=bool, help="Whether to turn on scheduler, off by default.")

    parser.print_help()
    args = parser.parse_args()
    
    scripted_model = load_pretrained_model(args.input_model)

    mod, params = import_onnx_to_relay(scripted_model)
    
    input_tensor = np.random.rand(1, 3, 640, 640).astype(np.float32)
    
    t0 = time.time()
    lib_navie = build_relay_graph(mod, params, args.target)
    t1 = time.time()
    print("Total time for default building {} on {} is: {}".format(args.input_model, args.target, t1-t0))
    lib_navie.export_library("libs/{}_{}_default_build.so".format(args.input_model, args.target.replace("/", "_")))
    tvm_output_navie = predict(input_tensor, lib_navie, input_name="images", benchmark=True)
      
    t0 = time.time()
    lib_tensorrt = build_relay_graph(mod, params, args.target, codegen="tensorrt")
    t1 = time.time()
    print("Total time for default building {} with tensorrt support on {} is: {}".format(args.input_model, args.target, t1-t0))
    lib_tensorrt.export_library("libs/{}_{}_tensorrt_build.so".format(args.input_model, args.target.replace("/", "_")))
    tvm_output_tensorrt = predict(input_tensor, lib_tensorrt, input_name="images", benchmark=True)
    
    t0 = time.time()
    lib_cutlass = build_relay_graph(mod, params, args.target, codegen="cutlass")
    t1 = time.time()
    print("Total time for default building {} with cutlass support on {} is: {}".format(args.input_model, args.target, t1-t0))
    tvm_output_cutlass = predict(input_tensor, lib_cutlass)
    
    if args.use_scheduler:
        log_file = "{}_{}_{}_tune_log.json".format(args.input_model, args.target.replace("/", "_"), args.strategy)
        t0 = time.time()
        lib_sch = schedule(mod, params, strategy=args.strategy, target=args.target, log_file=log_file)
        t1 = time.time()
        print("Total time for {} scheduling {} on {} is: {}".format(args.strategy, args.input_model, args.target, t1-t0))
        lib_sch.export_library("libs/{}_{}_{}_schedule.so".format(args.input_model, args.target.replace("/", "_"), args.strategy))
        tvm_output_sch = predict(input_tensor, lib_sch, input_name="images", benchmark=True)
    else:
        tvm_output_sch = tvm_output_cutlass
    
    np.testing.assert_allclose(tvm_output_navie.numpy(), tvm_output_sch.numpy(), rtol=1e-2, atol=1e-5)
