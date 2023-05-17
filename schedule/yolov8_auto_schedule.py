import time
import numpy as np

import torch
import torchvision
from torchvision import transforms
import onnx 

import tvm
from tvm import relay, auto_scheduler
from tvm.contrib.download import download_testdata
from tvm.contrib import graph_executor

from PIL import Image

# load pretrained pytorch model
def load_pretrained_model(model_name="yolov8l"):
    onnx_model = onnx.load_model(model_name+".onnx")

    return onnx_model 

def import_pytorch_to_relay(scripted_model, input_name:str="images", input_shape=[1, 3, 640, 640]):

    shape_dict = {input_name: input_shape}
    mod, params = relay.frontend.from_onnx(scripted_model, shape_dict)
    return mod, params

def build_relay_graph(mod, params, target:str="cuda"):
    target = tvm.target.Target(target)
    # dev = tvm.cuda(0)
    # dev = tvm.device(str(target), 0)
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)

    return lib

def load_test_image(image_url):
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
    input_tensor = np.expand_dims(img, 0)

    return input_tensor

def load_idx2key_dict():
    synset_url = "".join(
    [
        "https://raw.githubusercontent.com/Cadene/",
        "pretrained-models.pytorch/master/data/",
        "imagenet_synsets.txt",
    ]
    )
    synset_name = "imagenet_synsets.txt"
    synset_path = download_testdata(synset_url, synset_name, module="data")
    with open(synset_path) as f:
        synsets = f.readlines()

    synsets = [x.strip() for x in synsets]
    splits = [line.split(" ") for line in synsets]
    key_to_classname = {spl[0]: " ".join(spl[1:]) for spl in splits}

    class_url = "".join(
        [
            "https://raw.githubusercontent.com/Cadene/",
            "pretrained-models.pytorch/master/data/",
            "imagenet_classes.txt",
        ]
    )
    class_name = "imagenet_classes.txt"
    class_path = download_testdata(class_url, class_name, module="data")
    with open(class_path) as f:
        class_id_to_key = f.readlines()

    class_id_to_key = [x.strip() for x in class_id_to_key]

    return class_id_to_key, key_to_classname

def predict(input_tensor, lib, executor="tvm", input_name="input_tensor", benchmark=True):
    dtype = "float32"
    if executor == "tvm":
        dev = tvm.cuda(0)
        m = graph_executor.GraphModule(lib["default"](dev))
        m.set_input(input_name, tvm.nd.array(input_tensor.astype(dtype)))
        m.run()
        output = m.get_output(0)
        if benchmark:
            print(m.benchmark(dev, repeat=3, min_repeat_ms=500))

    return output

def schedule(mod, params, target:str="cuda", log_file="tune_log.json"):
    target = tvm.target.Target(target)
    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
    print("total tasks:", len(tasks))
    for i, task in enumerate(tasks):
        print("================= Task %d (workload key: %s) ================" %(i, task.workload_key))
        print(task.compute_dag)
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, 
            timeout=10)

    tuner = auto_scheduler.TaskScheduler(tasks, task_weights, load_log_file=log_file) 
    tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=200,
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

scripted_model = load_pretrained_model()

# print(scripted_model)

mod, params = import_pytorch_to_relay(scripted_model)

t0 = time.time()
lib_navie = build_relay_graph(mod, params, "nvidia/geforce-rtx-3070")
t1 = time.time()
print("Total time for default building yolov8l:", t1 - t0)




# lib_sch = schedule(mod, params, target="nvidia/geforce-rtx-3070")

# img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
# input_tensor = load_test_image(img_url)

# class_id_to_key, key_to_classname = load_idx2key_dict()

# tvm_output = predict(input_tensor, lib_navie)
# tvm_output = predict(input_tensor, lib_sch)
# top1_tvm = np.argmax(tvm_output.numpy()[0])
# tvm_class_key = class_id_to_key[top1_tvm]
# print("Relay top-1 id: {}, class name: {}".format(top1_tvm, key_to_classname[tvm_class_key]))

