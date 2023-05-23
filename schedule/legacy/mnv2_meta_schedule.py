# ref: https://gist.github.com/mkatanbaf/1a7e814e4cc757ea9babfcb013efe124
from types import MappingProxyType

import numpy as np

import torch
import torchvision
from torchvision import transforms

import tvm
from tvm import relay, auto_scheduler
from tvm import meta_schedule as ms
from tvm.contrib.download import download_testdata
from tvm.contrib import graph_executor

from PIL import Image

# load pretrained pytorch model
def load_pretrained_model(model_name="mobilenet_v2"):
    model = getattr(torchvision.models, model_name)(pretrained=True)
    model = model.eval()

    input_shape = [1, 3, 224, 224]
    input_tensor = torch.randn(input_shape)
    scripted_model = torch.jit.trace(model, input_tensor).eval()
    return scripted_model

def import_pytorch_to_relay(scripted_model, input_name:str="input_tensor", input_shape=[1, 3, 224, 224]):

    shape_list = [(input_name, input_shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
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

def meta_schedule(mod, params, target:str="cuda", work_dir='./work_dir', log_file="tune_log.json"):
    target = tvm.target.Target(target)
    
    with ms.Profiler() as profiler:
        rt_db: tvm.runtime.Module = ms.relay_integration.tune_relay(
            mod=mod,
            params=params,
            target=target,
            strategy="evolutionary",
            num_trials_per_iter=32,
            max_trials_per_task=32,
            max_trials_global=1000,
            work_dir=work_dir,
        )
        
        lib: tvm.runtime.Module = ms.relay_integration.compile_relay(
            database=rt_db,
            mod=mod,
            target=target,
            params=params,
            pass_config=MappingProxyType({
                "relay.backend.use_meta_schedule": True,
                "relay.backend.tir_converter": "default",
                "tir.disable_vectorize": True,
                }
            ),
        )

    print(profiler.table())
    
    return lib

scripted_model = load_pretrained_model()

# print(scripted_model)

mod, params = import_pytorch_to_relay(scripted_model)

lib_navie = build_relay_graph(mod, params, "nvidia/geforce-rtx-3070")


lib_sch = meta_schedule(mod, params, target="nvidia/geforce-rtx-3070")

img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
input_tensor = load_test_image(img_url)

class_id_to_key, key_to_classname = load_idx2key_dict()

tvm_output_navie = predict(input_tensor, lib_navie)
tvm_output_meta = predict(input_tensor, lib_sch)
if np.allclose(tvm_output_navie.numpy(), tvm_output_meta.numpy(), rtol=1e-4, atol=2e-4):
    print("PASS")
else:
    print("FAIL")

top1_tvm = np.argmax(tvm_output_meta.numpy()[0])
tvm_class_key = class_id_to_key[top1_tvm]
print("Relay top-1 id: {}, class name: {}".format(top1_tvm, key_to_classname[tvm_class_key]))

