
import torch
import torchvision

import tvm
from tvm import relay

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

scripted_model = load_pretrained_model()

# print(scripted_model)

mod, params = import_pytorch_to_relay(scripted_model)

build_relay_graph(mod, params, "nvidia/geforce-rtx-3070")
