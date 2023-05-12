import torch
import numpy as np
import torchvision

from tvm import relay

# load pretrained pytorch model and return traced graph
def load_pretrained_model(model_name="mobilenet_v2"):
    model = getattr(torchvision.models, model_name)(pretrained=True)
    model = model.eval()

    input_shape = [1, 3, 224, 224]
    input_tensor = torch.randn(input_shape)
    scripted_model = torch.jit.trace(model, input_tensor).eval()
    return scripted_model

# import pytorch graph into relay graph
def import_pytorch_to_relay(scripted_model, input_name:str="input_tensor", input_shape=[1, 3, 224, 224]):

    shape_list = [(input_name, input_shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    return mod, params

if __name__=='__main__':
    scripted_model = load_pretrained_model()
    mod, params = import_pytorch_to_relay(scripted_model)
    print("Parsed mod: ", mod)