import tvm
from tvm import relay

import torch
import torchvision

# load pretrained pytorch model
def load_pretrained_model(model_name="mobilenet_v2"):
    model = getattr(torchvision.models, model_name)(pretrained=True)
    model = model.eval()
    
    input_shape = [1, 3, 224, 224]
    input_tensor = torch.randn(input_shape)
    scripted_model = torch.jit.trace(model, input_tensor).eval()
    return scripted_model


model_list = torchvision.models.list_models(module=torchvision.models)
print(model_list)
load_pretrained_model()

