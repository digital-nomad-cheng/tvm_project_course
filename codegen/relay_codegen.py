import torch
import torchvision
from tvm import relay
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

# load pretrained pytorch model
def load_pretrained_model(model_name="mobilenet_v2"):
    model = getattr(torchvision.models, model_name)(weights=MobileNet_V2_Weights.DEFAULT)
    model = model.eval()

    input_shape = [1, 3, 224, 224]
    input_tensor = torch.randn(input_shape)
    scripted_model = torch.jit.trace(model, input_tensor).eval()
    return scripted_model

def import_pytorch_to_relay(scripted_model, input_name:str="input_tensor", input_shape=[1, 3, 224, 224]):

    shape_list = [(input_name, input_shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    return mod, params


# mnv2 relay representation
scripted_model = load_pretrained_model()
mnv2_mod, mnv2_params = import_pytorch_to_relay(scripted_model) 

with relay.build_config(opt_level=0):
    _, mnv2_lib, _ = relay.build_module.build(mnv2_mod, "llvm", params=mnv2_params)

# print relay ir/intermediate representation
print("relay ir:\n")
print(mnv2_mod.astext(show_meta_data=False))

# print source code
print("source code:\n")
print(mnv2_lib.get_source())
