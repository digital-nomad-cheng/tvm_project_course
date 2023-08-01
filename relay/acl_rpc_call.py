import numpy as np
import tvm
from tvm import rpc
from tvm.contrib import graph_executor
host = "192.168.0.154"
port = 9090

remote = rpc.connect(host, port)

remote.upload("./lib_acl.so")

dev = remote.cpu(0)
loaded_lib = remote.load_module("lib_acl.so")
gen_module = tvm.contrib.graph_executor.GraphModule(loaded_lib['default'](dev))

data_type = "float32"
data_shape = (1, 14, 14, 512)
d_data = np.random.uniform(0, 1, data_shape).astype(data_type)
map_inputs = {'data': d_data}

gen_module.set_input(**map_inputs)
gen_module.run()
output = gen_module.get_output(0)
print(output.shape)


