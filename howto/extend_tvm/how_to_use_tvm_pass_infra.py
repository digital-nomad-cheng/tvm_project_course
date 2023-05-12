import numpy as np
import tvm
from tvm import relay

def example():
    shape = (1, 64, 54, 54)
    c_data = np.empty(shape).astype("float32")
    c = relay.const(c_data)
    weight = relay.var("weight", shape=(64, 64, 3, 3))
    x = relay.var("x", relay.TensorType((1, 64, 56, 56), "float32"))
    conv = relay.nn.conv2d(x, weight)
    y = relay.add(c, c)
    y = relay.multiply(y, relay.const(2, "float32"))
    y = relay.add(conv, y)
    z = relay.add(y, c)
    z1 = relay.add(y, c)
    z2 = relay.add(z, z1)
    return relay.Function([x, weight], z2)

f = example()

mod = tvm.IRModule.from_expr(f)

print(mod)

@relay.transform.function_pass(opt_level=2)
class CustomPipeline:
    """Simple test function to replace one argument to another."""

    def __init__(self, multiplier):
        self.multiplier = multiplier

    # This function can define a pass.
    def transform_function(self, func, mod, ctx):
        obj = self

        class ReplaceConstant(tvm.relay.ExprMutator):
            def visit_constant(self, c):
                return relay.multiply(obj.multiplier, c)

        return ReplaceConstant().visit(func)


func_pass = CustomPipeline(relay.const(3, "float32"))
assert func_pass.info.opt_level == 2
assert func_pass.info.name == "CustomPipeline"

mod1 = func_pass(mod)

print(mod1)
