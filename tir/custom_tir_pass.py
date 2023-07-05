import tvm

from tvm import relay

x = relay.var("x", shape=[1, 10])
w = relay.var("w", shape=[20, 10])
y = relay.nn.dense(x, w)
fn = relay.Function([x, w], y)
fmod = tvm.IRModule.from_expr(fn)

@tvm.tir.transform.prim_func_pass(opt_level=0)
def print_tir(f, mod, ctx):
    print("print inside tir pass...")
    print(f)
    return f

with tvm.transform.PassContext(
    opt_level=3, config={"tir.add_lower_pass": [(3, print_tir)]}
    ):
    lib = relay.build(fmod, target="llvm")

print(fmod)
