import tvm
from tvm import relay

ttype1 = relay.TensorType((10, 10, 10), "float32")
ttype2 = relay.TensorType((10, ), "float32")

x = relay.var("x", ttype1)
beta = relay.var("beta", ttype2)
gamma = relay.var("gamma", ttype2)
moving_var = relay.var("moving_var", ttype2)
moving_mean = relay.var("moving_mean", ttype2)

y = x
eps = 1e-2
for i in range(3):
    y, _, _ = relay.nn.batch_norm(
        y + relay.const(1, "float32"),
        gamma, 
        beta,
        moving_mean,
        moving_var,
        epsilon=eps,
        axis=1,
    )

    y = relay.nn.dropout(y)

mod = tvm.ir.IRModule.from_expr(y)

mod = tvm.relay.transform.InferType()(mod)
mod.show()
mod = tvm.relay.transform.SimplifyInference()(mod)

mod.show()
