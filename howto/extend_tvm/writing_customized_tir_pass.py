import tvm
from tvm import te
import numpy as np

# write a simple vector and build with default schedule
n = tvm.tir.const(128, "int32")
a = te.placeholder((n, ), name="a")
b = te.placeholder((n, ), name="b")
c = te.compute((n, ), lambda i: a[i] + b[i], name="c")

sch = te.create_schedule(c.op)

ir = tvm.lower(sch, [a, b, c])
print(ir)

# wring a customized pass

# use tvm.tir.stmt_functor.post_order_visit(stmt, func) to gather information from the tir
loops = []

def find_width8(op):
    """Find all the 'tir.For' nodes whose extent can be divided by 8."""
    if isinstance(op, tvm.tir.For):
        if isinstance(op.extent, tvm.tir.IntImm):
            if op.extent.value % 8 == 0:
                loops.append(op)


def vectorize8(op):
    """Split can vectorize the loops found n find_width8"""
    if op in loops:
        extent = op.extent.value
        name = op.loop_var.name
        lo, li = te.var(name + ".outer"), te.var(name + ".inner")
        body = tvm.tir.stmt_functor.substitute(op.body, {op.loop_var: lo*8+li})
        body = tvm.tir.For(li, 0, 8, tvm.tir.ForKind.VECTORIZED, body)
        body = tvm.tir.For(lo, 0, extent // 8, tvm.tir.ForKind.SERIAL, body)
        
        return body
    
    return None


@tvm.tir.transform.prim_func_pass(opt_level=0)
def vectorize(f, mod, ctx):
    global loops
    tvm.tir.stmt_functor.post_order_visit(f.body, find_width8)

    if not loops:
        return f

    return f.with_body(tvm.tir.stmt_functor.ir_transform(f.body, None, vectorize8, ["tir.For"]))


with tvm.transform.PassContext(config={"tir.add_lower_pass": [(1, vectorize)]}):
    print(tvm.lower(sch, [a, b, c]))


