import tvm
from tvm.te import create_prim_func

n = tvm.te.var()
A = tvm.te.placeholder((n, n), name="A")
B = tvm.te.placeholder((n, n), name="B")
# define computation with ComputeOp
C = tvm.te.compute((n, n), lambda i, j: A[i, j] + B[i, j], name="C")
compute_func = create_prim_func([A, B, C])
print("computation definition:", compute_func.script())

s = tvm.te.create_schedule(C.op)
ret = tvm.lower(s, [A, B, C])
print(ret)
print(type(ret))

mod = tvm.build(s, [A, B, C], target="llvm", name="main")
print(mod)
print(type(mod))
