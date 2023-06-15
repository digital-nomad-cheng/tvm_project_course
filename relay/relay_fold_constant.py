import tvm
from tvm import relay
import numpy as np
from visualize import RelayVisualizer
from tvm.relay.testing import run_infer_type


def test_fold_dropout():
    def before():
        # A constant graph to fire fold constant
        data = relay.const(np.arange(10).astype(np.float32))
        data = data + relay.const(20.0)
        dropout = relay.nn.dropout(data)
        add = dropout + relay.const(1.0)
        add = add + relay.const(10.0)
        return relay.Function(relay.analysis.free_vars(add), add)

    passes = tvm.transform.Sequential(
        [
            relay.transform.InferType(),
            relay.transform.FoldConstant(),
        ]
    )

    before_mod = tvm.IRModule.from_expr(before())
    visualizer = RelayVisualizer()
    visualizer.visualize(before_mod, path="mnt/dropout.prototxt")


    with tvm.transform.PassContext(opt_level=3):
        after_mod = passes(before_mod)
    
    visualizer.visualize(after_mod, path="mnt/dropout_fold_constant.prototxt")
      
    tvm.ir.assert_structural_equal(run_infer_type(before_mod["main"]), after_mod["main"])

if __name__ == "__main__":
    test_fold_dropout()
