import tvm
from tvm import relay
from tvm.relay import testing
from relay_parse import load_pretrained_model, import_pytorch_to_relay

def list_ops(expr):
    class OpLister(tvm.relay.ExprVisitor):
        def visit_op(self, expr):
            if expr not in self.node_memo_map:
                self.node_list.append(expr)
            return super().visit_op(expr)
        
        def list_nodes(self, expr):
            self.node_memo_map = {}
            self.node_list = []
            self.visit(expr)
            return self.node_list
    
    ins = OpLister()
    return ins.list_nodes(expr)

scripted_model = load_pretrained_model()
mod, params = import_pytorch_to_relay(scripted_model)

op_names = list_ops(mod["main"])
print(op_names)

