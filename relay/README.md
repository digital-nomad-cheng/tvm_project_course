## parse

1. Create tvm.IRModule from pytorch graph, tvm.IRModule is the input and output of relay passes
2. Build `convert_map` which records how pytorch operators maps to relay IR
3. Perform conversion
4. Call `transfrom.RemoveUnusedFunctions`   

### Notes

1. `def create_convert_map(self):` creates a dict for mapping from pytorch operators to relay IR.
2. 
