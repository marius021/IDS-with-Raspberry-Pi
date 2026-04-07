import onnx
from onnx import helper, TensorProto, shape_inference

src = "ids_mlp_binary.onnx"
dst = "ids_mlp_binary_logits.onnx"

model = onnx.load(src)
graph = model.graph

# ștergem output-urile curente
del graph.output[:]

# adăugăm tensorul logits ca output oficial
graph.output.append(
    helper.make_tensor_value_info("/4/Gemm_output_0", TensorProto.FLOAT, [None, 1])
)

# inferăm shape-urile
model = shape_inference.infer_shapes(model)

onnx.save(model, dst)
print(f"Saved {dst}")