import onnxruntime as ort

MODEL_PATH = "onnx_models/palm_detection_mediapipe_2023feb.onnx"

session = ort.InferenceSession(
    MODEL_PATH,
    providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
)

print("Providers actually being used:", session.get_providers())
print("\nModel inputs:")
for inp in session.get_inputs():
    print(f"  name={inp.name}, shape={inp.shape}, type={inp.type}")

print("\nModel outputs:")
for out in session.get_outputs():
    print(f"  name={out.name}, shape={out.shape}, type={out.type}")
