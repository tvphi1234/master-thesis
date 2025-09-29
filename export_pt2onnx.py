import torch

from utils import load_model, DEVICE, IMG_SIZE

MODEL_PATH = "./models/xception_best_20250826.pth"


model = load_model(model_name="xception", 
                   num_classes=2,
                   model_path=MODEL_PATH)

# convert model to onnx
onnx_path = MODEL_PATH.replace('.pth', '.onnx')
torch.onnx.export(
    model,                                    # Your trained PyTorch model
    torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE),  # Dummy input tensor
    onnx_path,                               # Output file path
    export_params=True,                       # Include model parameters
    opset_version=11                          # ONNX opset version
)