import base64
from io import BytesIO
from PIL import Image
import torch
import torchvision.transforms as transforms
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ImageInput(BaseModel):
    file: str


# Load model
try:
    num_classes = 4
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
    model.load_state_dict(torch.load('mobilenet_model.pth', map_location=torch.device('cpu')))
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

LABELS = ["Grade 1", "Grade 2", "Grade 3", "Healthy"]


@app.post("/predict")
async def predict(image_data: ImageInput):
    try:
        img_bytes = base64.b64decode(image_data.file)
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

        percentages = (probabilities * 100).tolist()
        return {"probabilities": percentages, "labels": LABELS}
    except Exception as e:
        return {"error": str(e)}


# Run FastAPI app if executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)
