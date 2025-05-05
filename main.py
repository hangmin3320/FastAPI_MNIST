from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import torch
from torchvision import transforms
from PIL import Image
import io
import torch.nn as nn
import torch.nn.functional as F
import base64
import numpy as np

app = FastAPI()

templates = Jinja2Templates(directory="templates")


# 모델 클래스 정의
class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 10)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)

    def forward(self, x):
        x = self.pool(F.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(F.relu(self.batch_norm2(self.conv2(x))))
        x = self.pool(F.relu(self.batch_norm3(self.conv3(x))))
        x = self.dropout(x)
        x = x.view(-1, 128 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

device = torch.device("cpu")
model_path = "Model/result/MNIST_CNN.pth"

model = MNIST_CNN().to(device)
state_dict = torch.load(model_path, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.eval()


# 이미지 전처리
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 1.0 - x),
    transforms.Normalize((0.5,), (0.5,))
])


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # 이미지 전처리
        img_tensor = transform(image).unsqueeze(0).to(device)

        # 전처리된 이미지 시각화를 위한 처리
        # 텐서를 이미지로 변환 (첫 번째 배치, 첫 번째 채널)
        processed_img = img_tensor[0, 0].cpu().numpy()

        # 정규화 [-1, 1] -> [0, 1] 범위로 변환
        processed_img = (processed_img + 1) / 2

        # [0, 1] -> [0, 255] 범위로 변환하고 uint8로 변환
        processed_img = (processed_img * 255).astype(np.uint8)

        # PIL 이미지로 변환
        processed_pil_img = Image.fromarray(processed_img)

        # 이미지를 base64로 인코딩
        buffered = io.BytesIO()
        processed_pil_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # 모델 추론
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, 1).item()  # 가장 높은 확률의 클래스 반환
            confidence = probabilities[0, predicted_class].item()  # 해당 클래스의 확률 추출

        return JSONResponse(content={
            "predicted_class": predicted_class,
            "confidence": f"{confidence * 100:.2f}%",
            "detail": f"Predicted digit is {predicted_class} with {confidence * 100:.2f}% confidence",
            "processed_image": f"data:image/png;base64,{img_str}"
        })

    except Exception as e:
        return JSONResponse(content={
            "error": str(e)
        }, status_code=500)