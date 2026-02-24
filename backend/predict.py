import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import json
import io
import base64


class CNN(nn.Module):
    def __init__(self, num_classes=47):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.pool  = nn.MaxPool2d(2, 2)
        self.drop  = nn.Dropout(0.4)
        self.fc1   = nn.Linear(128 * 3 * 3, 256)
        self.fc2   = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.drop(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CNN(num_classes=47).to(device)
model.load_state_dict(torch.load('../model/model.pth', map_location=device))
model.eval()

with open('../model/label_map.json') as f:
    label_map = json.load(f)


def preprocess(image_data: str) -> torch.Tensor:
    if ',' in image_data:
        image_data = image_data.split(',')[1]

    img = Image.open(io.BytesIO(base64.b64decode(image_data))).convert('L')
    img = img.resize((28, 28), Image.LANCZOS)

    arr = np.array(img, dtype=np.float32) / 255.0

    bg_val = float(np.percentile(arr, 75))
    if bg_val > 0.5:
        arr = 1.0 - arr

    tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0).to(device)
    return tensor


def predict(image_data: str) -> list:
    tensor = preprocess(image_data)

    with torch.no_grad():
        out   = model(tensor)
        probs = F.softmax(out, dim=1)[0]

    top3 = torch.topk(probs, 3)

    results = []
    for prob, idx in zip(top3.values.tolist(), top3.indices.tolist()):
        results.append({
            'label': label_map[str(idx)],
            'confidence': round(prob * 100, 2)
        })

    return results