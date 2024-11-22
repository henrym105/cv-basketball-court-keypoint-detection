import torch
from torch.utils.data import DataLoader, Dataset
import json
import os
from torchvision import transforms
from ultralytics import YOLO  # Importing from ultralytics package

class SportCenterDataset(Dataset):
    def __init__(self, json_dir, transform=None):
        self.json_dir = json_dir
        self.transform = transform
        self.data = []
        for file in os.listdir(json_dir):
            if file.endswith('.json'):
                with open(os.path.join(json_dir, file)) as f:
                    self.data.append(json.load(f))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image_path = sample['filename']
        keypoints = sample['Hr']  # Assuming Hr contains keypoints
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, keypoints

# Define transformations
transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor(),
])

# Load dataset
dataset = SportCenterDataset('/Users/Henry/Desktop/github/cv-basketball-court-keypoint-detection/sportcenter_camerapose_dataset', transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Initialize model with pre-trained weights
model = YOLO('yolov11n.pt')  # Fetching YOLOv11n model weights from ultralytics

# Define loss and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for images, keypoints in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, keypoints)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the model
torch.save(model.state_dict(), 'yolov11n_keypoint_detection.pth')
