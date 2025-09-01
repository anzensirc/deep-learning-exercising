import torch, torchvision
from torchvision import datasets, transforms
from torch import nn, optim

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Data augmentasi dan loader
transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # untuk RGB
])

train_ds = datasets.ImageFolder(r".\train", transform=transform)
test_ds  = datasets.ImageFolder(r".\test", transform=transform)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_ds, batch_size=32)

# 2. Model CNN kecil
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,1,1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,1,1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*16*16, 128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

num_classes = len(train_ds.classes)
model = SimpleCNN(num_classes).to(device)

# 3. Loss dan optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 4. Training loop
for epoch in range(10):
    model.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluasi pada test set
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    print(f"Epoch {epoch+1} — Test Accuracy: {correct/total:.3f}")

