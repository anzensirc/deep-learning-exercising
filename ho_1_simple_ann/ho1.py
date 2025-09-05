# -------------------------------------------------
# Modul 01 - Eksplorasi ANN untuk Klasifikasi Biner
# -------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_circles, make_classification
from torchmetrics.classification import BinaryAccuracy

# -------------------------------------------------
# 1. Pilih Dataset
# -------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)

# Pilih salah satu dataset: "circles" atau "classification"
dataset_choice = "circles"  

num_data = 500
if dataset_choice == "circles":
    X, y = make_circles(n_samples=num_data, noise=0.02, random_state=42)
elif dataset_choice == "classification":
    X, y = make_classification(
        n_samples=num_data,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_classes=2,
        random_state=42
    )
else:
    raise ValueError("Pilihan dataset tidak valid!")

# Visualisasi data awal
plt.scatter(X[y == 0, 0], X[y == 0, 1], color="blue", label="Class 0")
plt.scatter(X[y == 1, 0], X[y == 1, 1], color="orange", label="Class 1")
plt.title(f"Data: {dataset_choice}")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

# -------------------------------------------------
# 2. Split data train/test & DataLoader
# -------------------------------------------------
train_size = 400
test_size = num_data - train_size
batch_size = 50

x_train_pt = torch.tensor(X[:train_size], dtype=torch.float32)
y_train_pt = torch.tensor(y[:train_size], dtype=torch.float32).unsqueeze(1)

x_test_pt = torch.tensor(X[train_size:], dtype=torch.float32)
y_test_pt = torch.tensor(y[train_size:], dtype=torch.float32).unsqueeze(1)

train_set = TensorDataset(x_train_pt, y_train_pt)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

# -------------------------------------------------
# 3. Definisi model ANN (Eksplorasi)
# -------------------------------------------------
class SimpleANN(torch.nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(SimpleANN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size1)
        self.fc2 = torch.nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = torch.nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = torch.nn.Linear(hidden_size3, output_size)

        # Aktivasi
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.leakyrelu = torch.nn.LeakyReLU(0.1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))       # Layer 1 pakai ReLU
        x = self.tanh(self.fc2(x))       # Layer 2 pakai Tanh
        x = self.leakyrelu(self.fc3(x))  # Layer 3 pakai LeakyReLU
        x = self.sigmoid(self.fc4(x))    # Output pakai Sigmoid
        return x

# -------------------------------------------------
# 4. Fungsi Training
# -------------------------------------------------
def train_model(model, dataloader, criterion, optimizer, epochs=50):
    model.train()
    loss_history = []
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    return loss_history

# -------------------------------------------------
# 5. Training Model
# -------------------------------------------------
model_ann = SimpleANN(input_size=2, hidden_size1=16, hidden_size2=12, hidden_size3=8, output_size=1)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model_ann.parameters(), lr=0.01)

loss_history = train_model(model_ann, train_loader, criterion, optimizer, epochs=100)

# Plot loss
plt.plot(loss_history, color="green")
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# -------------------------------------------------
# 6. Evaluasi Model
# -------------------------------------------------
model_ann.eval()
with torch.no_grad():
    y_pred = model_ann(x_test_pt)

metric = BinaryAccuracy(threshold=0.5)
accuracy = metric(y_pred, y_test_pt)
print(f"Test Accuracy: {accuracy:.4f}")

# -------------------------------------------------
# 7. Visualisasi Decision Boundary
# -------------------------------------------------
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_tensor = torch.tensor(grid_points, dtype=torch.float32)
with torch.no_grad():
    Z = model_ann(grid_tensor)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z.numpy(), levels=50, cmap="RdBu", alpha=0.6)
plt.scatter(X[y == 0, 0], X[y == 0, 1], c="blue", label="Class 0")
plt.scatter(X[y == 1, 0], X[y == 1, 1], c="orange", label="Class 1")
plt.title("Decision Boundary - ANN")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
