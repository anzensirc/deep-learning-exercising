import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons
from torchmetrics.classification import BinaryAccuracy

# -------------------------------------------------
# 1. Persiapan data
# -------------------------------------------------
torch.manual_seed(42)
num_data = 1000
X, y = make_moons(n_samples=num_data, noise=0.1, random_state=42)

# Visualisasi data awal
plt.scatter(X[y == 0, 0], X[y == 0, 1], color="blue", label="Class 0")
plt.scatter(X[y == 1, 0], X[y == 1, 1], color="yellow", label="Class 1")
plt.title("Data Make Moons")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

# -------------------------------------------------
# 2. Split data train/test & DataLoader
# -------------------------------------------------
batch_size = 50
train_ratio = 0.8
train_size = int(num_data * train_ratio)

x_train_pt = torch.tensor(X[:train_size], dtype=torch.float32)
y_train_pt = torch.tensor(y[:train_size], dtype=torch.float32).unsqueeze(1)

x_test_pt = torch.tensor(X[train_size:], dtype=torch.float32)
y_test_pt = torch.tensor(y[train_size:], dtype=torch.float32).unsqueeze(1)

train_set = TensorDataset(x_train_pt, y_train_pt)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

# -------------------------------------------------
# 3. Definisi model
# -------------------------------------------------
class SimpleANN(torch.nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(SimpleANN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size1)
        self.fc2 = torch.nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = torch.nn.Linear(hidden_size2, output_size)
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.tanh(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# -------------------------------------------------
# 4. Fungsi training
# -------------------------------------------------
def train_model(model, dataloader, criterion, optimizer, epochs=10):
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
            running_loss += loss.detach().cpu().numpy()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss}")
        loss_history.append(avg_loss)
    return loss_history

# -------------------------------------------------
# 5. Training model
# -------------------------------------------------
model_ann = SimpleANN(input_size=2, hidden_size1=10, hidden_size2=5, output_size=1)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model_ann.parameters(), lr=0.1)

loss_history = train_model(model_ann, train_loader, criterion, optimizer, epochs=100)

# Visualisasi loss
plt.plot(loss_history, color="green")
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# -------------------------------------------------
# 6. Evaluasi model
# -------------------------------------------------
model_ann.eval()
with torch.no_grad():
    y_pred = model_ann(x_test_pt)

metric = BinaryAccuracy(threshold=0.5)
accuracy = metric(y_pred, y_test_pt)

print(f"Accuracy: {accuracy}")

# -------------------------------------------------
# 7. Visualisasi Decision Boundary
# -------------------------------------------------
# Buat grid
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

# Prediksi untuk setiap titik di grid
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_tensor = torch.tensor(grid_points, dtype=torch.float32)
with torch.no_grad():
    Z = model_ann(grid_tensor)
Z = Z.reshape(xx.shape)

# Plot kontur decision boundary
plt.contourf(xx, yy, Z.numpy(), levels=50, cmap="RdBu", alpha=0.6)
plt.scatter(X[y == 0, 0], X[y == 0, 1], c="red", label="Class 0")
plt.scatter(X[y == 1, 0], X[y == 1, 1], c="blue", label="Class 1")
plt.title("Model Decision Boundary Contour on Moons Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()