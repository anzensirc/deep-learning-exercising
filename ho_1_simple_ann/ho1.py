import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_circles, make_classification
from torchmetrics.classification import BinaryAccuracy

torch.manual_seed(42)

# -------------------------------------------------
# 1. Pilih Dataset
# -------------------------------------------------
# Dataset 1: make_circles
X, y = make_circles(n_samples=500, noise=0.02, random_state=42)

# Dataset 2 (opsional): make_classification
# X, y = make_classification(
#     n_samples=500, n_features=2, n_informative=2,
#     n_redundant=0, n_classes=2, random_state=42
# )

# Visualisasi data
plt.scatter(X[y == 0, 0], X[y == 0, 1], color="red", label="Class 0")
plt.scatter(X[y == 1, 0], X[y == 1, 1], color="blue", label="Class 1")
plt.title("Data Distribusi (make_circles)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

# -------------------------------------------------
# 2. Split Train/Test
# -------------------------------------------------
train_size = 400
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# Konversi ke tensor
x_train_pt = torch.tensor(X_train, dtype=torch.float32)
y_train_pt = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
x_test_pt = torch.tensor(X_test, dtype=torch.float32)
y_test_pt = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# DataLoader
train_set = TensorDataset(x_train_pt, y_train_pt)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

# -------------------------------------------------
# 3. Model ANN Fleksibel
# -------------------------------------------------
class FlexibleANN(torch.nn.Module):
    def __init__(self, input_size, hidden_layers, hidden_neurons, output_size):
        super(FlexibleANN, self).__init__()
        layers = []
        in_features = input_size
        for _ in range(hidden_layers):
            layers.append(torch.nn.Linear(in_features, hidden_neurons))
            layers.append(torch.nn.ReLU())
            in_features = hidden_neurons
        layers.append(torch.nn.Linear(in_features, output_size))  # tanpa Sigmoid
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# -------------------------------------------------
# 4. Fungsi Training dengan Early Stopping
# -------------------------------------------------
def train_model(model, dataloader, criterion, optimizer, epochs=100, patience=10):
    model.train()
    loss_history = []
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        loss_history.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return loss_history

# -------------------------------------------------
# 5. Evaluasi Akurasi
# -------------------------------------------------
def evaluate(model, x_test, y_test):
    model.eval()
    with torch.no_grad():
        y_pred = torch.sigmoid(model(x_test))
    metric = BinaryAccuracy(threshold=0.5).to(device)
    return metric(y_pred, y_test).item()

# -------------------------------------------------
# 6. GPU Support
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training on:", device)

x_train_pt, y_train_pt = x_train_pt.to(device), y_train_pt.to(device)
x_test_pt, y_test_pt = x_test_pt.to(device), y_test_pt.to(device)

# -------------------------------------------------
# 7. Training Eksperimen
# -------------------------------------------------
# Model dengan 1 hidden layer, 8 neuron
model1 = FlexibleANN(input_size=2, hidden_layers=1, hidden_neurons=8, output_size=1).to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.01)
loss1 = train_model(model1, train_loader, criterion, optimizer1, epochs=100, patience=15)

# Model dengan 2 hidden layers, 16 neuron
model2 = FlexibleANN(input_size=2, hidden_layers=2, hidden_neurons=16, output_size=1).to(device)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.01)
loss2 = train_model(model2, train_loader, criterion, optimizer2, epochs=100, patience=15)

# -------------------------------------------------
# 8. Evaluasi Akurasi
# -------------------------------------------------
acc1 = evaluate(model1, x_test_pt, y_test_pt)
acc2 = evaluate(model2, x_test_pt, y_test_pt)

print(f"Akurasi Model 1 hidden layer: {acc1:.4f}")
print(f"Akurasi Model 2 hidden layers: {acc2:.4f}")

# -------------------------------------------------
# 9. Visualisasi Loss
# -------------------------------------------------
plt.plot(loss1, label="1 Hidden Layer (8 Neuron)", color="red")
plt.plot(loss2, label="2 Hidden Layers (16 Neuron)", color="blue")
plt.title("Training Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# -------------------------------------------------
# 10. Visualisasi Decision Boundary
# -------------------------------------------------
def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)
    with torch.no_grad():
        Z = torch.sigmoid(model(grid)).reshape(xx.shape).cpu()
    plt.contourf(xx, yy, Z.numpy(), levels=np.linspace(0,1,30), cmap="RdBu", alpha=0.6)
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c="red", label="Class 0")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c="blue", label="Class 1")
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()

plot_decision_boundary(model1, X, y, f"Decision Boundary - Model1 (Acc={acc1:.2f})")
plot_decision_boundary(model2, X, y, f"Decision Boundary - Model2 (Acc={acc2:.2f})")
