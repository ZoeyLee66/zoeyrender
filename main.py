import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


class IrisNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )

    def forward(self, x):
        return self.net(x)


def main():
    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    model = IrisNet()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, 201):
        model.train()
        logits = model(X_train_t)
        loss = loss_fn(logits, y_train_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_t)
                val_pred = torch.argmax(val_logits, dim=1)
                acc = (val_pred == y_val_t).float().mean().item()
            print(f"Epoch {epoch:3d} | loss {loss.item():.4f} | val acc {acc * 100:.1f}%")

    # Save model weights + scaler
    torch.save(model.state_dict(), "iris_model.pth")
    joblib.dump(scaler, "iris_scaler.joblib")
    print("Saved iris_model.pth and iris_scaler.joblib")


if __name__ == "__main__":
    main()
