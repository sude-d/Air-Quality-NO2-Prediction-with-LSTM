import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)
y_train = torch.tensor(
    y_train,
    dtype=torch.float32
).unsqueeze(1)
input_size = X_train.shape[2]
hidden_size = 64
num_layers = 2
dropout = 0.2
output_size = 1
learning_rate = 0.001
num_epochs = 5
class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        output_size,
        dropout
    ):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(
            hidden_size,
            output_size
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
model = LSTMModel(
    input_size,
    hidden_size,
    num_layers,
    output_size,
    dropout
)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate
)
loss_list = []
for epoch in range(num_epochs):
    model.train()
    output = model(X_train)
    loss = criterion(
        output,
        y_train
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_list.append(
        loss.item()
    )
    print(
        f"Epoch: {epoch + 1}/{num_epochs}, Loss: {loss.item():.6f}"
    )
torch.save(
    model.state_dict(),
    "lstm_air_quality_model.pth"
)
plt.figure(figsize=(10, 5))
plt.plot(
    range(1, num_epochs + 1),
    loss_list,
    marker="o"
)
plt.title(
    "LSTM Eğitim Kaybı"
)
plt.xlabel(
    "Epoch"
)
plt.ylabel(
    "MSE Loss"
)
plt.grid(True)
plt.tight_layout()
plt.show()
print("Model kaydedildi: lstm_air_quality_model.pth")