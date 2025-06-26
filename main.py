import pandas as pd
import numpy as np
import time
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import joblib

# 1. Data Loading
data = pd.read_csv('data/flow_data.csv')

categorical_features = ['protocol', 'service']
continuous_features = ['duration', 'src_bytes', 'dst_bytes']

scaler = MinMaxScaler()
data[continuous_features] = scaler.fit_transform(data[continuous_features])
joblib.dump(scaler, 'models/scaler.pkl')

# 2. Categorical Model
class OPSiFiCategorical:
    def __init__(self):
        self.counts = defaultdict(lambda: defaultdict(int))
        self.total = defaultdict(int)

    def update(self, features):
        for key, value in features.items():
            self.counts[key][value] += 1
            self.total[key] += 1

    def anomaly_score(self, features):
        score = 0
        for key, value in features.items():
            prob = self.counts[key][value] / self.total[key] if self.total[key] > 0 else 0
            score += -np.log(prob + 1e-6)
        return score

cat_model = OPSiFiCategorical()

# 3. GRU Autoencoder
class GRUAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUAutoencoder, self).__init__()
        self.encoder = nn.GRU(input_size, hidden_size, batch_first=True)
        self.decoder = nn.GRU(hidden_size, input_size, batch_first=True)

    def forward(self, x):
        _, h = self.encoder(x)
        out, _ = self.decoder(h.repeat(x.size(1), 1, 1).transpose(0, 1))
        return out

input_size = len(continuous_features)
model = GRUAutoencoder(input_size, hidden_size=8)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Train Autoencoder
def train_autoencoder(model, data, epochs=20):
    X = data[continuous_features].values.reshape(-1, 1, len(continuous_features))
    X = torch.tensor(X, dtype=torch.float32)

    for epoch in range(epochs):
        output = model(X)
        loss = criterion(output, X)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), 'models/autoencoder_model.pth')

train_autoencoder(model, data)

# 5. Streaming + Anomaly Detection
def stream_data(data, delay=0.5):
    for _, row in data.iterrows():
        yield row
        time.sleep(delay)

print("\n🚀 Streaming started...\n")
for row in stream_data(data):
    row = dict(row)

    cat_input = {k: row[k] for k in categorical_features}
    cat_model.update(cat_input)
    cat_score = cat_model.anomaly_score(cat_input)

    cont_input = np.array([row[k] for k in continuous_features]).reshape(1, 1, -1)
    cont_input = torch.tensor(cont_input, dtype=torch.float32)

    output = model(cont_input)
    cont_score = criterion(output, cont_input).item()

    final_score = 0.5 * cat_score + 0.5 * cont_score

    status = "🚨 Anomaly Detected" if final_score > 5 else "✅ Normal Traffic"

    print(f"[STATUS] {status} | Cat Score: {cat_score:.2f} | Cont Score: {cont_score:.2f} | Final: {final_score:.2f}")