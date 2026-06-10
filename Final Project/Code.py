import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. NFT Neural Network Architecture
# ==========================================
class NFTNet(nn.Module):
    def __init__(self, input_channels=2, hidden_channels=64, lstm_hidden=128):
        super(NFTNet, self).__init__()
        
        # Encoder: Conv1D (kernel=3, stride=2, padding=1, padding_mode='circular') + LeakyReLU
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels, kernel_size=3, stride=2, padding=1, padding_mode='circular'),
            nn.LeakyReLU(0.2)
        )
        
        # Bottleneck: LSTM
        self.lstm = nn.LSTM(input_size=hidden_channels, hidden_size=lstm_hidden, batch_first=True)
        
        # Decoder: ConvTranspose1D (kernel=3, stride=2, padding=1, output_padding=1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(lstm_hidden, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        # x shape: (Batch, Channels, Sequence_Length)
        encoded = self.encoder(x)
        
        # Prepare for LSTM: (Batch, Seq_Len, Channels)
        encoded = encoded.permute(0, 2, 1)
        lstm_out, _ = self.lstm(encoded)
        
        # Prepare for Decoder: (Batch, Channels, Seq_Len)
        lstm_out = lstm_out.permute(0, 2, 1)
        
        decoded = self.decoder(lstm_out)
        return decoded

# ==========================================
# 2. Synthetic Dataset Generation (Paper Specs)
# ==========================================
def generate_synthetic_data(num_samples, seq_length=256):
    """
    تولید داده تصادفی بر اساس پارامترهای مقاله:
    QAM 4/16/64, T0 in [0.7, 1.4], Subcarriers 32/64/128
    (در اینجا به صورت Tensor های تصادفی نرمالایز شده شبیه‌سازی می‌شود)
    """
    # X: Time domain signal (I and Q components -> 2 channels)
    # Y: Nonlinear spectrum / Ideal output
    X = torch.randn(num_samples, 2, seq_length) 
    Y = X + 0.1 * torch.sin(X) # شبیه‌سازی یک تبدیل غیرخطی ساده برای هدف آموزش
    return X, Y

# ==========================================
# 3. Training Setup & Loop
# ==========================================
# Hyperparameters from paper
EPOCHS = 200
LR = 3e-4
BATCH_SIZE = 64
SEQ_LENGTH = 256

# Generate small dataset for demonstration (In reality: 200,000 for train)
X_train, Y_train = generate_synthetic_data(1000, SEQ_LENGTH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NFTNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)

# RMSE Loss
def rmse_loss(y_pred, y_true):
    return torch.sqrt(torch.mean((y_pred - y_true)**2))

print(f"Training on {device}...")
loss_history = []

for epoch in range(1, EPOCHS + 1): # For fast demo in Colab, you can reduce this
    model.train()
    
    # Shuffle and batch (Simplified manual batching)
    permutation = torch.randperm(X_train.size()[0])
    epoch_loss = 0
    
    for i in range(0, X_train.size()[0], BATCH_SIZE):
        indices = permutation[i:i+BATCH_SIZE]
        batch_x, batch_y = X_train[indices].to(device), Y_train[indices].to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = rmse_loss(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    avg_loss = epoch_loss / (X_train.size()[0] / BATCH_SIZE)
    loss_history.append(avg_loss)
    
    if epoch % 20 == 0 or epoch == 1:
        print(f"Epoch [{epoch}/{EPOCHS}], RMSE Loss: {avg_loss:.4f}")

# ==========================================
# 4. Plotting the Results (For the Presentation)
# ==========================================
plt.figure(figsize=(12, 5))

# Plot 1: RMSE Loss Curve
plt.subplot(1, 2, 1)
plt.plot(loss_history, color='blue', linewidth=2)
plt.title('Training RMSE Loss over Epochs', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('RMSE', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

# Plot 2: Input vs Reconstructed Output (Real part of 1 sample)
model.eval()
with torch.no_grad():
    sample_x = X_train[0:1].to(device)
    sample_y = Y_train[0:1].cpu().numpy()
    pred_y = model(sample_x).cpu().numpy()

plt.subplot(1, 2, 2)
plt.plot(sample_y[0, 0, :], label='Ideal Spectral Data', color='green', alpha=0.7)
plt.plot(pred_y[0, 0, :], label='NFT-Net Output', color='red', linestyle='dashed')
plt.title('NFT Transform Output Comparison', fontsize=14)
plt.xlabel('Sequence Index', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
