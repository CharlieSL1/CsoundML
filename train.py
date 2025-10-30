from torch._tensor import Tensor
import torch 
from torch import nn , optim
from tqdm import tqdm
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import json
from pathlib import Path

# ====================Data Loading====================
# Load preprocessed data
preprocessed_dir = Path("preprocessed")
snaps_dir = Path("preprocessed_snaps")

# Load mel-spectrograms
mel_files = list(preprocessed_dir.glob("*.npy"))
print(f"Found {len(mel_files)} mel-spectrogram files")

# Load corresponding snap parameters
snap_files = list(snaps_dir.glob("*.json"))
print(f"Found {len(snap_files)} snap parameter files")

# Create X (mel-spectrograms) and y (snap parameters)
X_data = []
y_data = []

def extract_numeric_values(data):
    """Extract only numeric values from snap parameters"""
    numeric_values = []
    
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (int, float)):
                numeric_values.append(float(value))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, (int, float)):
                        numeric_values.append(float(item))
            elif isinstance(value, dict):
                # Recursively extract from nested dictionaries
                numeric_values.extend(extract_numeric_values(value))
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (int, float)):
                numeric_values.append(float(item))
    
    return numeric_values

for mel_file in mel_files:
    # Load mel-spectrogram
    mel = np.load(mel_file)
    
    # Find corresponding snap file
    snap_file = snaps_dir / (mel_file.stem + ".json")
    if snap_file.exists():
        with open(snap_file, 'r') as f:
            snap_params = json.load(f)
        
        # Extract only numeric values
        numeric_values = extract_numeric_values(snap_params)
        
        if len(numeric_values) > 0:  # Only add if we have numeric values
            X_data.append(torch.tensor(mel, dtype=torch.float32))
            y_data.append(torch.tensor(numeric_values, dtype=torch.float32))

print(f"Loaded {len(X_data)} samples")
if len(X_data) > 0:
    print(f"Mel shape: {X_data[0].shape}")
    print(f"Snap shape: {y_data[0].shape}")
    
    # Find max parameter count
    max_params = max(len(y) for y in y_data)
    min_params = min(len(y) for y in y_data)
    print(f"Parameter counts - Min: {min_params}, Max: {max_params}")

# Convert to tensors
if len(X_data) > 0:
    # Stack X (mel-spectrograms) - they all have the same shape
    X = torch.stack(X_data)
    
    # Pad y (snap parameters) to same length
    padded_y_data = []
    for y_tensor in y_data:
        pad_size = max_params - len(y_tensor)
        if pad_size > 0:
            padded_y = torch.cat([y_tensor, torch.zeros(pad_size)])
        else:
            padded_y = y_tensor
        padded_y_data.append(padded_y)
    
    y = torch.stack(padded_y_data)
    
    X = X.transpose(1, 2)
    N, L, Hin = X.shape
    print(f"Final X shape: {X.shape}, y shape: {y.shape}")
else:
    raise ValueError("No valid data found! Please check that preprocessed data and snap files exist in the correct directories.")

batchsize = 2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Create data loaders
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)

# ====================Model Architecture====================
Hout = 64
output_dim = y.shape[1]
# Initialize weights
Wx = torch.randn(Hout, Hin) * 0.1
bx = torch.zeros(Hout)
Wh = torch.randn(Hout, Hout) * 0.1
bh = torch.zeros(Hout)

# ====================Model====================
backbone = nn.RNN(input_size=Hin, hidden_size=Hout, batch_first=True)
backbone.weight_ih_l0.data = Wx
backbone.bias_ih_l0.data = bx
backbone.weight_hh_l0.data = Wh
backbone.bias_hh_l0.data = bh

head = nn.Linear(in_features=Hout, out_features=output_dim)

model = nn.ModuleList([backbone, head])

# ====================Training====================

loss_fn = nn.MSELoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Starting training...")
for epoch in tqdm(range(1000)):  
    model.train()
    total_loss = 0
    
    for xb, yb in train_loader:
        ht_1 = torch.zeros(1, xb.size(0), Hout)
        
        output, ht = backbone(xb, ht_1)
        
        y_pred = head(ht.squeeze(0))  
        
        loss = loss_fn(y_pred, yb)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)

# ====================Saving the Model====================

models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

model_save_path = models_dir / "trained_model.pt"
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': avg_loss,
    'epoch': 5,
    'model_config': {
        'Hin': Hin,
        'Hout': Hout,
        'output_dim': output_dim,
        'L': L
    }
}, model_save_path)

complete_model_path = models_dir / "model_complete.pt"
torch.save(model, complete_model_path)




