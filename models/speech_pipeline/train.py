import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
import torch.optim as optim
from models.dataset import get_dataloaders
import argparse
from tqdm import tqdm

class SpeechEmotionModel(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, num_classes=7):
        super(SpeechEmotionModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        # x shape: [batch, time_steps, features]
        out, (hn, cn) = self.lstm(x)
        # We take the final hidden state
        # hn is of shape [num_layers * num_directions, batch, hidden_size]
        # We concatenate the last hidden state of forward and backward direction
        hn_cat = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1)
        out = self.fc(hn_cat)
        return out, hn_cat

def train(data_dir, epochs=10, batch_size=32, lr=0.001):
    train_loader, test_loader = get_dataloaders(data_dir, batch_size=batch_size)
    model = SpeechEmotionModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            speech = batch['speech'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(speech)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            pbar.set_postfix({'loss': f"{running_loss/len(pbar):.4f}"})
            
        print(f"Epoch [{epoch+1}/{epochs}] completed.")
        
    os.makedirs('models/speech_pipeline/weights', exist_ok=True)
    torch.save(model.state_dict(), 'models/speech_pipeline/weights/speech_model.pth')
    print("Model saved to models/speech_pipeline/weights/speech_model.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help="Path to TESS dataset")
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    train(args.data_dir, epochs=args.epochs)
