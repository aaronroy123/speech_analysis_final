import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
import torch.optim as optim
from models.dataset import get_dataloaders
from transformers import AutoModel
import argparse
from tqdm import tqdm

class TextEmotionModel(nn.Module):
    def __init__(self, num_classes=7, freeze_bert=True):
        super(TextEmotionModel, self).__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
                
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.bert.config.hidden_size, num_classes)
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        out = self.fc(pooled_output)
        return out, pooled_output

def train(data_dir, epochs=10, batch_size=32, lr=0.001):
    train_loader, test_loader = get_dataloaders(data_dir, batch_size=batch_size)
    model = TextEmotionModel(freeze_bert=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=lr)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            input_ids = batch['text_input_ids'].to(device)
            attention_mask = batch['text_attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            pbar.set_postfix({'loss': f"{running_loss/len(pbar):.4f}"})
            
        print(f"Epoch [{epoch+1}/{epochs}] completed.")
        
    os.makedirs('models/text_pipeline/weights', exist_ok=True)
    torch.save(model.state_dict(), 'models/text_pipeline/weights/text_model.pth')
    print("Model saved to models/text_pipeline/weights/text_model.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help="Path to TESS dataset")
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    train(args.data_dir, epochs=args.epochs)
