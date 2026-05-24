import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
import torch.optim as optim
from models.dataset import get_dataloaders
from models.speech_pipeline.train import SpeechEmotionModel
from models.text_pipeline.train import TextEmotionModel
import argparse
from tqdm import tqdm

class FusionEmotionModel(nn.Module):
    def __init__(self, speech_model_path=None, text_model_path=None, num_classes=7):
        super(FusionEmotionModel, self).__init__()
        self.speech_model = SpeechEmotionModel()
        if speech_model_path and os.path.exists(speech_model_path):
            self.speech_model.load_state_dict(torch.load(speech_model_path, map_location='cpu'))
            print("Loaded pre-trained speech model.")
        
        self.text_model = TextEmotionModel(freeze_bert=True)
        if text_model_path and os.path.exists(text_model_path):
            self.text_model.load_state_dict(torch.load(text_model_path, map_location='cpu'))
            print("Loaded pre-trained text model.")
            
        # Freeze both base models
        for param in self.speech_model.parameters():
            param.requires_grad = False
        for param in self.text_model.parameters():
            param.requires_grad = False
            
        # Fusion Classifier
        # Speech feature dim: 128 * 2 = 256
        # Text feature dim: 768 (bert base)
        fusion_dim = 256 + 768
        
        self.fc = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, speech, input_ids, attention_mask):
        _, speech_features = self.speech_model(speech)
        _, text_features = self.text_model(input_ids, attention_mask)
        
        fused_features = torch.cat((speech_features, text_features), dim=1)
        out = self.fc(fused_features)
        return out, fused_features

def train(data_dir, speech_model_path, text_model_path, epochs=10, batch_size=32, lr=0.001):
    train_loader, test_loader = get_dataloaders(data_dir, batch_size=batch_size)
    model = FusionEmotionModel(speech_model_path, text_model_path)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=lr)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            speech = batch['speech'].to(device)
            input_ids = batch['text_input_ids'].to(device)
            attention_mask = batch['text_attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(speech, input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            pbar.set_postfix({'loss': f"{running_loss/len(pbar):.4f}"})
            
        print(f"Epoch [{epoch+1}/{epochs}] completed.")
        
    os.makedirs('models/fusion_pipeline/weights', exist_ok=True)
    torch.save(model.state_dict(), 'models/fusion_pipeline/weights/fusion_model.pth')
    print("Model saved to models/fusion_pipeline/weights/fusion_model.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help="Path to TESS dataset")
    parser.add_argument('--speech_model_path', type=str, default='models/speech_pipeline/weights/speech_model.pth')
    parser.add_argument('--text_model_path', type=str, default='models/text_pipeline/weights/text_model.pth')
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    train(args.data_dir, args.speech_model_path, args.text_model_path, epochs=args.epochs)
