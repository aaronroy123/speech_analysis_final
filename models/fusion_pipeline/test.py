import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
from models.dataset import get_dataloaders
from models.fusion_pipeline.train import FusionEmotionModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np
from sklearn.manifold import TSNE

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'pleasant_surprised', 'sad', 'neutral']

def test(data_dir, model_path):
    _, test_loader = get_dataloaders(data_dir, batch_size=32)
    # We do not need the base model paths to be valid here if we are just loading the whole state dict
    model = FusionEmotionModel()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_features = []
    
    with torch.no_grad():
        for batch in test_loader:
            speech = batch['speech'].to(device)
            input_ids = batch['text_input_ids'].to(device)
            attention_mask = batch['text_attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs, features = model(speech, input_ids, attention_mask)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_features.append(features.cpu().numpy())
            
    all_features = np.concatenate(all_features, axis=0)
    
    acc = accuracy_score(all_labels, all_preds)
    print(f"Fusion Model Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=EMOTIONS))
    
    # Plot Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.title('Fusion Pipeline Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('Results/plots/fusion_confusion_matrix.png')
    print("Saved confusion matrix to Results/plots/fusion_confusion_matrix.png")
    
    # TSNE visualization
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(all_features)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=all_labels, cmap='tab10', alpha=0.7)
    plt.legend(handles=scatter.legend_elements()[0], labels=EMOTIONS)
    plt.title('t-SNE of Fusion Emotion Features')
    plt.savefig('Results/plots/fusion_tsne.png')
    print("Saved t-SNE plot to Results/plots/fusion_tsne.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help="Path to TESS dataset")
    parser.add_argument('--model_path', type=str, default='models/fusion_pipeline/weights/fusion_model.pth')
    args = parser.parse_args()
    test(args.data_dir, args.model_path)
