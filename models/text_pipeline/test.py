import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
from models.dataset import get_dataloaders
from models.text_pipeline.train import TextEmotionModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np
from sklearn.manifold import TSNE

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'pleasant_surprised', 'sad', 'neutral']

def test(data_dir, model_path):
    _, test_loader = get_dataloaders(data_dir, batch_size=32)
    model = TextEmotionModel(freeze_bert=True)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_features = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['text_input_ids'].to(device)
            attention_mask = batch['text_attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs, features = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_features.append(features.cpu().numpy())
            
    all_features = np.concatenate(all_features, axis=0)
    
    acc = accuracy_score(all_labels, all_preds)
    print(f"Text Model Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=EMOTIONS))
    
    # Plot Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.title('Text Pipeline Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('Results/plots/text_confusion_matrix.png')
    print("Saved confusion matrix to Results/plots/text_confusion_matrix.png")
    
    # TSNE visualization
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(all_features)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=all_labels, cmap='tab10', alpha=0.7)
    plt.legend(handles=scatter.legend_elements()[0], labels=EMOTIONS)
    plt.title('t-SNE of Text Emotion Features')
    plt.savefig('Results/plots/text_tsne.png')
    print("Saved t-SNE plot to Results/plots/text_tsne.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help="Path to TESS dataset")
    parser.add_argument('--model_path', type=str, default='models/text_pipeline/weights/text_model.pth')
    args = parser.parse_args()
    test(args.data_dir, args.model_path)
