# Accuracy Tables

## Overall Performance

| Model Variant | Modalities Used | Overall Accuracy |
| :--- | :--- | :--- |
| **Speech-Only** | Mel-Spectrograms | **97.32%** |
| **Text-Only** | BERT Embeddings | **13.48%** |
| **Multimodal (Fusion)** | Speech + Text | **99.38%** |

## Per-Emotion Breakdown (F1-Score)

| Emotion | Speech-Only F1 | Text-Only F1 | Fusion F1 |
| :--- | :--- | :--- | :--- |
| **Angry** | 1.00 | 0.00 | 1.00 |
| **Disgust** | 0.93 | 0.00 | 0.99 |
| **Fear** | 1.00 | 0.00 | 1.00 |
| **Happy** | 0.98 | 0.08 | 0.99 |
| **Pleasant Surprise** | 0.98 | 0.23 | 0.98 |
| **Sad** | 0.94 | 0.00 | 0.99 |
| **Neutral** | 0.99 | 0.10 | 1.00 |

> Note: The extremely low accuracy of the Text-Only model is expected for the TESS dataset, as the transcripts strictly follow the carrier phrase pattern "Say the word [word]", containing no semantic emotional context.
