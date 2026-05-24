# Multimodal Emotion Recognition System Report

## A. Architecture Decisions

### 1. Preprocessing
**Speech Modality**: We use `torchaudio` to load the audio files and ensure a standard 16 kHz sampling rate via resampling. We pad or truncate the waveforms to 64000 samples (4 seconds) to maintain consistent batch shapes. We extract **Mel Spectrograms** (64 mel bins) because they represent how human ears perceive sound logarithmically, providing a rich, time-frequency representation for the downstream sequential model.

**Text Modality**: Given the TESS dataset carrier phrase structure ("Say the word [word]"), we derive transcripts directly from the audio filenames. We utilize the `bert-base-uncased` tokenizer to convert these phrases into tokenized IDs and attention masks with a max length of 16.

### 2. Feature Extraction
**Speech**: The Mel spectrograms serve as our dense time-series features.
**Text**: We utilize a pre-trained **BERT base** model to extract contextual word embeddings. BERT provides deeply contextualized representations that excel at capturing linguistic nuance. 

### 3. Temporal/Contextual Modelling
**Speech Modality**: We employ a **Bidirectional LSTM (BiLSTM)** (2 layers, 128 hidden units). Speech is sequential, and emotions are often conveyed through prosodic variations over time. A BiLSTM captures these temporal dependencies from both past and future contexts within the utterance. The final hidden states are concatenated to form a 256-dimensional summary vector.

**Text Modality**: We use the pre-trained `bert-base-uncased` model to process the tokens. The pooled output (the `[CLS]` token representation) is used as it aggregates the contextual meaning of the entire short sentence. We freeze the BERT weights to ensure stable and efficient training, given the simplicity of the carrier phrases.

### 4. Fusion Block
We implemented **Late Fusion**. The 256-dimensional summary from the Speech BiLSTM and the 768-dimensional `[CLS]` embedding from BERT are concatenated to form a 1024-dimensional fused representation. Late fusion allows each modality pipeline to specialize independently before their high-level representations are jointly mapped to the emotion categories.

### 5. Classifier
Each pipeline (Speech, Text, Fusion) ends with a sequence of fully connected linear layers (with Dropout for regularization) culminating in an output layer of size 7, corresponding to the TESS emotion classes.

---

## B. Experiments

We conducted experiments across three variants on a 80/20 train/test split:
1. **Speech-only**: BiLSTM trained on Mel-spectrograms. **Accuracy: 97.32%**
2. **Text-only**: Classification head trained on top of frozen BERT `[CLS]` embeddings. **Accuracy: 13.48%**
3. **Multimodal (Fusion)**: Classifier trained on concatenated features from frozen Speech and Text models. **Accuracy: 99.38%**

---

## C. Analysis

### 1. Which emotions are easiest/hardest to classify? Why?
In the **Speech-only** model, `fear`, `angry`, and `neutral` were the easiest to classify, all achieving an F1-score of 1.00 or 0.99. These emotions have very distinct acoustic and prosodic features (e.g., high pitch and energy for anger; flat pitch for neutral).
The hardest emotions were `disgust` (F1 = 0.93) and `sad` (F1 = 0.94), likely because their acoustic profiles (e.g., lower energy, slower tempo) can sometimes overlap or be subtler in carrier phrases compared to highly active emotions.

### 2. When does fusion help most?
Fusion improved the overall accuracy from 97.32% to **99.38%**. It specifically helped boost the F1-scores for the hardest emotions (`disgust` improved from 0.93 to 0.99; `sad` improved from 0.94 to 0.99). 
Interestingly, the Text modality alone performs terribly (13.48%, akin to random guessing across 7 classes). This is because the TESS dataset text consists entirely of the carrier phrase *"Say the word [word]"*. There is zero emotional semantic content in the text. However, in the Fusion model, the text embeddings provide the exact identity of the spoken word. The Fusion network learns to contextualize the acoustic variations *conditioned on* the specific word being spoken, effectively eliminating phonetic ambiguities that the Speech-only model struggled with.

### 3. Error analysis: Failure cases
The **Text-only** model failed universally, overwhelmingly predicting `pleasant_surprised` or `neutral` for almost everything because the text itself lacks emotional signal.
In the **Speech-only** model, the few failure cases involved confusing `disgust` with `sadness` (both are negative valence, lower arousal emotions). Once the identity of the spoken word was provided via the Text modality in the Fusion model, these overlapping acoustic features were disambiguated.

### 4. Separability Visualization (t-SNE)
We have generated t-SNE scatter plots mapping the high-dimensional features from each pipeline into a 2D space. The visualizations (`Results/plots/`) demonstrate how the clusters tighten or intermingle:
- **Temporal Modelling (Speech)** (`speech_tsne.png`): Shows clear clustering of the 7 emotions, verifying that Mel-spectrograms + BiLSTM successfully extract distinct emotional representations.
- **Contextual Modelling (Text)** (`text_tsne.png`): Shows a single massive overlapping blob, confirming that BERT embeddings for "Say the word X" do not naturally separate by emotion.
- **Fusion Block** (`fusion_tsne.png`): Shows perfectly tight and distant clusters for all 7 emotions, visually explaining the near-100% accuracy.
