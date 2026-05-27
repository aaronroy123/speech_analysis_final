# Multimodal Emotion Recognition System Report

> **Execution Hint:** Step-by-step instructions on how to set up the environment, download the dataset, and execute the training/evaluation pipelines are provided in the [README.md](README.md) and [HOW_TO_RUN.txt](HOW_TO_RUN.txt) files at the root of the project repository.

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

To analyze the performance limits of each model, we conducted a granular diagnostic evaluation across all 1,120 test samples.

#### Text-Only Pipeline (Severe Representation Collapse)
The **Text-only** model fails universally (13.48% accuracy), predicting either `pleasant_surprised` or `neutral` for almost all inputs. This collapse is mathematically bound by the dataset's design: since the input transcripts are uniform carrier phrases (*"Say the word [word]"*), they carry zero direct semantic emotional indicators. Consequently, the classification head cannot learn any discriminative boundary over frozen BERT embeddings.

#### Speech-Only Pipeline (30 Failures / 1,120 Samples)
The Speech-only model exhibits strong classification capacity (97.32% accuracy) but yields 30 failure cases. These are categorized into distinct acoustic and phonetic profiles:
1. **Low-Arousal Acoustic Overlap (Sadness vs. Disgust)**: This represents the dominant failure mode, accounting for 18 of the 30 errors. True `sad` files (such as `YAF_tool_sad.wav`, `YAF_hurl_sad.wav`, `OAF_dog_sad.wav`, and `OAF_keg_sad.wav`) were misclassified as `disgust`. Both emotions share depressed voice intensity, slower speech rates, and a flat fundamental frequency ($f_0$) pitch envelope, causing their spectral trajectories to overlap.
2. **Neutral-to-Disgust Shift**: Flat, low-intensity neutral files (e.g., `OAF_late_neutral.wav` and `OAF_name_neutral.wav`) were predicted as `disgust`, demonstrating that flat pitch profiles can bleed into the low-arousal territory of disgust.
3. **High-Arousal Positives (Pleasant Surprise vs. Happy)**: High-arousal positive-valence samples were confused (e.g., `OAF_choice_ps.wav` and `OAF_hurl_ps.wav` misclassified as `happy`; `OAF_hire_happy.wav` misclassified as `pleasant_surprised`). Both categories display elevated pitch peaks and steep rising intonation contours, making their raw prosody highly similar.
4. **Phonetic-Induced Prosodic Bias**: Certain phonetic starts influenced the sequential model. For instance, `YAF_came_sad.wav` was misclassified as `happy`. The voiceless plosive `/k/` in "came" creates a rapid, high-energy acoustic burst on the spectrogram that the BiLSTM misidentified as a high-arousal happy onset.

#### Multimodal Late Fusion Pipeline (7 Failures / 1,120 Samples)
By incorporating lexical identities from the BERT embeddings, the late fusion model resolved 23 of the 30 speech-only errors (reducing the error rate to a mere 0.62%). The 7 remaining failures illustrate the edge-case limits of the system:
1. **Persistent High-Arousal Overlaps**: `OAF_nag_happy.wav` and `OAF_hire_happy.wav` were predicted as `pleasant_surprised`. Because the textual carrier phrase is identical, the fusion model must rely entirely on acoustic features to separate happy from pleasant surprise, which remain highly overlapping.
2. **Persistent Low-Arousal Overlaps**: `YAF_rough_sad.wav` and `YAF_rose_sad.wav` were misclassified as `disgust`. Despite the model knowing the lexical token ("rough", "rose"), the actor's vocal realization of sadness in these specific trials was acoustically indistinguishable from disgust.
3. **Extreme Prosodic Outliers**: `OAF_moon_angry.wav` and `OAF_rush_disgust.wav` were misclassified as `pleasant_surprised`. For `OAF_moon_angry.wav`, the vowel nasalization `/m/` and `/n/` combined with high-arousal anger to distort the vocal tract features, skewing them toward the signature acoustic envelope of pleasant surprise.


### 4. Separability Visualization (t-SNE)
We have generated t-SNE scatter plots mapping the high-dimensional features from each pipeline into a 2D space. The visualizations (`Results/plots/`) demonstrate how the clusters tighten or intermingle:
- **Temporal Modelling (Speech)** (`speech_tsne.png`): Shows clear clustering of the 7 emotions, verifying that Mel-spectrograms + BiLSTM successfully extract distinct emotional representations.
- **Contextual Modelling (Text)** (`text_tsne.png`): Shows a single massive overlapping blob, confirming that BERT embeddings for "Say the word X" do not naturally separate by emotion.
- **Fusion Block** (`fusion_tsne.png`): Shows perfectly tight and distant clusters for all 7 emotions, visually explaining the near-100% accuracy.
