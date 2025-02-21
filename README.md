# VANPY 
**VANPY** (Voice Analysis framework in Python) is a flexible and extensible framework for voice analysis, feature extraction, and model inference. It provides a modular pipeline architecture for processing audio segments with near- and state-of-the-art deep learning models.

![VANPY](https://github.com/griko/vanpy/raw/main/images/VANPY_architecture.png)

<!-- ## Quick Start
Try VANPY in Google Colab:

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/griko/VANPY/blob/main/examples/VANPY_example.ipynb)
  
  Basic VANPY capabilities demo

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/griko/VANPY/blob/main/examples/using_VANPY_to_classify_emotions_on_RAVDESS_dataset.ipynb)

  Emotion classification on RAVDESS dataset
 -->


## Architecture
**VANPY** consists of three optional pipelines that can be used independently or in combination:

1. **Preprocessing Pipeline**: Handles audio format conversion and voice segment extraction
2. **Feature Extraction Pipeline**: Generates feature/latent vectors from voice segments
3. **Model Inference Pipeline**

You can use these pipelines flexibly based on your needs:

- Use only preprocessing for voice separation
- Combine preprocessing and classification for direct audio analysis
- Use all pipelines for complete feature extraction and classification 

## Models Trained as part of the VANPY project

<table>
  <tr>
    <th>Task</th>
    <th>Dataset</th>
    <th>Performance</th>
  </tr>
  <tr>
    <td rowspan="3">Gender Identification (Accuracy)</td>
    <td>VoxCeleb2</td>
    <td>98.9%</td>
  </tr>
  <tr>
    <td>Mozilla Common Voice v10.0</td>
    <td>92.3%</td>
  </tr>
  <tr>
    <td>TIMIT</td>
    <td>99.6%</td>
  </tr>
  <tr>
    <td rowspan="2">Emotion Recognition (Accuracy)</td>
    <td>RAVDESS (8-class)</td>
    <td>84.71%</td>
  </tr>
  <tr>
    <td>RAVDESS (7-class)</td>
    <td>86.24%</td>
  </tr>
  <tr>
    <td rowspan="3">Age Estimation (MAE in years)</td>
    <td>VoxCeleb2</td>
    <td>7.88</td>
  </tr>
  <tr>
    <td>TIMIT</td>
    <td>4.95</td>
  </tr>
  <tr>
    <td>Combined VoxCeleb2-TIMIT</td>
    <td>6.93</td>
  </tr>
  <tr>
    <td rowspan="2">Height Estimation (MAE in cm)</td>
    <td>VoxCeleb2</td>
    <td>6.01</td>
  </tr>
  <tr>
    <td>TIMIT</td>
    <td>6.02</td>
  </tr>
</table>

All of the models can be used as a part of the VANPY pipeline or separately and are available on 🤗[HuggingFace](https://huggingface.co/griko)


## Configuration
### Environment Setup

1. Create a `pipeline.yaml` configuration file. You can use the `src/pipeline.yaml` as a template.
2. For HuggingFace models (Pyannote components), create a `.env` file:
```
huggingface_ACCESS_TOKEN=<your_token>
```
3. Pipelines examples are available in `src/run.py`.

## Components
Each component expects as an input and returns as an output a `ComponentPayload` object.

Each component supports:
- Batch processing (if applicable)
- Progress tracking
- Performance monitoring and logging
- Incremental processing (skip already processed files)
- GPU acceleration where applicable
- Configurable parameters

### Preprocessing Components

| Component | Description |
|-----------|-------------|
| **Filelist-DataFrame Creator** | Initializes data pipeline by creating a DataFrame of audio file paths. Supports both directory scanning and loading from existing CSV files. Manages path metadata for downstream components. |
| **WAV Converter** | Standardizes audio format to WAV with configurable parameters including bit rate (default: 256k), channels (default: mono), sample rate (default: 16kHz), and codec (default: PCM 16-bit). Uses FFMPEG for robust conversion. |
| **WAV Splitter** | Handles large audio files by splitting them into manageable segments based on either duration or file size limits. Maintains audio quality and creates properly labeled segments with original file references. |
| **INA Voice Separator** | Separates audio into voice and non-voice segments, distinguishing between male and female speakers. Filters out non-speech content while preserving speaker gender information. |
| **Pyannote VAD** | Performs Voice Activity Detection using Pyannote's state-of-the-art deep learning model. Identifies and extracts speech segments with configurable sensitivity.
| **Silero VAD** | Alternative Voice Activity Detection using Silero's efficient model. Optimized for real-time performance with customizable parameters. |
| **Pyannote SD** | Speaker Diarization component that identifies and separates different speakers in audio. Creates individual segments for each speaker with timing information. Supports overlapping speech handling. |
| **MetricGAN SE** | Speech Enhancement using MetricGAN+ model from SpeechBrain. Reduces background noise and improves speech clarity. |
| **SepFormer SE** | Speech Enhancement using SepFormer model, specialized in separating speech from complex background noise. |

### Feature Extraction Components

| Component | Description |
|-----------|-------------|
| **Librosa Features Extractor** | Comprehensive audio feature extraction using the Librosa library. Supports multiple feature types including: MFCC (Mel-frequency cepstral coefficients), Delta-MFCC, zero-crossing rate, spectral features (centroid, bandwidth, contrast, flatness), fundamental frequency (F0), and tonnetz. |
| **Pyannote Embedding** | Generates speaker embeddings using Pyannote's deep learning models. Uses sliding window analysis with configurable duration and step size. Outputs high-dimensional embeddings optimized for speaker differentiation. |
| **SpeechBrain Embedding** | Extracts neural embeddings using SpeechBrain's pretrained models, particularly the ECAPA-TDNN architecture (default: spkrec-ecapa-voxceleb). |

### Model Inference Components

| Component | Description |
|-----------|-------------|
| **VanpyGender Classifier** | SVM-based binary gender classification using speech embeddings. Supports two models: ECAPA-TDNN (192-dim) and XVECT (512-dim) embeddings from SpeechBrain. Trained on VoxCeleb2 dataset with optimized hyperparameters. Provides both verbal ('female'/'male') and numeric label options. |
| **VanpyAge Regressor** | Multi-architecture age estimation supporting SVR and ANN models. Features multiple variants: pure SpeechBrain embeddings (192-dim), combined SpeechBrain and Librosa features (233-dim), and dataset-specific models (VoxCeleb2/TIMIT). |
| **VanpyEmotion Classifier** | 7-class SVM emotion classifier trained on RAVDESS dataset using SpeechBrain embeddings. Classifies emotions into: angry, disgust, fearful, happy, neutral/calm, sad, surprised. |
| **IEMOCAP Emotion** | SpeechBrain-based emotion classifier trained on the IEMOCAP dataset. Uses Wav2Vec2 for feature extraction. Supports four emotion classes: angry, happy, neutral, sad. |
| **Wav2Vec2 ADV** | Advanced emotion analysis using Wav2Vec2, providing continuous scores for arousal, dominance, and valence dimensions. |
| **Wav2Vec2 STT** | Speech-to-text transcription using Facebook's Wav2Vec2 model. |
| **Whisper STT** | OpenAI's Whisper model for robust speech recognition. Supports multiple model sizes and languages. Includes automatic language detection. |
| **Cosine Distance Clusterer** | a Clustering method that can be used for speaker diarization using cosine similarity metrics. Groups speech segments by speaker identity using embedding similarity. |
| **GMM Clusterer** | Gaussian Mixture Model-based speaker clustering. |
| **Agglomerative Clusterer** | Hierarchical clustering for speaker diarization. Uses distance-based merging with configurable threshold and maximum clusters. |
| **YAMNet Classifier** | Google's YAMNet model for general audio classification. Supports 521 audio classes from AudioSet ontology. |


## ComponentPayload Structure
The `ComponentPayload` class manages data flow between pipeline components:
```
class ComponentPayload:
    metadata: Dict  # Pipeline metadata
    df: pd.DataFrame  # Processing results
```    
### Metadata fields
- `input_path`: Path to the input directory (required for `FilelistDataFrameCreator` if no `df` is provided)
- `paths_column`: Column name for audio file paths
- `all_paths_columns`: List of all path columns
- `feature_columns`: List of feature columns
- `meta_columns`: List of metadata columns
- `classification_columns`: List of classification columns

### df fields
- `df`: pd.DataFrame
  
  Includes all the collected information through the preprocessing and classification
  - each preprocessor adds a column of paths where the processed files are hold
  - embedding/feature extraction components add the embedding/features columns
  - each model adds a model-results column

### Key Methods
- `get_features_df()`: Extract features DataFrame
- `get_classification_df()`: Extract classification results DataFrame


## Coming Soon
- Custom classifier integration guide
- Additional preprocessing components
- Extended model support
- Newer python and dependencies version support

## Citing VANPY
Please, cite VANPY if you use it

```bibtex
@misc{vanpy,
  title={VANPY: Voice Analysis Framework},
  author={Gregory Koushnir, Michael Fire, Galit Fuhrmann Alpert, Dima Kagan},
  year={2025},
  eprint={TBD},
  archivePrefix={arXiv},
  primaryClass={TBD},
  note={arXiv:TBD}
}
```
