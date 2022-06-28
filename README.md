# vanpy 
**vanpy** is a Voice Analysis framework, built in a 
flexible manner for easy extendability, that allows to extract and classify voice segments. 

![vanpy](https://user-images.githubusercontent.com/1709151/176196445-d80903ec-295f-4bc2-8a82-7f25f9f8dfc1.PNG)

## Examples
You can use the following google colab notebook to inspect the capabilities of the library
 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/griko/voice_characterizer/blob/main/example/voice_characterization_example.ipynb)

Voice emotion classification model training and evaluation on RAVDESS dataset using **vanpy**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/griko/voice_characterizer/blob/main/examples/using_voice_characterizer_to_classify_emotions_on_RAVDESS_dataset.ipynb)

Expending the library with a new classifier

*TODO*

## General description
**vanpy** can be useful in multiple ways. It contains of 3 optional pipelines, that make preprocessing, feature 
extraction and classification of a voice segments an easy task:

1. The preprocessing pipline deals with audio files format and voice segment cutting. 
2. The feature extraction pipline runs
on each segment and retrieves feature/latent vectors. 
3. Classification and Speech-To-Text (SST) pipline executes classification/STT models. 
There are classification components that might be used on the audio files themselves, while others require a feature-set.

You can stop at any point, with respect to the expected achievement. If the voice separation is all what is needed, 
execute the first pipline only. If your task is to generate text from a given audio with a pretrained model that expect 
an audio file as input - use (1) and (3). If you are training a new model and features is of the highest importance - 
go for (1) and (2) and consider expanding the library when you are satisfied with the result (see example 3).  

Configuration of all the components is made through the `pipline.yaml` configuration file.

## Pre-processing components
### Filelist-DataFrame Creator
Used to initialize the `ComponentPayload` with a list of paths to audio files from mapped in the *input_dir* directory in the config file **pipeline.yaml**
### WAVConverter
Converts files to 1-channel, 16KHz, 256k bitrate
### INA Voice Separator
Separates voice and music segments
### PyannoteVAD
Voice Activity Detector by pyannote 2.0 - removes low amplitude sections, leaving intense voice segments. 
### SileroVAD
Voice Activity Detector by silero - removes low amplitude sections, leaving intense voice segments.

## Feature extraction components
### Pyannote Embedding
Uses pyannote-audio 2.0 to extract mean 512-feature embedding vector from the segment
### SpeechBrainEmbedding
Uses SpeechBrain to extract mean 512 (?TODO: recheck)-feature embedding vector from the segment
### LibrosaFeaturesExtractor
Uses librosa to extract MFCC, delta-MFCC and zero-cross-rate from the segment

## Supported classification models:
### Common Voices Gender Classifier
Gender Classification, trained on Mozilla Common Voices Dataset after extracting pyannote embedding.

Predicts 'female'/'male'. Reaches 92.8% accuracy on the test set (!TODO: recheck)

*input*: 512 features of pyannote2.0 embedding
### Common Voices Age Classifier
Age group classification, trained on Mozilla Common Voices Dataset after extracting pyannote embedding.
Predicts 'teens'/'twenties'/'thirties'/'fourties'/'fifties+'. Reaches 34% accuracy on the test set.

Confusion matrix:

![age_cm](https://user-images.githubusercontent.com/1709151/171154228-1ed8927e-37e2-4a6d-ad2d-68f8bb485d1f.PNG)

*input*: 512 features of pyannote2.0 embedding
### IEMOCAP Emotion Classifier
Emotion classification model, trained on IEMOCAP dataset with Speech Brain using fine-tuned wav2vec2.

Predicts 'neu'/'ang'/'hap'/'sad'. Reaches 78.7% accuracy on the test set, as published [here](https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP).

*input*: audio file (doesn't require feature extraction)

## ComponentPayload
Received and passed further between `PiplineComponent`s. Includes:
- features: Dict
  - 'input_path' - the path to the input directory to map audio files from
  - 'paths_column' - rewritable parameter, each preprocessing component at the end of its action writes the column name of the **df** where the files' paths are listed
  - 'feature_columns' - a list of column names of **df** where the features for classifiers are hold
- df: pd.DataFrame
  - includes all the collected information through the preprocessing and classification
    - each preprocessor adds a column of paths where the processed files are hold
    - embedding/feature extraction components add the embedding/features columns
    - each classifier adds a classification column
