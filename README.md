## Example
You can use the following google colab notebook to inspect the capabilities off the library

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/griko/voice_characterizer/blob/main/example/voice_characterization_example.ipynb)


## Pre-processing components
### Filelist-DataFrame Creator
Used to initiate the `ComponentPayload` DataFrame with a list of paths to audio files from mapped in the *input_dir* directory in the config file **pipeline.yaml**
### WAVConverter
Converts files to 1-channel, 16KHz, 256k bitrate
### INA Voice Separator
Separates voice and music segments
## Feature extraction components
### Pyannote Embedding
Uses pyannote-audio 2.0 to extract mean 512-feature embedding vector from the segment

## Supported classification models:
### Common Voices Gender Classifier
Gender Classification, trained on Mozilla Common Voices Dataset after extracting pyannote embedding.

Predicts 'female'/'male'. Reaches 92.8% accuracy on the test set (! recheck)

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
