
## Pre-processing components
### FilelistDataFrameCreator
Used to initiate the `ComponentPayload` DataFrame with a list of paths to audio files from mapped in the *input_dir* directory in the config file **pipeline.yaml**
### WAVConverter
Converts files to 1-channel, 16KHz, 256k bitrate
### INAVoiceSeparator
Separates voice and music segments
## Feature extraction components
### PyannoteEmbedding
Uses pyannote-audio 2.0 to extract mean 512-feature embedding vector from the segment

## Supported classification models:
### CVGenderClassifier
Predicts 'female'/'male', trained on Mozilla Common Voices Dataset after extracting pyannote embedding. 
Reaches 92.8% accuracy on a train set (! recheck)
### CVAgeClassifier
Predicts 'teens'/'twenties'/'thirties'/'fourties'/'fifties+', trained on Mozilla Common Voices Dataset after extracting pyannote embedding. 
Reaches TBD% accuracy on a train set


## ComponentPayload
Received and passed further between `PiplineComponent`s. Includes:
- features: Dict
  - 'input_path' - the path to the input directory to map audio files from
  - 'paths_column' - rewritable parameter, each preprocessing component at the end of it's action writes the column name of the **df** where the files' paths are listed
  - 'feature_columns' - a list of column names of **df** where the features for classifiers are hold
- df: pd.DataFrame
  - includes all the collected information through the preprocessing and classification
    - each preprocessor adds a column of paths where the processed files are hold
    - embedding/feature extraction components adds the embedding/features columns
    - each classifier adds classification column