## ComponentPayload
Passed from received and passed further between *PiplineComponent*s. Includes:
- features: Dict
  - 'input_path' - the path to the input directory to map audio files from
  - 'paths_column' - rewritable parameter, each preprocessing component at the end of it's action writes the column name of the **df** where the files' paths are listed
  - 'feature_columns' - a list of column names of **df** where the features for classifiers are hold
- df: pd.DataFrame
  - includes all the collected information through the preprocessing and classification
    - each preprocessor adds a column of paths where the processed files are hold
    - embedding/feature extraction components adds the embedding/features columns
    - each classifier adds classification column