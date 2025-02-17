from vanpy.core.Pipeline import Pipeline
from vanpy.utils.utils import load_config
import logging

def main():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s:%(name)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    config = load_config('pipeline.yaml')
    

    # pipeline = Pipeline(['file_mapper', 'pyannote_sd', 'speechbrain_embedding', 'openai_whisper_stt'], config=config)
    # pipeline = Pipeline(['file_mapper', 'pyannote_vad', 'speechbrain_embedding', 'openai_whisper_stt', 'speech_brain_iemocap_emotion',
    #                     'wav2vec2adv', 'yamnet_classifier', 'cosine_distance_diarization', 'agglomerative_clustering_diarization', 'gmm_clustering_diarization'], config=config)
    
    pipeline = Pipeline(['file_mapper', 'pyannote_sd', 'librosa_features_extractor', 'speechbrain_embedding', 'vanpy_gender', 'vanpy_age', 'vanpy_height', 'vanpy_emotion'], config=config)

    processed_payload = pipeline.process()
    classification_df = processed_payload.get_classification_df()
    print(classification_df)


if __name__ == '__main__':
    main()
