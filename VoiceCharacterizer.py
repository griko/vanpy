from audio_pipeline.core.ClassificationPipline import ClassificationPipeline
from audio_pipeline.core.FeatureExtractionPipline import FeatureExtractionPipeline
from audio_pipeline.core.PreprocessPipline import PreprocessPipeline
from audio_pipeline.core.CombinedPipeline import CombinedPipeline
import yaml
import logging


def main():
    logging.basicConfig(level=logging.DEBUG)

    with open('pipeline.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    preprocessing_pipeline = PreprocessPipeline(
        ['file_mapper', 'silero_vad'], config=config)  # 'wav_converter', 'pyannote_vad'], config=config)
    feature_extraction_pipeline = FeatureExtractionPipeline(
        ['librosa_features_extractor'], config=config)  # speechbrain_embedding
    speaker_clf_pipeline = ClassificationPipeline(['wav2vec2stt'], config=config)  # speech_brain_iemocap_emotion
    pipline = CombinedPipeline(
        [preprocessing_pipeline, feature_extraction_pipeline, speaker_clf_pipeline], config=config)
    processed_payload = pipline.process()

    print(processed_payload.get_classification_df(all_paths_columns=True, meta_columns=True))
    # speaker_clf_pipeline = SpeakerClassificationPipeline(['common_voices_age', 'common_voices_gender'],
    #                                                      embedding='pyannote')
    # segment_clf_pipeline = SegmentClassificationPipeline(['speaker_id', 'speechbrain_emotion', 'trainscript_wav2vec'],
    #                                                      embedding='pyannote')
    #
    # preprocessed_files_dir, speaker_classification_df, segment_classification_df = Pipeline(pp_pipeline,
    #                                                                                         speaker_clf_pipeline,
    #                                                                                         segment_clf_pipeline)


if __name__ == '__main__':
    main()
