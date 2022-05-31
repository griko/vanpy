from core.ClassificationPipline import ClassificationPipeline
from core.FeatureExtractionPipline import FeatureExtractionPipeline
from core.PreprocessPipline import PreprocessPipeline
from core.CombinedPipeline import CombinedPipeline
import yaml
import logging


def main():
    logging.basicConfig(level=logging.DEBUG)

    with open('pipeline.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    preprocessing_pipeline = PreprocessPipeline(
        ['file_mapper', 'wav_converter', 'ina_speech_segmenter'], config=config)
    feature_extraction_pipeline = FeatureExtractionPipeline(
        ['pyannote_embedding'], config=config)
    speaker_clf_pipeline = ClassificationPipeline(
        ['common_voices_gender', 'common_voices_age', 'speech_brain_iemocap_emotion'], config=config)
    pipline = CombinedPipeline(
        [preprocessing_pipeline, feature_extraction_pipeline, speaker_clf_pipeline], config=config)
    processed_payload = pipline.process()

    print(processed_payload.get_classification_df())
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
