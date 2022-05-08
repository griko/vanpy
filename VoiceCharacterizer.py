from core.Pipline import Pipeline, PreprocessPipeline, SpeakerClassificationPipeline, SegmentClassificationPipeline
import yaml

with open('pipeline.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

pp_pipeline = PreprocessPipeline(
    ['wav_converter', 'ina_speech_segmenter'], config=config) #, 'pyannote_segmenter', 'pyannote_embedding'], config=config)
pipline = Pipeline(pp_pipeline, None, None, config=config)
preprocessed_files_dir, speaker_classification_df, segment_classification_df = pipline.process()
# speaker_clf_pipeline = SpeakerClassificationPipeline(['common_voices_age', 'common_voices_gender'],
#                                                      embedding='pyannote')
# segment_clf_pipeline = SegmentClassificationPipeline(['speaker_id', 'speechbrain_emotion', 'trainscript_wav2vec'],
#                                                      embedding='pyannote')
#
# preprocessed_files_dir, speaker_classification_df, segment_classification_df = Pipeline(pp_pipeline,
#                                                                                         speaker_clf_pipeline,
#                                                                                         segment_clf_pipeline)
