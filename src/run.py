from vanpy.core.ClassificationPipeline import ClassificationPipeline
from vanpy.core.FeatureExtractionPipeline import FeatureExtractionPipeline
from vanpy.core.PreprocessPipeline import PreprocessPipeline
from vanpy.core.Pipeline import Pipeline
import logging
from vanpy.utils.utils import load_config
# import asyncio


# async def main():
def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s:%(name)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    config = load_config('pipeline.yaml')

    # preprocessing_pipeline = PreprocessPipeline(
    #     ['file_mapper'], config=config)  #, 'wav_converter', 'silero_vad', 'pyannote_vad'], config=config)  # 'wav_converter', 'espnet-se', 'silero_vad', 'pyannote_vad', 'pyannote_sd'], config=config)
    # # feature_extraction_pipeline = FeatureExtractionPipeline(
    # #     ['librosa_features_extractor'], config=config)  # 'librosa_features_extractor', 'speechbrain_embedding', 'pyannote_embedding', 'speechbrain_embedding', 'vanpy_ravdess_emotion'
    # speaker_clf_pipeline = ClassificationPipeline(['openai_whisper_stt', 'wav2vec2stt'], config=config)  # speech_brain_iemocap_emotion
    # # pipline = CombinedPipeline(
    # #     [preprocessing_pipeline, feature_extraction_pipeline, speaker_clf_pipeline], config=config)
    # pipline = CombinedPipeline(
    #         [preprocessing_pipeline, speaker_clf_pipeline], config=config)
    # 'sepformer_se', 'metricgan_se', 'speechbrain_embedding', 'vanpy_voxceleb_gender', 'vanpy_voxceleb_age', 'vanpy_voxceleb_height', 'wav2vec2adv'], config=config)
    pipline = Pipeline(['file_mapper', 'wav_converter', 'pyannote_sd', 'speechbrain_embedding', 'openai_whisper_stt', 'speech_brain_iemocap_emotion',
                        'vanpy_voxceleb_gender'', ''vanpy_voxceleb_age', 'vanpy_voxceleb_height', 'vanpy_ravdess_emotion', 'wav2vec2adv', 'yamnet_classifier'], config=config)
    # 'wav_converter', 'metricgan_se',, 'silero_vad', 'speechbrain_embedding', 'cosine_distance_diarization' , 'pyannote_sd', 'openai_whisper_stt'
    # openai_whisper_stt, wav2vec2stt
    # processed_payload = await pipline.process()
    # cp = ComponentPayload(metadata={'paths_column': 'paths'}, df=pd.DataFrame(columns=['paths']))
    # df = pd.DataFrame({'sample_path': ['speech_examples_small/stream_1nwjWQJB_20220104_16_28_02_40.wav']})
    # metadata = {'paths_column': 'sample_path'}
    # cp = ComponentPayload(metadata=metadata, df=df)
    processed_payload = pipline.process()  # (cp)
    print(processed_payload.get_classification_df(all_paths_columns=False, meta_columns=False))


if __name__ == '__main__':
    # asyncio.run(main())
    main()
