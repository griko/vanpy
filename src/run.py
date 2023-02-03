from vanpy.core.FeatureExtractionPipeline import FeatureExtractionPipeline
from vanpy.core.PreprocessPipeline import PreprocessPipeline
from vanpy.core.CombinedPipeline import CombinedPipeline
import logging
from vanpy.utils.utils import load_config


# import asyncio


# async def main():
def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s:%(name)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    config = load_config('pipeline.yaml')

    preprocessing_pipeline = PreprocessPipeline(
        ['file_mapper', 'wav_converter', 'silero_vad', 'pyannote_vad'], config=config)  #'file_mapper', 'silero_vad', 'pyannote_vad'], config=config)  # 'wav_converter', 'espnet-se', 'silero_vad', 'pyannote_vad'], config=config)
    feature_extraction_pipeline = FeatureExtractionPipeline(
        ['librosa_features_extractor', 'speechbrain_embedding', 'pyannote_embedding'], config=config)  # 'librosa_features_extractor', 'speechbrain_embedding', 'pyannote_embedding'
    # speaker_clf_pipeline = ClassificationPipeline(['wav2vec2stt'], config=config)  # speech_brain_iemocap_emotion
    # pipline = CombinedPipeline(
    #     [preprocessing_pipeline, feature_extraction_pipeline, speaker_clf_pipeline], config=config)
    pipline = CombinedPipeline(
            [preprocessing_pipeline, feature_extraction_pipeline], config=config)
    # processed_payload = await pipline.process()
    # cp = ComponentPayload(metadata={'paths_column': 'paths'}, df=pd.DataFrame(columns=['paths']))
    # df = pd.DataFrame({'sample_path': ['speech_examples_small/stream_1nwjWQJB_20220104_16_28_02_40.wav']})
    # metadata = {'paths_column': 'sample_path'}
    # cp = ComponentPayload(metadata=metadata, df=df)
    processed_payload = pipline.process()  # (cp)

    print(processed_payload.get_classification_df(all_paths_columns=True, meta_columns=True))


if __name__ == '__main__':
    # asyncio.run(main())
    main()
