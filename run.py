from vanpy.core.ClassificationPipline import ClassificationPipeline
from vanpy.core.FeatureExtractionPipline import FeatureExtractionPipeline
from vanpy.core.PreprocessPipline import PreprocessPipeline
from vanpy.core.CombinedPipeline import CombinedPipeline
import yaml
import logging
from vanpy.utils.utils import yaml_placeholder_replacement
# import asyncio


# async def main():
def main():
    logging.basicConfig(level=logging.DEBUG)

    with open('pipeline.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = yaml_placeholder_replacement(config)

    preprocessing_pipeline = PreprocessPipeline(
        ['file_mapper', 'wav_converter'], config=config)  #, 'silero_vad', 'pyannote_vad'], config=config)  # 'wav_converter', 'espnet-se', 'silero_vad', 'pyannote_vad'], config=config)
    feature_extraction_pipeline = FeatureExtractionPipeline(
        ['librosa_features_extractor', 'speechbrain_embedding', 'pyannote_embedding'], config=config)  # 'librosa_features_extractor', 'speechbrain_embedding', 'pyannote_embedding'
    # speaker_clf_pipeline = ClassificationPipeline(['wav2vec2stt'], config=config)  # speech_brain_iemocap_emotion
    # pipline = CombinedPipeline(
    #     [preprocessing_pipeline, feature_extraction_pipeline, speaker_clf_pipeline], config=config)
    pipline = CombinedPipeline(
            [preprocessing_pipeline, feature_extraction_pipeline], config=config)
    # processed_payload = await pipline.process()
    processed_payload = pipline.process()

    print(processed_payload.get_classification_df(all_paths_columns=True, meta_columns=True))


if __name__ == '__main__':
    # asyncio.run(main())
    main()