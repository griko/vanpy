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
    # # pipeline = CombinedPipeline(
    # #     [preprocessing_pipeline, feature_extraction_pipeline, speaker_clf_pipeline], config=config)
    # pipeline = CombinedPipeline(
    #         [preprocessing_pipeline, speaker_clf_pipeline], config=config)
    # 'sepformer_se', 'metricgan_se', 'speechbrain_embedding', 'vanpy_voxceleb_gender', 'vanpy_voxceleb_age', 'vanpy_voxceleb_height', 'wav2vec2adv'], config=config)

    # pipeline = Pipeline(['file_mapper', 'wav_converter', 'pyannote_sd', 'speechbrain_embedding', 'openai_whisper_stt', 'speech_brain_iemocap_emotion',
    #                     'vanpy_voxceleb_gender', 'vanpy_voxceleb_age', 'vanpy_voxceleb_height', 'vanpy_ravdess_emotion', 'wav2vec2adv', 'yamnet_classifier'], config=config)
    # pipeline = Pipeline(['file_mapper', 'pyannote_sd', 'speechbrain_embedding', 'cosine_distance_diarization'], config=config)  # 'vanpy_speaker_embedding'

    pipeline = Pipeline(
        ['file_mapper', 'pyannote_sd', 'speechbrain_embedding', 'openai_whisper_stt', 'speech_brain_iemocap_emotion',
         'vanpy_voxceleb_gender', 'vanpy_voxceleb_age', 'vanpy_voxceleb_height', 'vanpy_ravdess_emotion', 'wav2vec2adv',
         'yamnet_classifier','cosine_distance_diarization'], config=config)
    # pipeline = Pipeline(['file_mapper', 'speechbrain_embedding', 'vanpy_ravdess_emotion'], config=config)
    # 'wav_converter', 'metricgan_se',, 'silero_vad', 'speechbrain_embedding', 'cosine_distance_diarization' , 'pyannote_sd', 'openai_whisper_stt'
    # openai_whisper_stt, wav2vec2stt
    # processed_payload = await pipeline.process()
    # cp = ComponentPayload(metadata={'paths_column': 'paths'}, df=pd.DataFrame(columns=['paths']))
    # df = pd.DataFrame({'sample_path': ['speech_examples_small/stream_1nwjWQJB_20220104_16_28_02_40.wav']})
    # metadata = {'paths_column': 'sample_path'}
    # cp = ComponentPayload(metadata=metadata, df=df)

    processed_payload = pipeline.process()  # (cp)

    import pandas as pd
    df = processed_payload.df
    df['diarization_classification_spacy_names'] = None
    df['authored_text'] = df.apply(lambda x: str(x['pyannote_diarization_classification']) + "/" + str(
        x['diarization_classification_spacy_names'] if x[
                                                           'diarization_classification_spacy_names'] is not None else '') + ": " + str(
        x['whisper_transcript']), axis=1)
    from vanpy.utils.srt_generator import to_srt

#     params:
#     clustering:
#     method: centroid
#     min_cluster_size: 15
#     threshold: 0.7153814381597874
#
#
# segmentation:
# min_duration_off: 0.5817029604921046
# threshold: 0.4442333667381752
    p_seg_off = config['preprocessing']['pyannote_sd']['hparams']['params']['segmentation']['min_duration_off']
    p_seg_t = config['preprocessing']['pyannote_sd']['hparams']['params']['segmentation']['threshold']
    p_clust_size = config['preprocessing']['pyannote_sd']['hparams']['params']['clustering']['min_cluster_size']
    p_clust_t = config['preprocessing']['pyannote_sd']['hparams']['params']['clustering']['threshold']
    df.to_csv(f'results/final/mkl_local_hparams_pyannote_{p_seg_off=:.2f}_{p_seg_t=:.2f}_{p_clust_size=:.2f}_{p_clust_t=:.2f}.csv', index=False)
    # to_srt(df, 'silero_vad_segment_start', 'silero_vad_segment_stop', 'authored_text')
    with open(f'results/final/subtitles_mkl_local_hparams_pyannote_{p_seg_off=:.2f}_{p_seg_t=:.2f}_{p_clust_size=:.2f}_{p_clust_t=:.2f}.srt', 'w') as f:
        # with open('subtitles_mkl_silero.srt', 'w') as f:
        # f.write(to_srt(df, 'silero_vad_segment_start', 'silero_vad_segment_stop', 'authored_text'))
        f.write(to_srt(df, 'pyannote_sd_segment_start', 'pyannote_sd_segment_stop', 'authored_text'))


    print(processed_payload.get_features_df())
    processed_payload.get_classification_df()
    print(processed_payload.get_classification_df(all_paths_columns=False, meta_columns=False))


if __name__ == '__main__':
    # asyncio.run(main())
    main()
