import os
import pandas as pd
from pydub import AudioSegment
from tqdm.auto import tqdm
from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.core.preprocess_components.BaseSegmenterComponent import BaseSegmenterComponent
from vanpy.utils.utils import create_dirs_if_not_exist
from yaml import YAMLObject

class WAVSplitter(BaseSegmenterComponent):
    """
    Audio splitting component for handling large files.

    Splits WAV files based on either maximum duration or file size while
    maintaining audio quality.

    :ivar max_audio_length: Maximum length of each segment in seconds.
    :ivar max_wav_file_size: Maximum size of each segment in bytes.
    """
    def __init__(self, yaml_config: YAMLObject):
        """
        Initializes the WAVSplitter class and creates initial splitting configuration parameters
        :param yaml_config: A YAMLObject containing the configuration for the pipeline
        """
        super().__init__(component_type='preprocessing', component_name='wav_splitter',
                         yaml_config=yaml_config)
        self.max_audio_length = self.config.get('max_audio_length', None)  # in seconds
        self.max_wav_file_size = self.config.get('max_wav_file_size', None)  # in bytes

    def process(self, input_payload: ComponentPayload) -> ComponentPayload:
        """
        Split WAV audio files by maximum length or file size.

        :param input_payload: A ComponentPayload object containing metadata and a DataFrame with audio file paths.
        :return: A ComponentPayload object containing metadata and a DataFrame with the split audio file paths.
        """
        metadata, df = input_payload.unpack()
        input_column = metadata['paths_column']
        if not self.max_audio_length and not self.max_wav_file_size:
            raise KeyError("WAV splitter cannot run without specifying either 'max_audio_length' or 'max_wav_file_size' in the configuration.")
        paths_list = df[input_column].dropna().tolist()
        output_dir = self.config['output_dir']
        create_dirs_if_not_exist(output_dir)

        processed_path = self.get_processed_path()
        p_df, paths_list = self.get_file_paths_and_processed_df_if_not_overwriting(paths_list, processed_path,
                                                                                   input_column, output_dir,
                                                                                   use_dir_prefix=self.config.get('use_dir_name_as_prefix', False))
        if not paths_list:
            self.logger.warning('You\'ve supplied an empty list to process')
            df = pd.merge(left=df, right=p_df, how='left', on=input_column)
            return ComponentPayload(metadata=metadata, df=df)

        for j, f in enumerate(tqdm(paths_list)):
            try:
                audio = AudioSegment.from_file(f)
                total_duration = len(audio) / 1000.0  # duration in seconds

                # Determine the appropriate splitting method
                if self.max_audio_length and total_duration > self.max_audio_length:
                    self.split_audio_by_length(audio, f, output_dir, p_df, input_column, j)
                elif self.max_wav_file_size and os.path.getsize(f) > self.max_wav_file_size:
                    self.split_audio_by_size(audio, f, output_dir, p_df, input_column, j)
                else:
                    # No splitting needed, just copy the original file path
                    f_df = pd.DataFrame.from_dict({processed_path: [f], input_column: [f]})
                    p_df = pd.concat([p_df, f_df], ignore_index=True)
            except Exception as e:
                self.logger.error(f'Error processing {f}: {e}')
                continue

        df = pd.merge(left=df, right=p_df, how='outer', on=input_column)
        metadata = self.enhance_metadata(metadata)
        return ComponentPayload(metadata=metadata, df=df)

    def split_audio_by_length(self, audio, original_path, output_dir, p_df, input_column, file_index):
        """
        Split audio file based on maximum duration.

        :param audio: AudioSegment object to split.
        :param original_path: Path to original audio file.
        :param output_dir: Directory to save split segments.
        :param p_df: DataFrame to store processing results.
        :param input_column: Column name for input file paths.
        :param file_index: Index for progress tracking.
        """
        filename = os.path.splitext(os.path.basename(original_path))[0]
        dir_prefix = os.path.basename(os.path.dirname(original_path)) + '_' if self.config.get('use_dir_name_as_prefix',
                                                                                               False) else ''

        num_segments = int(len(audio) / 1000.0 / self.max_audio_length)
        remaining_duration = len(audio) / 1000.0 - num_segments * self.max_audio_length  # Calculate remaining time

        for i in range(num_segments):
            start_time = i * self.max_audio_length * 1000  # in milliseconds
            end_time = (i + 1) * self.max_audio_length * 1000
            segment = audio[start_time:end_time]
            segment_filename = f'{dir_prefix}{filename}_part{i + 1}.wav'
            segment_filepath = os.path.join(output_dir, segment_filename)
            segment.export(segment_filepath, format='wav')

            f_df = pd.DataFrame.from_dict({self.get_processed_path(): [segment_filepath],
                                           input_column: [original_path]})
            p_df = pd.concat([p_df, f_df], ignore_index=True)
            self.latent_info_log(f'Split {original_path} into {segment_filename}', iteration=file_index)

        # Handle the remaining segment if it exists
        if remaining_duration > 0:
            start_time = num_segments * self.max_audio_length * 1000  # in milliseconds
            segment = audio[start_time:]
            segment_filename = f'{dir_prefix}{filename}_part{num_segments + 1}.wav'
            segment_filepath = os.path.join(output_dir, segment_filename)
            segment.export(segment_filepath, format='wav')

            f_df = pd.DataFrame.from_dict({self.get_processed_path(): [segment_filepath],
                                           input_column: [original_path]})
            p_df = pd.concat([p_df, f_df], ignore_index=True)
            self.latent_info_log(f'Split {original_path} into {segment_filename}', iteration=file_index)

    def split_audio_by_size(self, audio, original_path, output_dir, p_df, input_column, file_index):
        """
        Split audio file based on maximum file size.

        :param audio: AudioSegment object to split.
        :param original_path: Path to original audio file.
        :param output_dir: Directory to save split segments.
        :param p_df: DataFrame to store processing results.
        :param input_column: Column name for input file paths.
        :param file_index: Index for progress tracking.
        """
        filename = os.path.splitext(os.path.basename(original_path))[0]
        dir_prefix = os.path.basename(os.path.dirname(original_path)) + '_' if self.config.get('use_dir_name_as_prefix',
                                                                                               False) else ''

        # Convert max_wav_file_size from MB to bytes
        max_wav_file_size_bytes = self.max_wav_file_size * 1024 * 1024  # Convert MB to bytes

        # Estimate bytes per second based on the entire file
        export_temp_path = os.path.join(output_dir, f'{filename}_temp.wav')
        audio.export(export_temp_path, format='wav')
        bytes_per_second = os.path.getsize(export_temp_path) / len(audio) * 1000.0  # Calculate bytes per second
        os.remove(export_temp_path)

        segment_duration = max_wav_file_size_bytes / bytes_per_second  # Duration for each segment

        num_segments = int(len(audio) / 1000.0 / segment_duration)
        remaining_duration = len(audio) / 1000.0 - num_segments * segment_duration  # Calculate remaining time

        for i in range(num_segments):
            start_time = i * segment_duration * 1000  # in milliseconds
            end_time = (i + 1) * segment_duration * 1000
            segment = audio[start_time:end_time]
            segment_filename = f'{dir_prefix}{filename}_part{i + 1}.wav'
            segment_filepath = os.path.join(output_dir, segment_filename)
            segment.export(segment_filepath, format='wav')

            f_df = pd.DataFrame.from_dict({self.get_processed_path(): [segment_filepath],
                                           input_column: [original_path]})
            p_df = pd.concat([p_df, f_df], ignore_index=True)
            self.latent_info_log(f'Split {original_path} into {segment_filename}', iteration=file_index)

        # Handle the remaining segment if it exists
        if remaining_duration > 0:
            start_time = num_segments * segment_duration * 1000  # in milliseconds
            segment = audio[start_time:]
            segment_filename = f'{dir_prefix}{filename}_part{num_segments + 1}.wav'
            segment_filepath = os.path.join(output_dir, segment_filename)
            segment.export(segment_filepath, format='wav')

            f_df = pd.DataFrame.from_dict({self.get_processed_path(): [segment_filepath],
                                           input_column: [original_path]})
            p_df = pd.concat([p_df, f_df], ignore_index=True)
            self.latent_info_log(f'Split {original_path} into {segment_filename}', iteration=file_index)
