from concurrent.futures import ThreadPoolExecutor, as_completed
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Union, List
# import os

import pandas as pd
from yaml import YAMLObject
from logging import Logger
import logging
import pickle
from datetime import datetime
import time
from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.utils.utils import create_dirs_if_not_exist
from tqdm.auto import tqdm
from queue import Queue

@dataclass
class PipelineComponent(ABC):
    """
    The base class for all pipeline components.

    :param component_type: the type of component (e.g. "preprocessing", "feature_extraction", etc.)
    :param component_name: the name of the component (e.g. "pyannote_vad", "speechbrain_embedding", etc.)
    """
    component_type: str
    component_name: str
    config: Dict
    logger: Logger
    pretrained_models_dir: str

    def __init__(self, component_type: str, component_name: str, yaml_config: YAMLObject):
        """
        Initializes the PipelineComponent object with the given component type, component name, and YAML configuration.

        :param component_type: the type of component (e.g. "preprocessing", "feature_extraction", etc.)
        :param component_name: the name of the component (e.g. "pyannote_vad", "speechbrain_embedding", etc.)
        :param yaml_config: the YAML configuration for the component
        """
        self.component_type = component_type
        self.component_name = component_name
        self.config = self.import_config(yaml_config)
        self.logger = self.get_logger()
        self.latent_logger_enabled = self.config.get('latent_logger', False) and self.config['latent_logger'].get(
            'enabled', False)
        self.log_each_x_records = self.config.get('log_each_x_records', self.config['latent_logger'].get(
            'log_each_x_records', 10) if self.config.get('latent_logger', False) else 10)
        self.pretrained_models_dir = self.config.get('pretrained_models_dir',
                                                     f'pretrained_models/{self.component_name}')
        self.performance_measurement = self.config.get('performance_measurement', False)
        self.file_performance_column_name = self.config.get('file_performance_column_name',
                                                            f'perf_{self.get_name()}_get_features')
        self.max_workers = self.config.get('max_workers', 4)

    def latent_info_log(self, message: str, iteration: int, last_item: bool = False) -> None:
        """
        Logs the given message if the current iteration is a multiple of the log_each_x_records configuration or if it is the last item in the paths list.

        :param message: the message to log
        :param iteration: the current iteration
        :param last_item: whether this is the last item in the paths list
        """
        if iteration % self.log_each_x_records == 0 or last_item:
            self.logger.info(message)

    def import_config(self, yaml_config: YAMLObject) -> Dict:
        """
        Imports the YAML configuration for the component and returns it as a dictionary.

        :param yaml_config: the YAML configuration for the component
        :return: the imported configuration as a dictionary
        """
        if self.component_type in yaml_config and self.component_name in yaml_config[self.component_type]:
            config = yaml_config[self.component_type][self.component_name]
        else:
            config = {}
        for item in yaml_config:  # pass through all root level configs
            if isinstance(item, str) and item not in config:
                config[item] = yaml_config[item]
        return config

    def get_logger(self) -> Logger:
        """
        Returns the logger for the component.

        :return: the logger for the component
        """
        return logging.getLogger(f'{self.component_type} - {self.component_name}')

    def get_name(self) -> str:
        """
        Returns the name of the component.

        :return: the name of the component
        """
        return self.component_name

    @abstractmethod
    def process(self, input_payload: ComponentPayload) -> ComponentPayload:
        """
        Processes the input payload and returns the output.

        :param input_payload: the input payload to process
        :type input_payload: ComponentPayload
        :return: the output payload after processing
        """
        raise NotImplementedError

    def process_item(self, *args, **kwargs):
        """
        Processes a single item from the input payload.
        To be used in process_with_progress.
        """
        raise NotImplementedError

    def wrapper_process_item(self, q: Queue, *args, **kwargs):
        """
        Wrapper function to process a single item from the input payload.
        To be used in process_with_progress.
        """
        start_time = time.time()
        result = self.process_item(*args, **kwargs)
        end_time = time.time()
        q.put(end_time - start_time)
        return result

    def process_with_progress(self, iterable, metadata,  *args, **kwargs) -> pd.DataFrame:
        """
        Process items in parallel with progress tracking.

        Handles parallel processing of items using ThreadPoolExecutor, with
        progress bar and performance monitoring.

        :param iterable: Items to process.
        :param metadata: Metadata for processing.
        :param args: Additional positional arguments for processing.
        :param kwargs: Additional keyword arguments for processing.
        :return: DataFrame containing processed results.
        """
        self.logger.debug(f"Executing process_with_progress using {self.max_workers} thread workers")
        n_df = pd.DataFrame()
        time_queue = Queue()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.wrapper_process_item, time_queue, elem, *args, **kwargs): elem for elem in
                       iterable}

            for i, future in enumerate(tqdm(as_completed(futures), total=len(futures))):
                elem = futures[future]
                try:
                    f_df = future.result()
                    time_taken = time_queue.get()
                    self.add_performance_metadata(f_df, time_taken)
                    n_df = pd.concat([n_df, f_df], ignore_index=True, axis=0)
                    if self.latent_logger_enabled:
                        self.latent_info_log(
                            f'{self.component_name} processed {elem}, {i + 1}/{len(iterable)} in {time_taken} seconds',
                            iteration=i, last_item=(i == len(iterable) - 1))
                    self.save_intermediate_payload(i, ComponentPayload(metadata=metadata, df=n_df))
                except (RuntimeError, AssertionError, ValueError, TypeError) as e:
                    self.logger.error(f'An error occurred in {elem}: {e}')

        return n_df

    # @staticmethod
    def save_component_payload(self, input_payload: ComponentPayload, intermediate=False) -> None:
        """
        Saves the input payload to disk, if specified in the configuration.

        :param input_payload: the input payload to save
        :param intermediate: whether this is an intermediate payload or the final payload
        """
        subscript = 'intermediate' if intermediate else 'final'
        self.get_logger().info(
            f'Called Saved payload {self.get_name(), self.config.get("save_payload", False)}, intermediate {intermediate}')
        if self.config.get("save_payload", False):
            create_dirs_if_not_exist(self.config["intermediate_payload_path"])
            metadata, df = input_payload.unpack()
            if metadata:
                with open(
                        f'{self.config["intermediate_payload_path"]}/{self.component_type}_{self.component_name}_metadata_{datetime.now().strftime("%Y%m%d%H%M%S")}_{subscript}.pickle',
                        'wb') as handle:
                    pickle.dump(metadata, handle, protocol=pickle.HIGHEST_PROTOCOL)
            # input_payload.get_classification_df(all_paths_columns=True, meta_columns=True).to_csv(f'{self.config["intermediate_payload_path"]}/{self.component_type}_{self.component_name}_clf_df_{datetime.now().strftime("%Y%m%d%H%M%S")}_{subscript}.csv')
            df.to_csv(
                f'{self.config["intermediate_payload_path"]}/{self.component_type}_{self.component_name}_df_{datetime.now().strftime("%Y%m%d%H%M%S")}_{subscript}.csv',
                index=False)
            self.get_logger().info(f'Saved payload in {self.config["intermediate_payload_path"]}')

    def save_intermediate_payload(self, i: int, input_payload: ComponentPayload):
        """
        Save intermediate payload based on the save_payload_periodicity configuration.

        :param i: current iteration count
        :param input_payload: the payload to be saved
        """
        if 'save_payload_periodicity' in self.config and i % self.config['save_payload_periodicity'] == 0 and i > 0:
            self.save_component_payload(input_payload, intermediate=True)

    def add_performance_column_to_metadata(self, metadata: Dict) -> Dict:
        """
        Adds a performance column to the metadata.

        :param metadata: the metadata to add the performance column to
        :return: the metadata with the performance column added
        """
        if self.config.get('performance_measurement', True):
            self.file_performance_column_name = f'perf_{self.get_name()}'
            metadata['meta_columns'].extend([self.file_performance_column_name])
        return metadata

    def add_classification_columns_to_metadata(self, metadata: Dict, cols: Union[List[str], str]) -> Dict:
        """
        Adds classification columns to the metadata.

        :param metadata: the metadata to add the classification columns to
        :param cols: the classification columns to add
        :return: the metadata with the classification columns added
        """
        if cols:
            if isinstance(cols, str):
                cols = [cols]
            metadata['classification_columns'].extend(cols)
        elif self.config.get('classification_column_name', True):
            self.classification_column_name = f'clf_{self.get_name()}'
            cols = [self.classification_column_name]
            metadata['classification_columns'].app(cols)
        else:
            self.logger.warning(f'No classification columns were added to metadata for {self.get_name()}')
        return metadata

    def add_performance_metadata(self, f_d: Union[pd.DataFrame, Dict], t_start: float, t_end: Union[None, float]=None):
        """
        Adds performance metadata for the audio segments.

        :param f_d: The temporal DataFrame that is enhanced with the performance time of audio segments.
        :param t_start: The start time of the audio segment extraction. If passed without t_end, it is assumed that the performance time is the time it took to process the audio segment.
        :param t_end: The end time of the audio segment extraction. If None, it is assumed that the performance time is the time it took to process the audio segment is passed through t_start.
        """
        if self.performance_measurement:
            if t_end:
                f_d[self.file_performance_column_name] = t_end - t_start
            else:
                f_d[self.file_performance_column_name] = t_start