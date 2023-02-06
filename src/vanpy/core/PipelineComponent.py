from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict
from yaml import YAMLObject
from logging import Logger
import logging
import pickle
from datetime import datetime
from vanpy.core.ComponentPayload import ComponentPayload
from vanpy.utils.utils import create_dirs_if_not_exist


@dataclass
class PipelineComponent(ABC):
    component_type: str
    component_name: str
    config: Dict
    logger: Logger

    def __init__(self, component_type: str, component_name: str, yaml_config: YAMLObject):
        """
        Initializes the PipelineComponent object with the given component type, component name, and YAML configuration.

        :param component_type: the type of component (e.g. "preprocessing", "feature_extraction", etc.)
        :type component_type: str
        :param component_name: the name of the component (e.g. "pyannote_vad", "speechbrain_embedding", etc.)
        :type component_name: str
        :param yaml_config: the YAML configuration for the component
        :type yaml_config: YAMLObject
        """
        self.component_type = component_type
        self.component_name = component_name
        self.config = self.import_config(yaml_config)
        self.logger = self.get_logger()

    def latent_info_log(self, message: str, iteration: int, last_item: bool = False) -> None:
        """
        Logs the given message if the current iteration is a multiple of the log_each_x_records configuration or if it is the last item in the paths list.

        :param message: the message to log
        :type message: str
        :param iteration: the current iteration
        :type iteration: int
        :param last_item: whether this is the last item in the paths list
        :type last_item: bool
        """
        log_each_x_records = self.config['log_each_x_records'] \
            if 'log_each_x_records' in self.config and self.config['log_each_x_records'] else 10
        last_item = False
        if self.config['records_count']:
            last_item = iteration == self.config['records_count'] - 1
        if iteration % log_each_x_records == 0 or last_item:
            self.logger.info(message)

    def import_config(self, yaml_config: YAMLObject) -> Dict:
        """
        Imports the YAML configuration for the component and returns it as a dictionary.

        :param yaml_config: the YAML configuration for the component
        :type yaml_config: YAMLObject
        :return: the imported configuration as a dictionary
        :rtype: Dict
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
        :rtype: Logger
        """
        return logging.getLogger(f'{self.component_type} - {self.component_name}')

    def get_name(self) -> str:
        """
        Returns the name of the component.

        :return: the name of the component
        :rtype: str
        """
        return self.component_name

    @abstractmethod
    def process(self, input_payload: ComponentPayload) -> ComponentPayload:
        """
        Processes the input payload and returns the output.

        :param input_payload: the input payload to process
        :type input_payload: ComponentPayload
        :return: the output payload after processing
        :rtype: ComponentPayload
        """
        pass

    # @staticmethod
    def save_component_payload(self, input_payload: ComponentPayload, intermediate=False) -> None:
        """
        Saves the input payload to disk, if specified in the configuration.

        :param input_payload: the input payload to save
        :type input_payload: ComponentPayload
        :param intermediate: whether this is an intermediate payload or the final payload
        :type intermediate: bool
        """
        subscript = 'intermediate' if intermediate else 'final'
        self.get_logger().info(
            f'Called Saved payload {self.get_name(), "save_payload" in self.config and self.config["save_payload"]}, intermediate {intermediate}')
        if "save_payload" in self.config and self.config["save_payload"]:
            create_dirs_if_not_exist(self.config["intermediate_payload_path"])
            metadata, df = input_payload.unpack()
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
        :type i: int
        :param input_payload: the payload to be saved
        :type input_payload: ComponentPayload
        """
        if 'save_payload_periodicity' in self.config and i % self.config['save_payload_periodicity'] == 0 and i > 0:
            self.save_component_payload(input_payload, intermediate=True)
