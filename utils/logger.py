import utils.config as config

import logging
import os


class Logger:
    def __init__(self, log_file=f"{config.path["log_path"]}/app.log", log_level=logging.INFO):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        if not any(isinstance(handler, logging.FileHandler) for handler in self.logger.handlers):
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)

            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(formatter)

            self.logger.addHandler(file_handler)

    def get_logger(self):
        return self.logger
