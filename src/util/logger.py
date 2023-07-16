from dataclasses import dataclass
import logging
import os
import neptune.new as neptune
from typing import Dict
from pathlib import Path
from definitions import PROJECT_VARS

# Define Dataclasses
@dataclass
class LoggerConfig:
    """Dataclass for Logger configuration."""
    log_dir: str = f'{PROJECT_VARS.ROOT_DIR}/logs/'
    log_file_name: str = 'log_all_.log'

@dataclass
class NeptuneConfig:
    """Dataclass for Neptune configuration."""
    project_name: str
    tags: list
    source_files: list

# Define LogConfigurator
class LogConfigurator:
    """Configures the Python logger based on the given configuration."""

    def __init__(self, config: LoggerConfig):
        """Initializes the LogConfigurator with a LoggerConfig."""
        self.config = config

    def configure(self):
        """Configures the Python logger."""
        Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)
        log_file_path = os.path.join(self.config.log_dir, self.config.log_file_name)
        print(f"Logging to {log_file_path}")
        logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(message)s', filemode='w')

# Define NeptuneRunner
class NeptuneRunner:
    """Runs Neptune with the given configuration."""

    def __init__(self, config: NeptuneConfig):
        """Initializes the NeptuneRunner with a NeptuneConfig."""
        self.config = config

    def run(self):
        """Runs Neptune."""
        try:
            nep_run = neptune.init(
                project=self.config.project_name,
                api_token=os.getenv('NEPTUNE_API_TOKEN'),
                tags=self.config.tags,
                source_files=self.config.source_files,
            )
        except Exception as e:
            print(f"Failed to initialize Neptune: {e}")
            nep_run = None
        return nep_run



# Define Logger
class Logger:
    def __init__(self, nep_run, log_configurator, local_log: bool = True, use_neptune: bool = True):
        self.local_log = local_log
        self.use_neptune = use_neptune
        self.nep_run = nep_run
        self.log_configurator = log_configurator

        if self.local_log:
            self.log_configurator.configure()

    def log(self, key, value):
        if self.nep_run and self.use_neptune:
            self.nep_run[key].log(value)

        if self.local_log:
            logging.info(f'{key}: {value}')


# Use LogConfigurator and NeptuneRunner
if __name__ == "__main__":

    os.environ['NEPTUNE_API_TOKEN'] = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2NWQwMGIyZi1mNzM5LTRiMjEtOTg2MC1mNTc4ODRiMWU2ZGYifQ=='
    log_config = LoggerConfig()
    log_configurator = LogConfigurator(log_config)
    log_configurator.configure()

    neptune_config = NeptuneConfig(project_name="ratulalahy/scifact-paraphrase-T5-evo",
                                   tags=['separate_t5_for_majority', 'tech_term_2', 'mlnli'],
                                   source_files=["**/*.ipynb", "*.yaml"])
    neptune_runner = NeptuneRunner(neptune_config)
    nep_run = neptune_runner.run()
    
    logger = Logger(nep_run, log_configurator)
    logger.log("parameters/learning_rate", 0.001)
