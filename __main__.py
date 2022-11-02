from dataclasses import dataclass

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from config import SciFactT5Config

conf_store = ConfigStore.instance()
conf_store.store(name="scifact_t5_config", node=SciFactT5Config)

# cs = ConfigStore.instance()
# cs.store(name="mnist_config", node=MNISTConfig)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: SciFactT5Config) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
