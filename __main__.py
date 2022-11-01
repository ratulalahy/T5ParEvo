import hydra
from hydra.core.config_store import ConfigStore

from config import SciFactT5Config

conf_store = ConfigStore.instance()
conf_store.store(name="scifact_t5_config", node=SciFactT5Config)


@hydra.main(config_path="conf", config_name="config")
def main(cfg: SciFactT5Config) -> None:
    print(cfg.params)
