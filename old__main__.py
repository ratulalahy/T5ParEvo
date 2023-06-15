
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from config import SciFactT5Config

conf_store = ConfigStore.instance()
conf_store.store(name="base_config", node=SciFactT5Config)

@hydra.main(config_path="conf", config_name="base_config", version_base=None)
def main(cfg: SciFactT5Config) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
