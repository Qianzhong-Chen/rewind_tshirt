import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

@hydra.main(config_path="config", config_name=None)
def main(cfg: DictConfig):
    # print(OmegaConf.to_yaml(cfg))
    workspace = instantiate(cfg)
    workspace.train()

if __name__ == "__main__":
    main()
