import os

from omegaconf import OmegaConf

from lavis.common.registry import registry

root_dir = os.path.dirname(os.path.abspath(__file__))
default_cfg = OmegaConf.load(os.path.join(root_dir, "configs/default.yaml"))

registry.register_path("library_root", root_dir)
registry.register_path("cache_root", default_cfg.env.cache_root)
