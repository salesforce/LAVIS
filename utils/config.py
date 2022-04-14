import logging
import json
import warnings

from omegaconf import OmegaConf
from common.registry import registry

logger = logging.getLogger(__name__)


class Config:
    def __init__(self, args):  # , default_only=False):
        self.config = {}

        self.args = args

        # Register the config and configuration for setup
        registry.register("configuration", self)

        user_config = self._build_opt_list(self.args.options)

        config = OmegaConf.load(self.args.cfg_path)

        runner_config = self.build_runner_config(config)
        model_config = self.build_model_config(config, **user_config)
        dataset_config = self.build_dataset_config(config)

        # Override the default configuration with user options.
        self.config = OmegaConf.merge(
            runner_config, model_config, dataset_config, user_config
        )

    def _build_opt_list(self, opts):
        opts_dot_list = self._convert_to_dot_list(opts)
        return OmegaConf.from_dotlist(opts_dot_list)

    @staticmethod
    def build_model_config(config, **kwargs):
        model = config.get("model", None)
        assert model is not None, "Missing model configuration file."

        model_cls = registry.get_model_class(model.arch)
        assert (
            model_cls is not None
        ), f"No model named '{model.arch}' has been registered"

        model_type = kwargs.get("model.model_type", None)
        if not model_type:
            model_type = model.get("model_type", None)
        # else use the model type selected by user.

        if model_type:
            default_model_config_path = model_cls.default_config_path(
                model_type=model_type
            )
        else:
            default_model_config_path = model_cls.default_config_path()

        model_config = OmegaConf.create()
        # hiararchy override, customized config > default config
        model_config = OmegaConf.merge(
            model_config, OmegaConf.load(default_model_config_path), config
        )

        return model_config

    @staticmethod
    def build_runner_config(config):
        return config.run

    @staticmethod
    def build_dataset_config(config):
        datasets = config.get("datasets", None)
        if datasets is None:
            raise KeyError(
                "Expecting 'datasets' as the root key for dataset configuration."
            )

        dataset_config = OmegaConf.create()

        for dataset in datasets:
            builder_cls = registry.get_builder_class(dataset)
            default_dataset_config_path = builder_cls.default_config_path()

            # hiararchy override, customized config > default config
            dataset_config = OmegaConf.merge(
                dataset_config, OmegaConf.load(default_dataset_config_path), config
            )

        return dataset_config

    def _convert_to_dot_list(self, opts):
        if opts is None:
            opts = []

        if len(opts) == 0:
            return opts

        has_equal = opts[0].find("=") != -1

        if has_equal:
            return opts

        return [(opt + "=" + value) for opt, value in zip(opts[0::2], opts[1::2])]

    def get_config(self):
        return self.config

    @property
    def run_cfg(self):
        return self.config.run

    @property
    def datasets_cfg(self):
        return self.config.datasets

    @property
    def model_cfg(self):
        return self.config.model

    def pretty_print(self):
        logger.info("\n=====  Running Parameters    =====")
        logger.info(self._convert_node_to_json(self.config.run))

        logger.info("\n======  Dataset Attributes  ======")
        datasets = self.config.datasets

        for dataset in datasets:
            if dataset in self.config.datasets:
                logger.info(f"\n======== {dataset} =======")
                dataset_config = self.config.datasets[dataset]
                logger.info(self._convert_node_to_json(dataset_config))
            else:
                logger.warning(f"No dataset named '{dataset}' in config. Skipping")

        logger.info(f"\n======  Model Attributes  ======")
        logger.info(self._convert_node_to_json(self.config.model))

    def _convert_node_to_json(self, node):
        container = OmegaConf.to_container(node, resolve=True)
        return json.dumps(container, indent=4, sort_keys=True)
