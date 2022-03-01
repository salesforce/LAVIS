import logging
import json

from omegaconf import OmegaConf
from common.registry import registry

logger = logging.getLogger(__name__)


class Config:
    def __init__(self, args): # , default_only=False):
        self.config = {}

        self.args = args

        # Register the config and configuration for setup
        registry.register("configuration", self)

        user_config = self._build_opt_list(self.args.options)


        runner_config = self._build_runner_config(self.args.cfg_run)
        model_config = self._build_model_config(self.args.cfg_model)
        dataset_config = self._build_dataset_config(self.args.cfg_data)
        
        # Override the default configuration with user options.
        self.config = OmegaConf.merge(
           runner_config, model_config, dataset_config, user_config
        )


    def _build_opt_list(self, opts):
        opts_dot_list = self._convert_to_dot_list(opts)
        return OmegaConf.from_dotlist(opts_dot_list)


    def _build_model_config(self, config_path):
        config = OmegaConf.load(config_path)

        root_keys = config.keys()
        assert len(root_keys) == 1, "Missing or duplicate root keys for runner configuration file."
        assert "model" in config, \
            "Root key for model configuration is expected to be 'model', found '{}'.".format(list(root_keys)[0])

        # model = config.model
        # if model is None:
        #     raise KeyError("Required argument 'model' not passed")
        # model_cls = registry.get_model_class(model)

        # if model_cls is None:
        #     warning = f"No model named '{model}' has been registered"
        #     warnings.warn(warning)
        #     return OmegaConf.create()

        # default_model_config_path = model_cls.config_path()

        # if default_model_config_path is None:
        #     warning = "Model {}'s class has no default configuration provided".format(
        #         model
        #     )
        #     warnings.warn(warning)
        #     return OmegaConf.create()

        # return load_yaml(default_model_config_path)

        return config 


    def _build_runner_config(self, config_path):
        config = OmegaConf.load(config_path)

        root_keys = config.keys()
        assert len(root_keys) == 1, "Missing or duplicate root keys for runner configuration file."
        assert "run" in config, \
            "Root key for runner configuration is expected to be 'run', found '{}'.".format(list(root_keys)[0])

        return config


    def _build_dataset_config(self, config_path):
        config = OmegaConf.load(config_path)

        datasets = config.get('datasets', None)
        if datasets is None:
            raise KeyError("Expecting 'datasets' as the root key for dataset configuration.")

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
        logger.info(
            self._convert_node_to_json(self.config.model)
        )

    def _convert_node_to_json(self, node):
        container = OmegaConf.to_container(node, resolve=True)
        return json.dumps(container, indent=4, sort_keys=True)
