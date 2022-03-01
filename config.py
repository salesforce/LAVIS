import argparse
from omegaconf import OmegaConf


class OliveConfig:
    def __init__(self, args): # , default_only=False):
        self.config = {}

        self.args = args
        # self._register_resolvers()

        # self._default_config = self._build_default_config()

        # Initially, silently add opts so that some of the overrides for the defaults
        # from command line required for setup can be honored
        # self._default_config = _merge_with_dotlist(
        #     self._default_config, args.opts, skip_missing=True, log_info=False
        # )

        # Register the config and configuration for setup
        # registry.register("config", self._default_config)
        # registry.register("configuration", self)

        # if default_only:
        #     other_configs = {}
        # else:

        user_config = self._build_opt_list(self.args.options)
        # user_config = self._build_user_config(opts_config)
        # self._user_config = user_config

        # self._user_config = user_config

        # self.import_user_dir()

        runner_config = self._build_runner_config(self.args.cfg_run)
        model_config = self._build_model_config(self.args.cfg_model)
        dataset_config = self._build_dataset_config(self.args.cfg_data)

        # args_overrides = self._build_demjson_config(self.args.config_override)
        # other_configs = OmegaConf.merge(
        #     model_config, dataset_config, user_config # , args_overrides
        # )
        self.configs = OmegaConf.merge(
           runner_config, model_config, dataset_config, user_config
        )

        # self.config = OmegaConf.merge(self._default_config, other_configs)

        # self.config = _merge_with_dotlist(self.config, args.opts)
        # self._update_specific(self.config)
        # self.upgrade(self.config)
        # # Resolve the config here itself after full creation so that spawned workers
        # # don't face any issues
        # self.config = OmegaConf.create(
        #     OmegaConf.to_container(self.config, resolve=True)
        # )

        # # Update the registry with final config
        # registry.register("config", self.config)

    # def _build_default_config(self):
    #     self.default_config_path = get_default_config_path()
    #     default_config = load_yaml(self.default_config_path)
    #     return default_config

    # def _build_other_configs(self):
    #     opts_config = self._build_opt_list(self.args.opts)
    #     user_config = self._build_user_config(opts_config)

    #     self._opts_config = opts_config
    #     self._user_config = user_config

    #     self.import_user_dir()

    #     model_config = self._build_model_config(opts_config)
    #     dataset_config = self._build_dataset_config(opts_config)
    #     args_overrides = self._build_demjson_config(self.args.config_override)
    #     other_configs = OmegaConf.merge(
    #         model_config, dataset_config, user_config, args_overrides
    #     )

    #     return other_configs

    def _build_opt_list(self, opts):
        opts_dot_list = self._convert_to_dot_list(opts)
        return OmegaConf.from_dotlist(opts_dot_list)

    # def _build_user_config(self, opts):
    #     user_config = {}

    #     # Update user_config with opts if passed
    #     self.config_path = opts.config
    #     if self.config_path is not None:
    #         user_config = load_yaml(self.config_path)

    #     return user_config

    # def import_user_dir(self):
    #     # Try user_dir options in order of MMF configuration hierarchy
    #     # First try the default one, which can be set via environment as well
    #     user_dir = self._default_config.env.user_dir

    #     # Now, check user's config
    #     user_config_user_dir = self._user_config.get("env", {}).get("user_dir", None)

    #     if user_config_user_dir:
    #         user_dir = user_config_user_dir

    #     # Finally, check opts
    #     opts_user_dir = self._opts_config.get("env", {}).get("user_dir", None)
    #     if opts_user_dir:
    #         user_dir = opts_user_dir

    #     if user_dir:
    #         import_user_module(user_dir)

    def _build_model_config(self, config_path):

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

        return OmegaConf.load(config_path)


    def _build_runner_config(self, config_path):
        return OmegaConf.load(config_path)


    def _build_dataset_config(self, config_path):

        # dataset = config.get("dataset", None)
        # datasets = config.get("datasets", None)

        # if dataset is None and datasets is None:
        #     raise KeyError("Required argument 'dataset|datasets' not passed")

        # if datasets is None:
        #     config.datasets = dataset
        #     datasets = dataset.split(",")
        # else:
        #     datasets = datasets.split(",")

        # dataset_config = OmegaConf.create()

        # for dataset in datasets:
        #     builder_cls = registry.get_builder_class(dataset)

        #     if builder_cls is None:
        #         warning = f"No dataset named '{dataset}' has been registered"
        #         warnings.warn(warning)
        #         continue
        #     default_dataset_config_path = builder_cls.config_path()

        #     if default_dataset_config_path is None:
        #         warning = (
        #             f"Dataset {dataset}'s builder class has no default configuration "
        #             + "provided"
        #         )
        #         warnings.warn(warning)
        #         continue

        #     dataset_config = OmegaConf.merge(
        #         dataset_config, load_yaml(default_dataset_config_path)
        #     )

        # return dataset_config
        return OmegaConf.load(config_path)

    def get_config(self):
        self._register_resolvers()
        return self.config

    # def _build_demjson_config(self, demjson_string):
    #     if demjson_string is None:
    #         return OmegaConf.create()

    #     try:
    #         import demjson
    #     except ImportError:
    #         logger.warning("demjson is required to use config_override")
    #         raise

    #     demjson_dict = demjson.decode(demjson_string)
    #     return OmegaConf.create(demjson_dict)

    def _get_args_config(self, args):
        args_dict = vars(args)
        return OmegaConf.create(args_dict)

    # def _register_resolvers(self):
    #     OmegaConf.clear_resolvers()
    #     # Device count resolver
    #     device_count = max(1, torch.cuda.device_count())
    #     OmegaConf.register_new_resolver("device_count", lambda: device_count)
    #     OmegaConf.register_new_resolver("resolve_cache_dir", resolve_cache_dir)
    #     OmegaConf.register_new_resolver("resolve_dir", resolve_dir)

    # def freeze(self):
    #     OmegaConf.set_struct(self.config, True)

    # def defrost(self):
    #     OmegaConf.set_struct(self.config, False)

    def _convert_to_dot_list(self, opts):
        if opts is None:
            opts = []

        if len(opts) == 0:
            return opts

        has_equal = opts[0].find("=") != -1

        if has_equal:
            return opts

        return [(opt + "=" + value) for opt, value in zip(opts[0::2], opts[1::2])]

    # def pretty_print(self):
    #     if not self.config.training.log_detailed_config:
    #         return

    #     logger.info("=====  Training Parameters    =====")
    #     logger.info(self._convert_node_to_json(self.config.training))

    #     logger.info("======  Dataset Attributes  ======")
    #     datasets = self.config.datasets.split(",")

    #     for dataset in datasets:
    #         if dataset in self.config.dataset_config:
    #             logger.info(f"======== {dataset} =======")
    #             dataset_config = self.config.dataset_config[dataset]
    #             logger.info(self._convert_node_to_json(dataset_config))
    #         else:
    #             logger.warning(f"No dataset named '{dataset}' in config. Skipping")

    #     logger.info("======  Optimizer Attributes  ======")
    #     logger.info(self._convert_node_to_json(self.config.optimizer))

    #     if self.config.model not in self.config.model_config:
    #         raise ValueError(f"{self.config.model} not present in model attributes")

    #     logger.info(f"======  Model ({self.config.model}) Attributes  ======")
    #     logger.info(
    #         self._convert_node_to_json(self.config.model_config[self.config.model])
    #     )

    # def _convert_node_to_json(self, node):
    #     container = OmegaConf.to_container(node, resolve=True)
    #     return json.dumps(container, indent=4, sort_keys=True)

    # def _update_specific(self, config):
    #     # tp = self.config.training

    #     # if args["seed"] is not None or tp['seed'] is not None:
    #     #     print(
    #     #         "You have chosen to seed the training. This will turn on CUDNN "
    #     #         "deterministic setting which can slow down your training "
    #     #         "considerably! You may see unexpected behavior when restarting "
    #     #         "from checkpoints."
    #     #     )

    #     # if args["seed"] == -1:
    #     #     self.config["training"]["seed"] = random.randint(1, 1000000)

    #     if (
    #         "learning_rate" in config
    #         and "optimizer" in config
    #         and "params" in config.optimizer
    #     ):
    #         lr = config.learning_rate
    #         config.optimizer.params.lr = lr

    #     # TODO: Correct the following issue
    #     # This check is triggered before the config override from
    #     # commandline is effective even after setting
    #     # training.device = 'xla', it gets triggered.
    #     if not torch.cuda.is_available() and "cuda" in config.training.device:
    #         warnings.warn(
    #             "Device specified is 'cuda' but cuda is not present. "
    #             + "Switching to CPU version."
    #         )
    #         config.training.device = "cpu"

    #     return config

    # def upgrade(self, config):
    #     mapping = {
    #         "training.resume_file": "checkpoint.resume_file",
    #         "training.resume": "checkpoint.resume",
    #         "training.resume_best": "checkpoint.resume_best",
    #         "training.load_pretrained": "checkpoint.resume_pretrained",
    #         "training.pretrained_state_mapping": "checkpoint.pretrained_state_mapping",
    #         "training.run_type": "run_type",
    #     }

    #     for old, new in mapping.items():
    #         value = OmegaConf.select(config, old)
    #         if value:
    #             OmegaConf.update(config, new, value)