# Copyright (c) Facebook, Inc. and its affiliates.
# from mmf.utils.env import setup_imports

class Registry:
    mapping = {
        "builder_name_mapping": {},
        "task_name_mapping": {},

        "state": {},
    }

    @classmethod
    def register_builder(cls, name):
        r"""Register a dataset builder to registry with key 'name'

        Args:
            name: Key with which the builder will be registered.

        Usage:

            from common.registry import registry
            from datasets.base_dataset_builder import BaseDatasetBuilder
        """

        def wrap(builder_cls):
            from datasets.builders.base_dataset_builder import BaseDatasetBuilder

            assert issubclass(
                builder_cls, BaseDatasetBuilder
            ), "All builders must inherit BaseDatasetBuilder class"
            cls.mapping["builder_name_mapping"][name] = builder_cls
            return builder_cls

        return wrap
    
    @classmethod
    def register_task(cls, name):
        r"""Register a task to registry with key 'name'

        Args:
            name: Key with which the task will be registered.

        Usage:

            from common.registry import registry
        """
        def wrap(task_cls):
            from tasks.base_task import BaseTask

            assert issubclass(
                task_cls, BaseTask
            ), "All tasks must inherit BaseTask class"
            cls.mapping["task_name_mapping"][name] = task_cls
            return task_cls
        
        return wrap

    @classmethod
    def register(cls, name, obj):
        r"""Register an item to registry with key 'name'

        Args:
            name: Key with which the item will be registered.

        Usage::

            from common.registry import registry

            registry.register("config", {})
        """
        path = name.split(".")
        current = cls.mapping["state"]

        for part in path[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[path[-1]] = obj

    # @classmethod
    # def get_trainer_class(cls, name):
    #     return cls.mapping["trainer_name_mapping"].get(name, None)

    @classmethod
    def get_builder_class(cls, name):
        return cls.mapping["builder_name_mapping"].get(name, None)
    
    @classmethod
    def get_task_class(cls, name):
        return cls.mapping["task_name_mapping"].get(name, None)

    @classmethod
    def get(cls, name, default=None, no_warning=False):
        r"""Get an item from registry with key 'name'

        Args:
            name (string): Key whose value needs to be retrieved.
            default: If passed and key is not in registry, default value will
                     be returned with a warning. Default: None
            no_warning (bool): If passed as True, warning when key doesn't exist
                               will not be generated. Useful for MMF's
                               internal operations. Default: False
        Usage::

            from mmf.common.registry import registry

            config = registry.get("config")
        """
        original_name = name
        name = name.split(".")
        value = cls.mapping["state"]
        for subname in name:
            value = value.get(subname, default)
            if value is default:
                break

        if (
            "writer" in cls.mapping["state"]
            and value == default
            and no_warning is False
        ):
            cls.mapping["state"]["writer"].warning(
                "Key {} is not present in registry, returning default value "
                "of {}".format(original_name, default)
            )
        return value

    @classmethod
    def unregister(cls, name):
        r"""Remove an item from registry with key 'name'

        Args:
            name: Key which needs to be removed.
        Usage::

            from mmf.common.registry import registry

            config = registry.unregister("config")
        """
        return cls.mapping["state"].pop(name, None)


registry = Registry()