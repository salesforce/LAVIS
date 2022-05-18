# from torch.utils.data.dataset import ConcatDataset
from datasets.datasets.base_dataset import ConcatDataset


def concat_datasets(datasets):
    if len(datasets) == 1:
        return datasets[list(datasets.keys())[0]]
    else:
        concat_datasets = dict()

        # reorganize by split
        for _, dataset in datasets.items():
            for split_name, dataset_split in dataset.items():
                if split_name not in concat_datasets:
                    concat_datasets[split_name] = [dataset_split]
                else:
                    concat_datasets[split_name].append(dataset_split)

        # concatenate datasets in the same split
        for split_name in concat_datasets:
            if split_name != "train":
                assert (
                    len(concat_datasets[split_name]) == 1
                ), "Do not support multiple {} datasets.".format(split_name)
                concat_datasets[split_name] = concat_datasets[split_name][0]
            else:
                concat_datasets["train"] = ConcatDataset(concat_datasets["train"])

        return concat_datasets
