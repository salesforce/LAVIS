import os
import copy
from dataclasses import dataclass
import json
from glob import glob
import random
from typing import Dict, Optional, Sequence, List, Iterator
from operator import itemgetter
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler, Sampler
import transformers

from PIL import Image

import conversation as conversation_lib
from data_utils import DataInfo

from open_flamingo.train.any_res_data_utils import process_anyres_image
from data_configs.data_paths import IMAGE_FOLDER_DICT_GCP


LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN = "<image>"


def get_image_fullpath(image_file):
    image_file_fp = None
    for k, v in IMAGE_FOLDER_DICT_GCP.items():
        if k in image_file:
            image_file_fp = image_file.replace(k, v)
            break
    if image_file_fp is None:
        print(f"File not found: {image_file}")
        exit(0)
    return image_file_fp


def preprocess_phi_3(
    sources,
    conv_template,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conv_template.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations.
    # Truncate to 2048 to save memory.
    if tokenizer.model_max_length > 2048:
        max_len = 2048
    else:
        max_len = tokenizer.model_max_length

    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=max_len, 
        truncation=True,
    ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.PHI_3

    # Mask targets
    sep = conv.roles[1] + "\n"
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2+'\n')
        rounds_len = len(rounds)
        cur_len = 0 # No <bos> token.
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            rou += conv.sep2+'\n'
            if sep in rou:
                # assistant round
                round_ids = tokenizer(rou,
                              max_length=max_len, 
                              truncation=True).input_ids
                role_prefix_ids = tokenizer(sep).input_ids
                len_prefix = len(role_prefix_ids)
                round_ids = round_ids[len_prefix:]
                round_len = len(round_ids)
            elif conv.roles[0] in rou:
                # user round
                rou += sep
                if has_image:
                    round_ids = tokenizer(rou,
                              max_length=max_len, 
                              truncation=True).input_ids
                    if i > 0:
                        round_ids = round_ids[2:] # Skip the bos tokens
                    round_len = len(round_ids)
                    instruction_len = round_len # All are instructions.
                else:
                    round_ids = tokenizer(rou).input_ids
                    if i > 0:
                        round_ids = round_ids[2:] # Skip the bos tokens
                    round_len = len(round_ids)
                    instruction_len = round_len
                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            else:
                # system round
                round_ids = tokenizer(rou,
                              max_length=max_len, 
                              truncation=True).input_ids
                round_len = len(round_ids)
                instruction_len = round_len # All are instructions.
                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX

        if cur_len < max_len: # The input_ids are truncated to this max length.
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_phi_3_new(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    role_mapping = {"human": "user", "gpt": "assistant"}
    roles = ("<|user|>", "<|assistant|>")
    sep="<s>"
    sep2="<|end|>"

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        # TODO: add system prompt is there's not any in source.
        
        # Update key names
        for i, rnd in enumerate(source):
            if "from" in rnd:
                if rnd["from"] in ["human", "gpt"]:
                    rnd["role"] = role_mapping[rnd.pop("from")]
                else:
                    rnd["role"] = rnd.pop("from")
            if "value" in rnd:
                rnd["content"] = rnd.pop("value")
        # Apply chat template
        tokenizer.chat_template = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') %}{{'<|user|>' + '\n' + message['content'] + '<|end|>' + '\n' + '<|assistant|>' + '\n'}}{% elif message['role'] == 'system' %}{{ '<|system|>' + '\n' + message['content'] + '<|end|>' + '\n'}}{% elif (message['role'] == 'assistant') %}{{message['content'] + '<|end|>' + '\n'}}{% endif %}{% endfor %}"
        chat_conv = tokenizer.apply_chat_template(source, tokenize=False)
        chat_conv = chat_conv.replace(tokenizer.bos_token,'')

        conversations.append(chat_conv)

    # Tokenize conversations
    if tokenizer.model_max_length > 2048:
        max_len = 2048
    else:
        max_len = tokenizer.model_max_length
    
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=max_len,
        truncation=True,
    ).input_ids

    targets = input_ids.clone()

    # assert conv.sep_style == conversation_lib.SeparatorStyle.PHI_3

    # Mask targets
    sep = roles[1] + "\n"
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(sep2+'\n')
        cur_len = 0 # No <bos> token.
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            rou += sep2+'\n'
            if sep in rou:
                # assistant round
                round_ids = tokenizer(rou,
                              max_length=max_len, 
                              truncation=True).input_ids
                role_prefix_ids = tokenizer(sep).input_ids
                len_prefix = len(role_prefix_ids)
                round_ids = round_ids[len_prefix:]
                round_len = len(round_ids)
            elif roles[0] in rou:
                # user round
                rou += sep
                round_ids = tokenizer(rou,
                          max_length=max_len, 
                          truncation=True).input_ids
                if i > 0:
                    round_ids = round_ids[1:] # Skip the bos tokens
                round_len = len(round_ids)
                instruction_len = round_len # All are instructions.
                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            else:
                # system round
                round_ids = tokenizer(rou,
                          max_length=max_len, 
                          truncation=True).input_ids
                round_len = len(round_ids)
                instruction_len = round_len # All are instructions.
                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX

        if cur_len < max_len: # The input_ids are truncated to this max length.
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    conv_template_name: Optional[str] = None,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conv_template_name is not None and conv_template_name in conversation_lib.conv_templates.keys():
        # Use the specified preproseccing func.
        conv_template = conversation_lib.conv_templates[conv_template_name]
    else:
        conv_template = conversation_lib.default_conversation

    if conv_template.version.startswith("phi_3"):
        return preprocess_phi_3_new(sources, tokenizer)
    else:
        raise NotImplementedError


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 image_processor,
                 data_args,
                #  data_args: DataArguments
                 ):
        super(LazySupervisedDataset, self).__init__()
        if isinstance(data_path, str) and os.path.isfile(data_path):
            # Load the default 650k data mix.
            list_data_dict = json.load(open(data_path, "r"))
        elif isinstance(data_path, str) and os.path.isdir(data_path):
            # Load a custom mixture of data with a list of json files.
            json_lists = glob(os.path.join(data_path, '*.json'))
            list_data_dict = []
            for json_file in json_lists:
                list_data_dict.extend(json.load(open(json_file, "r")))
        elif isinstance(data_path, Dict):
            list_data_dict = []
            for json_file, n_sample in data_path.items():
                d_json = json.load(open(json_file, "r"))
                if n_sample > len(d_json):
                    list_data_dict.extend(random.Random(42).choices(d_json, k=n_sample))
                else:
                    list_data_dict.extend(random.Random(42).sample(d_json, k=n_sample))
        else:
            raise ValueError(f"Unknown data_path type: {data_path}")

        # rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.conv_template_name = data_args.conv_template_name
        self.list_data_dict = list_data_dict
        self.data_args = data_args

        self.anyres_grids = []
        base_img_size = self.image_processor.transforms[0].size[0]
        for (m,n) in data_args.anyres_grids:
            self.anyres_grids.append([base_img_size*m, base_img_size*n])

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def _process_single_image(self, image_file) -> Dict[str, torch.Tensor]:
        image_file_fullpath = get_image_fullpath(image_file)
        success = True
        try:
            image = Image.open(image_file_fullpath).convert('RGB')
        except:
            print(f"error opening the file: {image_file_fullpath}")
            success = False
            return success, None, None
        processor = self.image_processor
        img_size = image.size
        if self.data_args.image_aspect_ratio == 'pad':
            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result
            # FIXME: Hardcoded workaround to work with torchvision.Compose()
            image = expand2square(image, tuple(int(x*255) for x in processor.transforms[-1].mean)) 
            image = processor(image) # FIXME: whether to take the 0-th item.
        elif self.data_args.image_aspect_ratio == "anyres":
            # Return image shape: [N_patch, C, H, W]
            image = process_anyres_image(image, processor, self.anyres_grids)
        else:
            image = processor(image)
        
        return success, image, img_size
    
    def _check_img_token_nums(self, source):
        keep_sample = True
        if 'image' not in source:
            # Make sure no <image> token in text-only samples.
            for conv in source["conversations"]:
                n_img_token = conv["value"].count(DEFAULT_IMAGE_TOKEN)
                if n_img_token > 0:
                    keep_sample = False
                    break
            return keep_sample, source
        n_image = len(source['image']) if isinstance(source['image'], list) else 1
        if n_image > 1:
            # FIXME: the checker below doesn't work for mantis. Currently only check for single image data.
            return keep_sample, source
        for conv in source["conversations"]:
            if conv["from"] == "human":
                n_img_token = conv["value"].count(DEFAULT_IMAGE_TOKEN)
                if not n_img_token == n_image:
                    # print(source)
                    conv["value"] = conv["value"].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                    conv["value"] = f"{DEFAULT_IMAGE_TOKEN}\n" * n_image + conv["value"]
                break
        return keep_sample, source

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        keep_sample, sources = self._check_img_token_nums(sources)
        if not keep_sample:
            return self.__getitem__(i+1)
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        # Add the system prompt.
        system_round = {
              "from": "system",
              "value": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
            }
        if sources[0]["conversations"][0]["from"] != "system":
            sources[0]["conversations"] = [system_round] + sources[0]["conversations"]

        if 'image' in sources[0]:
            has_image = True
            image_file = sources[0]['image']
            if isinstance(image_file, list):
                # FIXME: Skipping samples with more than 4 images to avoid OOM issue.
                if len(image_file) > 4:
                    return self.__getitem__(i+1)
                image = []
                img_size = []
                for single_image in image_file:
                    success, image_i, img_size_i = self._process_single_image(single_image)
                    if not success:
                        # Skip the entire sample if one of the images can't be opened.
                        return self.__getitem__(i+1)
                    image.append(image_i)
                    img_size.append(img_size_i)
            elif isinstance(image_file, str):
                success, image, img_size = self._process_single_image(image_file)
                if not success:
                    # Skip the entire sample if one of the images can't be opened.
                    return self.__getitem__(i+1)
            else:
                raise NotImplementedError(f"Unknown image_file type: {image_file}")
            sources = copy.deepcopy([e["conversations"] for e in sources])
        else:
            has_image = False
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            conv_template_name=self.conv_template_name)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if has_image:
            if isinstance(image, list):
                # Multi-image, each image can be of 4-dim (anyres) or 3-dim (base res)
                data_dict['image'] = image 
                if image[0].ndim == 3:
                    # Stack base res image groups along the T-dim.
                    image = torch.stack(image, dim=0)
                    data_dict['image'] = image.unsqueeze(1) # [T, 1, C, H, W]
            elif image.ndim == 4: # Any-res image patches of a single image - use the F dim for N-patches.
                data_dict['image'] = image[None, :]
            else: # single image, single frame
                data_dict['image'] = image[None, None, :] # Expand dims with [T_img, F] to be compatible with flamingo-like vision encoding.
            data_dict['image_size'] = img_size
        else:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.image_processor.transforms[0].size # FIXME: Hardcoded workaround to work with torchvision.Compose()
            data_dict['image'] = torch.zeros(1, 1, 3, crop_size[0], crop_size[1]) # Expand dims with [T_img, F] to be compatible with flamingo-like vision encoding.
            data_dict['image_size'] = crop_size
        return data_dict


def stack_with_padding(list_of_tensors, padding_value=0, padding_side="right"):
    """
    Stack a list of tensors with padding on one side
    Args:
        list_of_tensors (list[torch.Tensor]): List of tensors to stack
        padding_value (int, optional): Value to pad with. Defaults to 0.
        padding_side (str, optional): Side to pad on. Defaults to "right".
    Returns:
        torch.Tensor: Stacked tensors
    """
    max_tokens = max(tensor.size(0) for tensor in list_of_tensors)
    padded_tensors = []
    for tensor in list_of_tensors:
        num_tokens = tensor.size(0)

        padding = torch.full(
            (max_tokens - num_tokens,) + tuple(tensor.shape[1:]),
            padding_value,
            dtype=tensor.dtype,
            device=tensor.device,
        )

        padded_tensor = (
            torch.cat((tensor, padding), dim=0)
            if padding_side == "right"
            else torch.cat((padding, tensor), dim=0)
        )
        padded_tensors.append(padded_tensor)
    return torch.stack(padded_tensors)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    image_aspect_ratio: str

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            image_size = [instance['image_size'] for instance in instances]
            batch['image_size'] = image_size
            if any(isinstance(x, list) for x in images):
                images_list = []
                for x in images:
                    if isinstance(x, torch.Tensor):
                        images_list.append([x])
                    elif isinstance(x, list):
                        images_list.append(x)
                    else:
                        raise NotImplementedError(f"Unknown data type: {x}")
                image_size_list = []
                for x in image_size:
                    if not isinstance(x, list):
                        image_size_list.append([x])
                    else:
                        image_size_list.append(x)
                batch['images'] = images_list
                batch['image_size'] = image_size_list
            elif images[0].shape[0] == 1 and all(x is not None and x.shape == images[0].shape for x in images):
                # stacking images when not using anyres.
                batch['images'] = torch.stack(images)
            elif images[0].ndim == 5 and self.image_aspect_ratio != 'anyres':
                # Stacking batch of multi-image base-res image groups with padding.
                batch['images'] = stack_with_padding(images)
            else:
                batch['images'] = images

        return batch


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    if generator is not None:
        torch.manual_seed(42)
    megabatch_indices = torch.randperm(len(megabatches), generator=generator.manual_seed(42))
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.

        Args:
            index: index of the element in the dataset

        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)



class DistributedSamplerWrapper(DistributedSampler):
    """
    https://github.com/catalyst-team/catalyst/blob/master/catalyst/data/sampler.py
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.

    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.

    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """

        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
                distributed training
            rank (int, optional): Rank of the current process
                within ``num_replicas``
            shuffle (bool, optional): If true (default),
                sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.

        Returns:
            python iterator
        """
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                image_processor,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                image_processor=image_processor,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, 
                                                     image_aspect_ratio=data_args.image_aspect_ratio)
    
    if data_args.data_sampler_group_by_length:
        # Use length grouped sampler for more balanced GPU usages.
        lengths = train_dataset.modality_lengths
        sampler_inner = LengthGroupedSampler(
                    data_args.batch_size,
                    world_size=data_args.world_size * data_args.gradient_accumulation_steps,
                    lengths=lengths,
                    group_by_modality=True,
                    generator=torch.Generator().manual_seed(42),
                )
        sampler = DistributedSamplerWrapper(
            sampler=sampler_inner,
            num_replicas=data_args.world_size,
            rank=data_args.rank,
            shuffle=False
        )
    else:
        sampler = DistributedSampler(
                    train_dataset,
                    shuffle=True,
                    num_replicas=dist.get_world_size(),
                    rank=dist.get_rank(),
                )
    # sampler = None
    
    data_loader = DataLoader(
        train_dataset,
        batch_size=data_args.batch_size,
        num_workers=data_args.workers,
        pin_memory=True, 
        sampler=sampler,
        shuffle=sampler is None,
        collate_fn=data_collator,
    )
    return DataInfo(
        name='instruction-finetune-mix',
        dataloader=data_loader,
        batch_size=data_args.batch_size,
        loss_multiplier=1.0,
        shared_epoch=None,
        sampler=sampler,
    ), len(train_dataset)

