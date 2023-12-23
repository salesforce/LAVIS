"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

"""
Requires Transformer 4.28 and above, implementation may change according the Llama implementation
"""
import logging
import string
from packaging import version
import os
from omegaconf import OmegaConf

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
from torch.nn.modules.module import _IncompatibleKeys

from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
)

import transformers
import random
from lavis.common.registry import registry
from lavis.models.base_model import BaseModel
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train, LayerNorm
from lavis.models.ulip_models.ULIP_models import ULIP_PointBERT
from lavis.tasks.multimodal_classification import MultimodalClassificationTask

from lavis.common.utils import is_url
from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel
from lavis.common.dist_utils import download_cached_file
from lavis.processors.blip_processors import BlipCaptionProcessor

class CastOutputToFloat(nn.Sequential):
    def forward(self, x): return super().forward(x).to(torch.float32)


@registry.register_model("blip2_vicuna_xinstruct")
class Blip2VicunaXInstruct(Blip2Base):
    """
    BLIP2 Vicuna model.
    Supported model types:
        - vicuna7b
        - vicuna13b
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_vicuna_xinstruct", "vicuna7b")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "vicuna7b": "configs/models/blip2/blip2_xinstruct_vicuna7b.yaml",
        "vicuna13b": "configs/models/blip2/blip2_xinstruct_vicuna13b.yaml",
    }

    SEQUENCIAL_ENCODERS = [
        "eva_clip_g", 
        "beats"
    ]

    SEQUENCIAL_MODALITIES = [
        "video", 
        "audio"
    ]

    MODALITY_TO_CUE = {
        "image": " image: ",
        "pc": " 3d: ",
        "video": " video: ",
        "audio": " audio: ",
    }

    def __init__(
        self,

        modalities = ["image", "pc", "audio", "video"],
        use_cues=True,
        num_query_token=32, 
        qformer_text_input=True,
        llm_text_input=False,
        apply_lemmatizer=False,

        ## encoders
        image_model="eva_clip_g",
        pc_model="ulip2_pointbert",
        video_model="eva_clip_g",
        audio_model="beats",

        image_encoder_kwargs = {"image_size": 224, "drop_path_rate": 0, "use_grad_checkpoint": False},
        pc_encoder_kwargs = {},
        video_encoder_kwargs = {},
        audio_encoder_kwargs = {},

        image_precision="fp16",
        pc_precision="fp16",
        video_precision="fp16",
        audio_precision="fp16",

        freeze_image=True,
        freeze_pc=True,
        freeze_video=True,
        freeze_audio=True,

        ## load pretrained parameters
        pretrained_image_qformer=None,
        pretrained_pc_qformer=None,
        pretrained_video_qformer=None,
        pretrained_audio_qformer=None,

        load_attention_image_qformer=False,
        load_attention_pc_qformer=False,
        load_attention_video_qformer=False,
        load_attention_audio_qformer=False,

        load_qformer_type_image="",
        load_qformer_type_pc="",
        load_qformer_type_video="",
        load_qformer_type_audio="",

        load_ln_type_image="",
        load_ln_type_pc="",
        load_ln_type_video="",
        load_ln_type_audio="",

        load_projection_image=True,
        load_projection_pc=True,
        load_projection_video=True,
        load_projection_audio=True,

        load_projection_type_image="",
        load_projection_type_pc="",
        load_projection_type_video="",
        load_projection_type_audio="",
        
        ## llm model parameters
        llm_model="",
        lora_model="",
        lora=False,

        ## generation parameters
        prompt="",
        prefix="",
        postfix="",
        max_txt_len=128,
        max_output_txt_len=256,
        special_qformer_input_prompt=False,
        enumerate_inputs=False,
        add_space=False,
        remove_start=False,
        clean_tokenization=False, # if set to true removes whitespace from cue, and start token from prompt. 

        ## shared Q-former setup
        shared_qformer=False,
        pretrained_shared_qformer=None,
        load_attention_shared_qformer=False,
        load_qformer_type_shared="",
        load_projection_shared=False,
        load_projection_type_shared="",
        encoder_projection_type_image="",
        encoder_projection_type_pc="",  
        encoder_projection_type_video="", 
        encoder_projection_type_audio="", 
        shared_qformer_num_features=512,

        ## use cached features
        cached_audio=False,
        cached_image=False,
        cached_pc=False,
        cached_video=False,

        ## num features for modality (only needed in cached cases.)
        num_features_audio=768,
        num_features_image=1408,
        num_features_video=1408,
        num_features_pc=512,

        joint_video_audio=False,

        ## DisCRN
        use_caption=False,
        use_describe=False,

        ## classification setup
        predict_with_gen=False,
        format_candidates_prompt="{}",


        ## projection only parameters
        projection_only=False,
        projection_only_audio=False,
        projection_only_pc=False,
        projection_only_video=False,
        projection_only_image=False,

        projection_path_audio=False,
        projection_path_pc=False,
        projection_path_video=False,
        projection_path_image=False,

        proj_dim=1,


        ):

        super().__init__()

        transformers_version = version.parse(transformers.__version__)
        assert transformers_version >= version.parse("4.28"), "BLIP-2 Vicuna requires transformers>=4.28"        
        from transformers import LlamaTokenizer
        from lavis.models.blip2_models.modeling_llama import LlamaForCausalLM
        logging.info(f"Using modalities {modalities}")
        self.modalities = modalities

        logging.info(f"Shared Qformer is set to {shared_qformer}")
        self.shared_qformer = shared_qformer

        logging.info(f"Video-audio interleaving is set to {joint_video_audio}")
        self.joint_video_audio = joint_video_audio

        logging.info(f"Using Spacy en_core_wb_sm lemmatizer is set to {apply_lemmatizer}")
        self._lemmatizer = None
        self.apply_lemmatizer = apply_lemmatizer

        logging.info(f"Qformer text input {qformer_text_input} and LLM Text Input {llm_text_input}")
        self.qformer_text_input = qformer_text_input
        self.llm_text_input = llm_text_input

        self.projection_only = projection_only
        self.proj_dim = proj_dim
        logging.info(f"Projection only setup is set to {projection_only} with dimension {proj_dim}")

        for modality in self.modalities:
            setattr(self, f"cached_{modality}", locals()[f"cached_{modality}"])
            if locals()[f"cached_{modality}"]:
                setattr(self, f"num_features_{modality}", locals()[f"num_features_{modality}"])
                logging.info(f"Using cached {modality} representation with {getattr(self, f'num_features_{modality}')} embedding dim.")
                
        ### Initialize modality enoders ###
        for modality in self.modalities:
            modality_model = locals()[f"{modality}_model"]
            modality_precision = locals()[f"{modality}_precision"]
            modality_kwargs = locals()[f"{modality}_encoder_kwargs"]
            modality_kwargs['load_ln_path'] = locals()[f"pretrained_shared_qformer"] if shared_qformer else \
                locals()[f"pretrained_{modality}_qformer"]
            setattr(self, f"projection_only_{modality}", locals()[f"projection_only_{modality}"])
            setattr(self, f"projection_path_{modality}", locals()[f"projection_path_{modality}"])
            modality_kwargs['load_ln_type'] = locals()[f"load_ln_type_{modality}"]
            if self.projection_only or locals()[f"projection_only_{modality}"]:
               modality_kwargs['load_ln_path']  = getattr(self, f"projection_path_{modality}")
               modality_kwargs['load_ln_type']  = modality
            setattr(self, f"load_ln_type_{modality}", locals()[f"load_ln_type_{modality}"])
            setattr(self, f"pretrained_{modality}_qformer", locals()[f"pretrained_{modality}_qformer"])
            modality_encoder, modality_ln = getattr(self, f"init_{modality}_encoder")(
                modality_model, 
                precision=modality_precision, 
                **modality_kwargs     
            )
            
            freeze_modality = locals()[f"freeze_{modality}"]
            cached_modality = locals()[f"cached_{modality}"]
            if cached_modality:
                setattr(self, f"{modality}_encoder", modality_encoder)
                setattr(self, f"{modality}_ln", modality_ln)
                continue
            if freeze_modality:
                for name, param in modality_encoder.named_parameters():
                    param.requires_grad = False
                modality_encoder = modality_encoder.eval()
                modality_encoder.train = disabled_train
                logging.info(f"freeze {modality} encoder")
            
            setattr(self, f"{modality}_encoder", modality_encoder)
            setattr(self, f"{modality}_ln", modality_ln)

        ##### Init QFormers ####
        self.tokenizer = self.init_tokenizer(truncation_side="left") # 30523 tokens. 
        self.num_query_token = num_query_token
        if self.shared_qformer: 
            logging.info(f"Initializing shared QFormer with {shared_qformer_num_features} \
            number of features and query tokens of length {num_query_token}")
            setattr(self, f"pretrained_shared_qformer", pretrained_shared_qformer)
            setattr(self, f"load_qformer_type_shared", load_qformer_type_shared)
            self.shared_Qformer, self.shared_query_tokens = self.init_Qformer(
                num_query_token, 
                shared_qformer_num_features,
                pretrained_qformer=pretrained_shared_qformer, 
                load_attention=load_attention_shared_qformer,
                load_qformer_type=load_qformer_type_shared
            )

            if not qformer_text_input:
                self.shared_Qformer.bert.embeddings.word_embeddings = None
                self.shared_Qformer.bert.embeddings.position_embeddings = None
                for layer in self.shared_Qformer.bert.encoder.layer:
                    layer.output = None
                    layer.intermediate = None
            else:
                self.shared_Qformer.resize_token_embeddings(len(self.tokenizer))
            self.shared_Qformer.cls = None

            # Map shared Qformer by reference to all modalities. 
            for modality in self.modalities:
                setattr(self, f"{modality}_Qformer", self.shared_Qformer)
                setattr(self, f"{modality}_query_tokens", self.shared_query_tokens)
                encoder_proj_type=locals()[f"encoder_projection_type_{modality}"]
                setattr(self, f"encoder_projection_type_{modality}", locals()[f"encoder_projection_type_{modality}"])
                modality_encoder_features = getattr(self, f"{modality}_encoder").num_features
                setattr(self, f"{modality}_encoder_projection", self.init_encoder_projection(modality_encoder_features, shared_qformer_num_features, pretrained_shared_qformer, encoder_proj_type))
        else:
            for modality in self.modalities:
                if getattr(self,f"cached_{modality}"):
                    modality_num_features = locals()[f"num_features_{modality}"]
                else:
                    modality_num_features = getattr(self, f"{modality}_encoder").num_features
                
                setattr(self, f"pretrained_{modality}_qformer", locals()[f"pretrained_{modality}_qformer"])
                setattr(self, f"load_qformer_type_{modality}", locals()[f"load_qformer_type_{modality}"])


                setattr(self, f"projection_only_{modality}", locals()[f"projection_only_{modality}"])
                setattr(self, f"projection_path_{modality}", locals()[f"projection_path_{modality}"])

                if self.projection_only or locals()[f"projection_only_{modality}"]:
                    logging.info(f"Initializing {modality} projection")
                    setattr(self, f"pretrained_{modality}_qformer", False)
                    if modality == 'audio' and proj_dim == 1:
                        modality_num_features *= 256 # hack to get full beats embedding. define better.
                    modality_projection = self.init_vicuna_projection(
                                        modality_num_features, 
                                        num_query_token*proj_dim,
                                        load_projection_path=getattr(self, f"projection_path_{modality}"), 
                                        load_projection_type=modality,
                                        projection_key=f"{modality}_projection"
                                        )
                    setattr(self, f"{modality}_projection", modality_projection)
                else:
                    logging.info(f"Initializing {modality} QFormer and query tokens of length {num_query_token}")
                    modality_qformer, modality_query_tokens = self.init_Qformer(
                        num_query_token, 
                        modality_num_features,
                        pretrained_qformer=locals()[f"pretrained_{modality}_qformer"],
                        load_attention=locals()[f"load_attention_{modality}_qformer"],
                        load_qformer_type=locals()[f"load_qformer_type_{modality}"]
                    ) 

                    if not qformer_text_input:
                        modality_qformer.bert.embeddings.word_embeddings = None
                        modality_qformer.bert.embeddings.position_embeddings = None
                        for layer in modality_qformer.bert.encoder.layer:
                            layer.output = None
                            layer.intermediate = None
                    else:
                        modality_qformer.resize_token_embeddings(len(self.tokenizer))
                    modality_qformer.cls = None
                    setattr(self, f"{modality}_Qformer", modality_qformer)
                    setattr(self, f"{modality}_query_tokens", modality_query_tokens)

        ### Set up LLM ###
        logging.info(f"Setting up llm model {llm_model}")
        self.lora = lora
        print(f"Lora is set to {self.lora}")
        self.llm_tokenizer = LlamaTokenizer.from_pretrained(llm_model, use_fast=False, truncation_side="left")
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})
        if self.lora:
            # https://github.com/lxe/llama-peft-tuner/blob/main/finetune_peft.py
            self.llm_model = LlamaForCausalLM.from_pretrained(
            llm_model, 
            load_in_8bit=True,
            torch_dtype=torch.float16
            )
            self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))
            self.peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=8,
                lora_alpha=32, lora_dropout=0.1,
                target_modules=['q_proj', 'v_proj']
            )
            self.llm_model.gradient_checkpointing_enable()
            self.llm_model.enable_input_require_grads()
            self.llm_model.lm_head = CastOutputToFloat(self.llm_model.lm_head)
            self.llm_model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
            self.llm_hidden_size = self.llm_model.config.hidden_size
            self.llm_model = get_peft_model(self.llm_model, self.peft_config)
            self.lora_model = lora_model

        else:
            self.llm_model = LlamaForCausalLM.from_pretrained(
            llm_model, torch_dtype=torch.float16
            )
            self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))
            self.llm_hidden_size = self.llm_model.config.hidden_size

            for name, param in self.llm_model.named_parameters():
                param.requires_grad = False

        # Load LM projections
        if self.shared_qformer and load_projection_shared:
            qformer = getattr(self, f"shared_Qformer")
            load_projection_path = locals()[f"load_projection_shared"]
            if load_projection_path:
                load_projection_path = locals()[f"pretrained_shared_qformer"]
            load_projection_type = locals()[f"load_projection_type_shared"]
            setattr(self, f"load_projection_shared", load_projection_path)
            setattr(self, f"load_projection_type_shared", locals()[f"load_projection_type_shared"])
            logging.info(f"Loading shared Qformer projection.")
            proj = self.init_vicuna_projection(
                qformer.config.hidden_size, 
                self.llm_hidden_size,
                load_projection_path=load_projection_path
                )
            # Map projection by reference to all modalities. 
            for modality in self.modalities:
                setattr(self, f"{modality}_llm_proj", proj)
        else:
            for modality in self.modalities:
                load_projection_path = locals()[f"load_projection_{modality}"]
                if load_projection_path == True:
                    load_projection_path = locals()[f"pretrained_{modality}_qformer"]
                load_projection_type = locals()[f"load_projection_type_{modality}"]
                setattr(self, f"load_projection_{modality}", load_projection_path)
                setattr(self, f"load_projection_type_{modality}", load_projection_type) 
                if self.projection_only or getattr(self, f"projection_only_{modality}"):
                    proj = self.init_vicuna_projection(
                    self.num_query_token if proj_dim==1 else proj_dim,
                    self.num_query_token*self.llm_hidden_size if proj_dim==1 else self.llm_hidden_size,
                    load_projection_path=getattr(self, f"projection_path_{modality}"),  
                    load_projection_type=load_projection_type,
                    )
                else:
                    qformer = getattr(self, f"{modality}_Qformer")
                    proj = self.init_vicuna_projection(
                        qformer.config.hidden_size, 
                        self.llm_hidden_size,
                        load_projection_path=load_projection_path, 
                        load_projection_type=load_projection_type
                        )
                setattr(self, f"{modality}_llm_proj", proj)

        self.clean_tokenization = clean_tokenization
        logging.info(f"Clean tokenization is set to {self.clean_tokenization}")

        self.max_txt_len = max_txt_len
        self.max_output_txt_len = max_output_txt_len
        self.prompt = prompt
        
        self.prefix = prefix
        if self.prefix:
            self.tokenized_prefix = self.llm_tokenizer(self.prefix, return_tensors="pt")

        self.postfix = postfix
        if type(self.postfix) != str or not self.postfix:
            self.postfix = ""
        logging.info(f"Using prefix set to {self.prefix} and postfix set to {self.postfix}.")

        self.use_cues = use_cues
        logging.info(f"Using cues set to {self.use_cues}.")
        if self.use_cues:
            logging.info(f"Modality to cue {Blip2VicunaXInstruct.MODALITY_TO_CUE}")
            self.tokenized_cue = {}
            self.emb_cue = {}
            self.att_cue = {}
            for modality in self.modalities:
                if self.clean_tokenization:
                    Blip2VicunaXInstruct.MODALITY_TO_CUE[modality] = Blip2VicunaXInstruct.MODALITY_TO_CUE[modality].lstrip()
                self.tokenized_cue[modality] = self.llm_tokenizer(Blip2VicunaXInstruct.MODALITY_TO_CUE[modality], return_tensors="pt")
                self.emb_cue[modality] = self.llm_model.get_input_embeddings()(self.tokenized_cue[modality].input_ids.to(self.device))
                self.att_cue[modality] = self.tokenized_cue[modality].attention_mask.to(self.device)
        
       
        ## generation parameters
        self.use_caption=use_caption
        self.use_describe=use_describe
        self.predict_with_gen=predict_with_gen
        self.format_candidates_prompt=format_candidates_prompt
        self.special_qformer_input_prompt=special_qformer_input_prompt
        self.enumerate_inputs=enumerate_inputs
        self.add_space=add_space
        self.remove_start=remove_start
        if self.projection_only:
            self.qformer_text_input=False

    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            llm_tokens['input_ids'].append(
                torch.cat([
                    input_ids[i][:this_input_ones],
                    output_ids[i][1:],
                    input_ids[i][this_input_ones:]
                ])
            )
            llm_tokens['attention_mask'].append(
                torch.cat([
                    input_atts[i][:this_input_ones],
                    output_atts[i][1:],
                    input_atts[i][this_input_ones:]
                ])
            )
        llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
        llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
        return llm_tokens, input_part_targets_len

    def forward(self, samples):
        # print('-----------------')
        # print(samples["text_input"])
        # print(samples["text_output"])
        # print('-----------------')
        if samples == None or samples == {} or not any([modality in samples for modality in self.modalities]):
            return {"loss": torch.tensor(0.0)}

        random.shuffle(self.modalities)

        curr_modalities = [modality for modality in self.modalities if modality in samples]
        excess_modalities = [modality for modality in self.modalities if modality not in curr_modalities]
        # disable gradient in excess modalities
        dummy_loss = 0.
        for modality in excess_modalities:
            if self.shared_qformer:
                for name, param in getattr(self, f"{modality}_encoder_projection").named_parameters():
                    # param.requires_grad = False
                    dummy_loss += param.sum()*0.
            for name, param in getattr(self,f"{modality}_ln").named_parameters():
                # param.requires_grad = False
                dummy_loss += param.sum()*0.
            dummy_loss += getattr(self, f"{modality}_query_tokens").sum()*0.
            for name, param in getattr(self, f'{modality}_Qformer').named_parameters():
                    # param.requires_grad = False
                    dummy_loss += param.sum()*0.
            for name, param in getattr(self, f'{modality}_llm_proj').named_parameters():
                    # param.requires_grad = False
                    dummy_loss += param.sum()*0.
        
        embeds = {}
        query_tokens = {}
        data_atts = {}
        for modality in curr_modalities:
            data = samples[modality]
            ln = getattr(self, f"{modality}_ln")
            encoder = getattr(self, f"{modality}_encoder")
            if modality == "video" and self.video_enc_name in Blip2VicunaXInstruct.SEQUENCIAL_ENCODERS:
                embeds[modality] = []
                data_atts[modality] = []
                for j in range(data.size(2)):
                    this_frame = data[:,:,j,:,:]
                    with self.maybe_autocast():
                        embeds[modality].append(ln(encoder(this_frame)))
                        if self.shared_qformer:
                            embeds[modality][-1] = getattr(self, f"{modality}_encoder_projection")(embeds[modality][j])
                        data_atts[modality].append(torch.ones(embeds[modality][j].size()[:-1], dtype=torch.long).to(self.device))
                # B, Token Size, LM EMB
                if not self.projection_only and not getattr(self, f"projection_only_{modality}"):
                    query_tokens[modality] = getattr(self, f"{modality}_query_tokens").expand(data.size(0), -1, -1)

            elif modality == 'audio' and self.audio_enc_name in Blip2VicunaXInstruct.SEQUENCIAL_ENCODERS:
                embeds[modality] = []
                data_atts[modality] = []
                for j in range(data.size(1)):
                    this_frame = data[:,j,:,:]
                    with self.maybe_autocast():
                        embeds[modality].append(ln(encoder(this_frame)))
                        if self.shared_qformer:
                            embeds[modality][j] = getattr(self, f"{modality}_encoder_projection")(embeds[modality][j])
                    data_atts[modality].append(torch.ones(embeds[modality][j].size()[:-1], dtype=torch.long).to(self.device))
                # B, Token Size, LM EMB
                if not self.projection_only and not getattr(self, f"projection_only_{modality}"):
                    query_tokens[modality] = getattr(self, f"{modality}_query_tokens").expand(data.size(0), -1, -1)
            else:
                with self.maybe_autocast():
                    embeds[modality] = ln(encoder(data))
                if len(embeds[modality].size()) == 2:
                    # B, C, D
                    embeds[modality] = embeds[modality].unsqueeze(1)
                # B, C
                if self.shared_qformer:
                    embeds[modality] = getattr(self, f"{modality}_encoder_projection")(embeds[modality])
                data_atts[modality] = torch.ones(embeds[modality].size()[:-1], dtype=torch.long).to(self.device)
            
                # B, Token Size, LM EMB
                if not self.projection_only and not getattr(self, f"projection_only_{modality}"):
                    query_tokens[modality] = getattr(self, f"{modality}_query_tokens").expand(embeds[modality].shape[0], -1, -1)
                    
        query_outputs = {}
        if self.qformer_text_input:
            text_Qformer = self.tokenizer(
                samples["text_input"] if not self.special_qformer_input_prompt else self.special_qformer_input_prompt,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(self.device)

        
            Qformer_atts = {}
            query_atts = {}
            
            for modality in curr_modalities:
                # B, Token Size
                query_atts[modality] = torch.ones(query_tokens[modality].size()[:-1], dtype=torch.long).to(self.device)
                # B, Token Size + Inp Size
                Qformer_atts[modality] = torch.cat([query_atts[modality],text_Qformer.attention_mask],dim=1)
                if modality in Blip2VicunaXInstruct.SEQUENCIAL_MODALITIES and getattr(self, f'{modality}_enc_name') in Blip2VicunaXInstruct.SEQUENCIAL_ENCODERS:
                    num = len(embeds[modality])
                    bs = embeds[modality][0].shape[0]
                    indices = [j_+r for r,j in enumerate([[i*bs for i in range(num)]]*bs) for j_ in j]
                    reordered_embeds = torch.cat(embeds[modality])[indices]
                    reordered_atts = torch.cat(data_atts[modality])[indices]
                    if self.projection_only or getattr(self, f"projection_only_{modality}"):
                        if self.proj_dim != 1:
                            query_outputs[modality] = getattr(self, f"{modality}_projection")(reordered_embeds.mean(1,keepdim=True)).view(bs*num, self.num_query_token, -1)
                        else:
                            query_outputs[modality] = getattr(self, f"{modality}_projection")(reordered_embeds.view(reordered_embeds.shape[0],-1))
                        continue
                    query_output = getattr(self, f"{modality}_Qformer").bert(
                        text_Qformer.input_ids.repeat(num, 1),
                        attention_mask=Qformer_atts[modality].repeat(num, 1),
                        query_embeds=query_tokens[modality].repeat(num, 1, 1),
                        encoder_hidden_states=reordered_embeds,
                        encoder_attention_mask=reordered_atts,
                        return_dict=True,
                    )
                    query_outputs[modality] = query_output
                else:
                    if self.projection_only or getattr(self, f"projection_only_{modality}"):
                        if self.proj_dim != 1:
                            query_outputs[modality] = getattr(self, f"{modality}_projection")(embeds[modality].mean(1, keepdim=True)).reshape(bs, self.num_query_token,-1)
                        else:
                            query_outputs[modality] = getattr(self, f"{modality}_projection")(embeds[modality]).reshape(bs, self.num_query_token,-1)
                        continue  
                    query_outputs[modality] = getattr(self, f"{modality}_Qformer").bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts[modality],
                        query_embeds=query_tokens[modality], 
                        encoder_hidden_states=embeds[modality].to(torch.float32), 
                        encoder_attention_mask=data_atts[modality], 
                        return_dict=True,
                    )
        else:
            for modality in curr_modalities:
                if modality in Blip2VicunaXInstruct.SEQUENCIAL_MODALITIES and getattr(self, f'{modality}_enc_name') in Blip2VicunaXInstruct.SEQUENCIAL_ENCODERS:
                    num = len(embeds[modality])
                    bs  = embeds[modality][0].shape[0]
                    indices = [j_+r for r,j in enumerate([[i*bs for i in range(num)]]*bs) for j_ in j]
                    reordered_embeds = torch.cat(embeds[modality])[indices]
                    reordered_atts = torch.cat(data_atts[modality])[indices]
                    if self.projection_only or getattr(self, f"projection_only_{modality}"):
                        if self.proj_dim != 1:
                            query_outputs[modality] = getattr(self, f"{modality}_projection")(reordered_embeds.mean(1,keepdim=True)).view(bs*num, self.num_query_token, -1)
                        else:
                            query_outputs[modality] = getattr(self, f"{modality}_projection")(reordered_embeds.view(reordered_embeds.shape[0],-1))
                        continue
                    query_output = getattr(self, f"{modality}_Qformer").bert(
                        query_embeds=query_tokens[modality].repeat(num, 1, 1),
                        encoder_hidden_states=reordered_embeds,
                        encoder_attention_mask=reordered_atts,
                        return_dict=True,
                    )
                    query_outputs[modality] = query_output
                else: 
                    bs = embeds[modality].shape[0]
                    if self.projection_only or getattr(self, f"projection_only_{modality}"):
                        if self.proj_dim != 1:
                            query_outputs[modality] = getattr(self, f"{modality}_projection")(embeds[modality].mean(1, keepdim=True)).reshape(bs, self.num_query_token,-1)
                        else:
                            query_outputs[modality] = getattr(self, f"{modality}_projection")(embeds[modality]).reshape(bs, self.num_query_token,-1)
                        continue  
                    query_outputs[modality] = getattr(self, f"{modality}_Qformer").bert(
                        query_embeds=query_tokens[modality],
                        encoder_hidden_states=embeds[modality].to(torch.float32), # pc data is floa16.
                        encoder_attention_mask=data_atts[modality],
                        return_dict=True,
                    )
        
        inputs_llm = {}
        atts_llm = {}
        for modality in curr_modalities:
            if modality in Blip2VicunaXInstruct.SEQUENCIAL_MODALITIES and getattr(self, f'{modality}_enc_name') in Blip2VicunaXInstruct.SEQUENCIAL_ENCODERS:
                # num*bs, num query tokens, llm emb size
                if self.projection_only or getattr(self, f"projection_only_{modality}"):
                    if self.proj_dim != 1:
                        inputs_llm[modality] = getattr(self, f"{modality}_llm_proj")(query_outputs[modality].unsqueeze(1)).reshape(bs*num, self.num_query_token, -1)
                    else:
                        inputs_llm[modality] = getattr(self, f"{modality}_llm_proj")(query_outputs[modality]).reshape(bs*num, self.num_query_token, -1)
                    inputs_llm[modality] = inputs_llm[modality].reshape(bs, num, self.num_query_token, -1).view(bs, num*self.num_query_token, -1)
                    atts_llm[modality] =  torch.ones(inputs_llm[modality].size()[:-1], dtype=torch.long).to(self.device)
                    continue
                inputs_llm[modality] = getattr(self, f"{modality}_llm_proj")(query_outputs[modality].last_hidden_state[:,:query_tokens[modality].size(1),:]) 
                # bs, num, num query tokens, llm emb size -> bs, num*num query tokens, llm emb size
                inputs_llm[modality] = inputs_llm[modality].reshape(bs, num, self.num_query_token, -1).view(bs, num*self.num_query_token, -1)
                atts_llm[modality] =  torch.ones(inputs_llm[modality].size()[:-1], dtype=torch.long).to(self.device)
            else:
                if self.projection_only or getattr(self, f"projection_only_{modality}"):
                    if self.proj_dim == 1:
                        inputs_llm[modality] = getattr(self, f"{modality}_llm_proj")(query_outputs[modality].mean(-1)).reshape(bs, self.num_query_token, -1)
                    else:
                        inputs_llm[modality] = getattr(self, f"{modality}_llm_proj")(query_outputs[modality].reshape(bs, self.num_query_token, -1))
                    atts_llm[modality] =  torch.ones(inputs_llm[modality].size()[:-1], dtype=torch.long).to(self.device)
                    continue
                inputs_llm[modality] = getattr(self, f"{modality}_llm_proj")(query_outputs[modality].last_hidden_state[:,:query_tokens[modality].size(1),:])
                atts_llm[modality] = torch.ones(inputs_llm[modality].size()[:-1], dtype=torch.long).to(self.device)

        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'left'

        if self.llm_text_input:
            text_input_tokens = self.llm_tokenizer(
                [f"{t}{self.postfix}" for t in samples['text_input']] if self.postfix else samples['text_input'],
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                add_special_tokens= not self.clean_tokenization
            ).to(self.device)

        self.llm_tokenizer.truncation_side = 'right'
        text_output_tokens = self.llm_tokenizer(
            [t + self.llm_tokenizer.eos_token for t in samples['text_output']],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_output_txt_len,
        ).to(self.device)

        if self.llm_text_input:
            llm_tokens, input_part_targets_len = self.concat_text_input_output(
                text_input_tokens.input_ids,
                text_input_tokens.attention_mask,
                text_output_tokens.input_ids,
                text_output_tokens.attention_mask,
            )
        else:
            llm_tokens = text_output_tokens
            input_part_targets_len = [0 for _ in range(llm_tokens['input_ids'].shape[0])] # input length is 0

        
        # do not apply loss to the padding
        targets = llm_tokens['input_ids'].masked_fill(
            llm_tokens['input_ids'] == self.llm_tokenizer.pad_token_id, -100
        )

        # do not apply loss to the text input (i.e., instruction)
        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100

        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])

        bs = inputs_embeds.shape[0]

        att_list = []
        inp_list = []

        if self.prefix:
            att_list = [self.tokenized_prefix.attention_mask.repeat(bs, 1).to(self.device)]
            inp_list = [self.llm_model.get_input_embeddings()(self.tokenized_prefix.input_ids.to(self.device)).repeat(bs, 1, 1)]            
        for modality in curr_modalities:
            if self.use_cues:
                if self.prefix and self.clean_tokenization:
                     att_list.extend([self.att_cue[modality][:,1:].repeat(bs, 1).to(self.device), atts_llm[modality]])
                     inp_list.extend([self.emb_cue[modality][:,1:].repeat(bs, 1, 1).to(self.device), inputs_llm[modality]])
                att_list.extend([self.att_cue[modality].repeat(bs, 1).to(self.device), atts_llm[modality]])
                inp_list.extend([self.emb_cue[modality].repeat(bs, 1, 1).to(self.device), inputs_llm[modality]])
            else:
                att_list.extend([atts_llm[modality]])
                inp_list.extend([inputs_llm[modality]])
       
        # do not apply loss to the query tokens
        empty_targets = (
            torch.ones(torch.cat(att_list, dim=1).size(), dtype=torch.long).to(self.device).fill_(-100)
        )

        # append llm prompt + output to queries
        att_list.append(llm_tokens['attention_mask'])
        inp_list.append(inputs_embeds)
        
        inputs_embeds = torch.cat(inp_list, dim=1)
        attention_mask = torch.cat(att_list, dim=1)
        targets = torch.cat([empty_targets, targets], dim=1)

       
        
        with self.maybe_autocast():
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )

        loss = dummy_loss+outputs.loss



        return {"loss": loss}


    def init_image_encoder(self, 
                           model_name, 
                           precision,
                           **kwargs):

        load_ln_path = kwargs['load_ln_path']
        del kwargs['load_ln_path']
        load_ln_type=kwargs['load_ln_type']
        del kwargs['load_ln_type']

        encoder, _ = super().init_vision_encoder(model_name, kwargs['image_size'], kwargs['drop_path_rate'], kwargs['use_grad_checkpoint'], precision)
        ln = self.init_ln(encoder.num_features, load_ln_path=load_ln_path, load_ln_type=load_ln_type)
        return encoder, ln

    def init_pc_encoder(
        self, model_name, precision, **kwargs
        ):
        assert model_name in [
            "ulip1_pointbert",
            "ulip2_pointbert",
            "ulip_shapenet", 
            "ulip_objaverse",
            "objaverse_shapenet_k_1",
            "ulip2_scaledup"
            ""
        ], "pc model must be in [ulip1_pointbert,ulip2_pointbert]"

        load_ln_path = kwargs['load_ln_path']
        del kwargs['load_ln_path']
        load_ln_type=kwargs['load_ln_type']
        del kwargs['load_ln_type']

        if model_name == "ulip2_pointbert":
            pc_encoder = ULIP_PointBERT(ulip_v=2)
        elif model_name == "ulip_shapenet":
            pc_encoder = ULIP_PointBERT(ulip_v="shapenet")
        elif model_name == "ulip_objaverse":
            pc_encoder = ULIP_PointBERT(ulip_v="objaverse_k_1")
        elif model_name == "objaverse_shapenet_k_1":
             pc_encoder = ULIP_PointBERT(ulip_v="objaverse_shapenet_k_1")
        elif model_name == "ulip2_scaledup":
            pc_encoder = ULIP_PointBERT(ulip_v="ulip2_scaledup")
        else:
            pc_encoder = ULIP_PointBERT(ulip_v=1)
        ln_pc = self.init_ln(pc_encoder.num_features, load_ln_path=load_ln_path, load_ln_type=load_ln_type)
        self.pc_enc_name = model_name
        return pc_encoder, ln_pc


    def init_video_encoder(
        self, model_name, precision, **kwargs
        ):
        assert model_name in [
            "eva_clip_g",
            "eva2_clip_L",
            "clip_L",
        ], "video_model must be in [eva_clip_g, eva2_clip_L, clip_L]"

        if model_name in  ["eva_clip_g","eva2_clip_L","clip_L",]:
            video_encoder, ln_video =  self.init_image_encoder(
                model_name, precision=precision, **kwargs
            )
        self.video_enc_name = model_name
        return video_encoder, ln_video
    
    def init_audio_encoder(
        self, model_name, precision, **kwargs
        ):
        assert model_name in [
            'beats'
        ], "audio model must be in [beats]"

        load_ln_path = kwargs['load_ln_path']
        del kwargs['load_ln_path']
        load_ln_type=kwargs['load_ln_type']
        del kwargs['load_ln_type']
        if "beats" in model_name:
            from lavis.models.beats_encoder import BeatsEncoder
            if self.cached_audio:
                audio_encoder = lambda x: x
                ln_audio = self.init_ln(768, load_ln_path=load_ln_path, load_ln_type=load_ln_type)
            else:
                audio_encoder = BeatsEncoder(**kwargs)
        if not self.cached_audio:
            ln_audio = self.init_ln(audio_encoder.num_features, load_ln_path=load_ln_path, load_ln_type=load_ln_type)
        self.audio_enc_name = model_name
        return audio_encoder, ln_audio

    @torch.no_grad()
    def get_query_outputs(
        self,
        samples
        ):
        if samples == None or samples == {}:
            return 

        curr_modalities = [modality for modality in self.modalities if modality in samples]
        if len(curr_modalities) == 0:
            print("Model modalities do not match sample modalities.")
            return
        
        # get batch size
        bs = None
        for modality in curr_modalities:
            data = samples[modality]
            bs = data.size(0)
            break
        
        if "prompt" in samples.keys():
            prompt = samples["prompt"]
        elif "text_input" in samples.keys():
            prompt = samples["text_input"]
        else:
            prompt = self.prompt

        if isinstance(prompt, str):
            prompt = [prompt] * bs
        else:
            assert len(prompt) == bs, "The number of prompts must be equal to the batch size."

        embeds = {}
        query_tokens = {}
        data_atts = {}

        for modality in curr_modalities:
            data = samples[modality]
            ln = getattr(self, f"{modality}_ln")
            encoder = getattr(self, f"{modality}_encoder")
            if modality == "video" and self.video_enc_name in Blip2VicunaXInstruct.SEQUENCIAL_ENCODERS:
                embeds[modality] = []
                data_atts[modality] = []
                for j in range(data.size(2)):
                    this_frame = data[:,:,j,:,:]
                    with self.maybe_autocast():
                        embeds[modality].append(ln(encoder(this_frame)))
                        if self.shared_qformer:
                            embeds[modality][-1] = getattr(self, f"{modality}_encoder_projection")(embeds[modality][j])
                        data_atts[modality].append(torch.ones(embeds[modality][j].size()[:-1], dtype=torch.long).to(self.device))
                # B, Token Size, LM EMB
                query_tokens[modality] = getattr(self, f"{modality}_query_tokens").expand(data.size(0), -1, -1)

            elif modality == 'audio' and self.audio_enc_name in Blip2VicunaXInstruct.SEQUENCIAL_ENCODERS:
                embeds[modality] = []
                data_atts[modality] = []
                for j in range(data.size(1)):
                    this_frame = data[:,j,:,:]
                    with self.maybe_autocast():
                        embeds[modality].append(ln(encoder(this_frame)))
                        if self.shared_qformer:
                            embeds[modality][j] = getattr(self, f"{modality}_encoder_projection")(embeds[modality][j])
                    data_atts[modality].append(torch.ones(embeds[modality][j].size()[:-1], dtype=torch.long).to(self.device))
                # B, Token Size, LM EMB
                if not self.projection_only and not getattr(self, f"projection_only_{modality}"):
                    query_tokens[modality] = getattr(self, f"{modality}_query_tokens").expand(data.size(0), -1, -1)
            else:
                with self.maybe_autocast():
                    embeds[modality] = ln(encoder(data))
                if len(embeds[modality].size()) == 2:
                    # B, C, D
                    embeds[modality] = embeds[modality].unsqueeze(1)
                # B, C
                if self.shared_qformer:
                    embeds[modality] = getattr(self, f"{modality}_encoder_projection")(embeds[modality])
                
                data_atts[modality] = torch.ones(embeds[modality].size()[:-1], dtype=torch.long).to(self.device)
            
                # B, Token Size, LM EMB
                if not self.projection_only and not getattr(self, f"projection_only_{modality}"):
                    query_tokens[modality] = getattr(self, f"{modality}_query_tokens").expand(embeds[modality].shape[0], -1, -1)

        query_outputs = {}
        if self.qformer_text_input:
            text_Qformer = self.tokenizer(
                prompt,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(self.device)

           
            Qformer_atts = {}
            query_atts = {}
            num = {}
            for modality in curr_modalities:
                # B, Token Size
                if not self.projection_only and not getattr(self, f"projection_only_{modality}"):
                    query_atts[modality] = torch.ones(query_tokens[modality].size()[:-1], dtype=torch.long).to(self.device)
                    # B, Token Size + Inp Size
                    Qformer_atts[modality] = torch.cat([query_atts[modality],text_Qformer.attention_mask],dim=1)
                if modality in Blip2VicunaXInstruct.SEQUENCIAL_MODALITIES and getattr(self, f'{modality}_enc_name') in Blip2VicunaXInstruct.SEQUENCIAL_ENCODERS:
                    num[modality] = len(embeds[modality])
                    bs = embeds[modality][0].shape[0]
                    indices = [j_+r for r,j in enumerate([[i*bs for i in range(num[modality])]]*bs) for j_ in j]
                    reordered_embeds = torch.cat(embeds[modality])[indices]
                    reordered_atts = torch.cat(data_atts[modality])[indices]
                    if self.projection_only or getattr(self, f"projection_only_{modality}"):
                        if self.proj_dim != 1:
                            query_outputs[modality] = getattr(self, f"{modality}_projection")(reordered_embeds.mean(1,keepdim=True)).view(bs*num[modality], self.num_query_token, -1)
                        else:
                            query_outputs[modality] = getattr(self, f"{modality}_projection")(reordered_embeds.view(reordered_embeds.shape[0],-1))
                        continue
                    query_output = getattr(self, f"{modality}_Qformer").bert(
                        text_Qformer.input_ids.repeat(num[modality], 1),
                        attention_mask=Qformer_atts[modality].repeat(num[modality], 1),
                        query_embeds=query_tokens[modality].repeat(num[modality], 1, 1),
                        encoder_hidden_states=reordered_embeds,
                        encoder_attention_mask=reordered_atts,
                        return_dict=True,
                    )
                    query_outputs[modality] = query_output
                else:
                    if self.projection_only or getattr(self, f"projection_only_{modality}"):
                        if self.proj_dim != 1:
                            query_outputs[modality] = getattr(self, f"{modality}_projection")(embeds[modality].mean(1, keepdim=True)).reshape(bs, self.num_query_token,-1)
                        else:
                            query_outputs[modality] = getattr(self, f"{modality}_projection")(embeds[modality]).reshape(bs, self.num_query_token,-1)
                        continue  
                    query_outputs[modality] = getattr(self, f"{modality}_Qformer").bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts[modality],
                        query_embeds=query_tokens[modality], 
                        encoder_hidden_states=embeds[modality].to(torch.float32), 
                        encoder_attention_mask=data_atts[modality], 
                        return_dict=True,
                    )
        else:
            num = {}
            for modality in curr_modalities:
                if modality in Blip2VicunaXInstruct.SEQUENCIAL_MODALITIES and getattr(self, f'{modality}_enc_name') in Blip2VicunaXInstruct.SEQUENCIAL_ENCODERS:
                    num[modality] = len(embeds[modality])
                    bs  = embeds[modality][0].shape[0]
                    indices = [j_+r for r,j in enumerate([[i*bs for i in range(num[modality])]]*bs) for j_ in j]
                    reordered_embeds = torch.cat(embeds[modality])[indices]
                    reordered_atts = torch.cat(data_atts[modality])[indices]
                    if self.projection_only or getattr(self, f"projection_only_{modality}"):
                        if self.proj_dim != 1:
                            query_outputs[modality] = getattr(self, f"{modality}_projection")(reordered_embeds.mean(1,keepdim=True)).view(bs*num, self.num_query_token, -1)
                        else:
                            query_outputs[modality] = getattr(self, f"{modality}_projection")(reordered_embeds.view(reordered_embeds.shape[0],-1))
                        continue
                    query_output = getattr(self, f"{modality}_Qformer").bert(
                        query_embeds=query_tokens[modality].repeat(num[modality], 1, 1),
                        encoder_hidden_states=reordered_embeds,
                        encoder_attention_mask=reordered_atts,
                        return_dict=True,
                    )
                    query_outputs[modality] = query_output
                else:   
                    if self.projection_only or getattr(self, f"projection_only_{modality}"):
                        if self.proj_dim != 1:
                            query_outputs[modality] = getattr(self, f"{modality}_projection")(embeds[modality].mean(1, keepdim=True)).reshape(bs, self.num_query_token,-1)
                        else:
                            query_outputs[modality] = getattr(self, f"{modality}_projection")(embeds[modality]).reshape(bs, self.num_query_token,-1)
                        continue  
                    query_outputs[modality] = getattr(self, f"{modality}_Qformer").bert(
                        query_embeds=query_tokens[modality],
                        encoder_hidden_states=embeds[modality].to(torch.float32), # pc data is floa16.
                        encoder_attention_mask=data_atts[modality],
                        return_dict=True,
                    )

        for modality in curr_modalities:
            if modality in Blip2VicunaXInstruct.SEQUENCIAL_MODALITIES and getattr(self, f'{modality}_enc_name') in Blip2VicunaXInstruct.SEQUENCIAL_ENCODERS:
                if self.projection_only or getattr(self, f"projection_only_{modality}"):  
                    if self.proj_dim != 1:
                        query_outputs[f'llm_proj_{modality}'] = getattr(self, f"{modality}_llm_proj")(query_outputs[modality].unsqueeze(1)).reshape(bs*num, self.num_query_token, -1)
                    else:
                        query_outputs[f'llm_proj_{modality}'] = getattr(self, f"{modality}_llm_proj")(query_outputs[modality]).reshape(bs*num, self.num_query_token, -1)
                    query_outputs[f'llm_proj_{modality}'] = query_outputs[f'llm_proj_{modality}'].reshape(bs, num[modality], self.num_query_token, -1).contiguous().view(bs, num[modality]*self.num_query_token, -1)
                    query_outputs[modality] = query_outputs[modality].view(bs, num[modality]*self.num_query_token, -1)
                else:
                    query_outputs[f'llm_proj_{modality}']  = getattr(self, f"{modality}_llm_proj")(query_outputs[modality]['last_hidden_state'][:,:query_tokens[modality].size(1),:]).contiguous().view(bs, num[modality]*self.num_query_token, -1)
                    query_outputs[modality] = query_outputs[modality]['last_hidden_state'][:,:query_tokens[modality].size(1),:].contiguous().view(bs, num[modality]*self.num_query_token, -1)


            else:
                if self.projection_only or getattr(self, f"projection_only_{modality}"):
                    if self.proj_dim == 1:
                        query_outputs[f'llm_proj_{modality}'] = getattr(self, f"{modality}_llm_proj")(query_outputs[modality].mean(-1)).reshape(bs, self.num_query_token, -1)
                    else:
                        query_outputs[f'llm_proj_{modality}']= getattr(self, f"{modality}_llm_proj")(query_outputs[modality].reshape(bs, self.num_query_token, -1))
                else:
                    query_outputs[modality] = query_outputs[modality].last_hidden_state[:,:query_tokens[modality].size(1),:]
                    query_outputs[f'llm_proj_{modality}'] = getattr(self, f"{modality}_llm_proj")(query_outputs[modality])

        for modality in curr_modalities:
            query_outputs[f'embeds_{modality}'] = embeds[modality]
        return query_outputs 

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1,
        num_captions=1,
        temperature=1,
        special_qformer_input_prompt=False
        ):
        self.llm_tokenizer.padding_side = "left"

        if samples == None or samples == {}:
            return 

        if 'modalities' in samples:
            curr_modalities = samples['modalities'][0] if isinstance(samples['modalities'][0], list) else  samples['modalities']
        elif self.joint_video_audio:
            curr_modalities = ["video", "audio"]
        else:
            curr_modalities = [modality for modality in self.modalities if modality in samples]

        
        if len(curr_modalities) == 0:
            print("Model modalities do not match sample modalities.")
            return
            
        # get batch size
        bs = None
        for modality in curr_modalities:
            data = samples[modality]
            if isinstance(data, torch.Tensor):
                bs = data.size(0)
            else:
                bs = len(data)
            break
        
        if "prompt" in samples.keys():
            prompt = samples["prompt"]
        elif self.prompt and 'text_input' in samples and '{}' in self.prompt:
            prompt = [self.prompt.format(t) for t in samples["text_input"]]
        elif "text_input" in samples.keys():
            prompt = samples["text_input"]
        else:
            prompt = self.prompt

        if isinstance(prompt, str):
            prompt = [prompt] * bs
        else:
            assert len(prompt) == bs, "The number of prompts must be equal to the batch size."            

        # For TextCaps
        if "ocr_tokens" in samples.keys() and "{}" in prompt[0]:
            prompt = [p.format(', '.join(samples['ocr_tokens'][i][:30])) for i, p in enumerate(prompt)]


        if 'discrn' in samples and self.use_caption: ## discriminatory reasoning
            if self.postfix:
                prompt = [f'{t}{self.postfix}' for t in prompt]
            if self.enumerate_inputs:
                prompt = [f'{self.prefix}(a){Blip2VicunaXInstruct.MODALITY_TO_CUE[samples["modalities"][i][0]] if self.use_cues else " "}{samples["baseline_captions"][i][0]} (b){Blip2VicunaXInstruct.MODALITY_TO_CUE[samples["modalities"][i][1]] if self.use_cues else " "}{samples["baseline_captions"][i][1]} {prompt[i]}' for i in range(bs)]
            else:
                prompt = [f'{self.prefix}{Blip2VicunaXInstruct.MODALITY_TO_CUE[samples["modalities"][i][0]]}{samples["baseline_captions"][i][0] if self.use_cues else " "}{Blip2VicunaXInstruct.MODALITY_TO_CUE[samples["modalities"][i][1]] if self.use_cues else " "}{samples["baseline_captions"][i][1]} {prompt[i]}' for i in range(bs)]
            llm_tokens = self.llm_tokenizer(
                prompt,
                padding="longest",
                return_tensors="pt"
            ).to(self.device)
            inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)
        
            with self.maybe_autocast():
                outputs = self.llm_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=llm_tokens.attention_mask,
                    do_sample=use_nucleus_sampling,
                    top_p=top_p,
                    temperature=temperature,
                    num_beams=num_beams,
                    max_length=max_length,
                    min_length=min_length,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    num_return_sequences=num_captions,
                )
        
            outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)
            output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            output_text = [o.strip() for o in output_text]
            # print(output)
            return output_text

        query_tokens = {}
        for modality in curr_modalities:
            if not self.projection_only and not getattr(self, f"projection_only_{modality}"):
                query_tokens[modality] = getattr(self, f"{modality}_query_tokens").expand(bs, -1, -1)
        if self.qformer_text_input:
            if self.special_qformer_input_prompt or special_qformer_input_prompt:  
                qformer_prompt = special_qformer_input_prompt if special_qformer_input_prompt else self.special_qformer_input_prompt
                qformer_prompt = [qformer_prompt] * len(prompt)
                if "text_input" in samples.keys():
                    if type(samples["text_input"][0]) == list:
                        qformer_prompt = [qformer_prompt[i].format(*samples["text_input"][i]) for i in range(len(qformer_prompt))]
                    else:
                        qformer_prompt = [qformer_prompt[i].format(samples["text_input"][i]) for i in range(len(qformer_prompt))]
                text_Qformer = self.tokenizer(
                    qformer_prompt,
                    padding='longest',
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(self.device)

            elif self.use_describe:
                modality2prompt = {
                    "video": "a short description of the video",
                    "audio": "an audio that shows",
                    "image": "a short image caption",
                    "pc": "a 3d model of"
                }
                qformer_prompt = [modality2prompt[modality] for _ in samples['text_input']]

                text_Qformer = self.tokenizer(
                    qformer_prompt,
                    padding='longest',
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(self.device)

            else:
                text_Qformer = self.tokenizer(
                    prompt,
                    padding='longest',
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(self.device)

            Qformer_atts = {}
            query_atts = {}
            
            for modality in curr_modalities:
                if not  getattr(self, f"projection_only_{modality}"):
                    # B, Token Size
                    query_atts[modality] = torch.ones(query_tokens[modality].size()[:-1], dtype=torch.long).to(self.device)
                    # B, Token Size + Inp Size
                    Qformer_atts[modality] = torch.cat([query_atts[modality],text_Qformer.attention_mask],dim=1)

        embeds = {}
        data_atts = {}
        for modality in curr_modalities:
            data = samples[modality]
            ln = getattr(self, f"{modality}_ln")
            encoder = getattr(self, f"{modality}_encoder")
            if modality == "video" and "clip" in self.video_enc_name:
                embeds[modality] = []
                data_atts[modality] = []
                for j in range(data.size(2)):
                    this_frame = data[:,:,j,:,:]
                    with self.maybe_autocast():
                        embeds[modality].append(ln(encoder(this_frame)))
                        if self.shared_qformer:
                            embeds[modality][j] = getattr(self, f"{modality}_encoder_projection")(embeds[modality][j])
                        data_atts[modality].append(torch.ones(embeds[modality][j].size()[:-1], dtype=torch.long).to(self.device))
            elif modality == 'audio' and 'beats' in self.audio_enc_name:
                embeds[modality] = []
                data_atts[modality] = []
                for j in range(data.size(1)):
                    this_frame = data[:,j,:,:]
                    with self.maybe_autocast():
                        embeds[modality].append(ln(encoder(this_frame)))
                        if self.shared_qformer:
                            embeds[modality][j] = getattr(self, f"{modality}_encoder_projection")(embeds[modality][j])
                    data_atts[modality].append(torch.ones(embeds[modality][j].size()[:-1], dtype=torch.long).to(self.device))
            else:
                with self.maybe_autocast():
                    embeds[modality] = ln(encoder(data))
                if len(embeds[modality].size()) == 2:
                    embeds[modality] = embeds[modality].unsqueeze(1)
                if self.shared_qformer:
                    with self.maybe_autocast():
                        embeds[modality] = getattr(self, f"{modality}_encoder_projection")(embeds[modality])
                data_atts[modality] = torch.ones(embeds[modality].size()[:-1], dtype=torch.long).to(self.device)
            
        query_outputs = {}
        num = {}
        if self.qformer_text_input:
            for modality in curr_modalities:
                if modality in Blip2VicunaXInstruct.SEQUENCIAL_MODALITIES and getattr(self, f'{modality}_enc_name') in Blip2VicunaXInstruct.SEQUENCIAL_ENCODERS:
                    num[modality] = len(embeds[modality])
                    bs = embeds[modality][0].shape[0]
                    indices = [j_+r for r,j in enumerate([[i*bs for i in range(num[modality])]]*bs) for j_ in j]
                    reordered_embeds = torch.cat(embeds[modality])[indices]
                    reordered_atts = torch.cat(data_atts[modality])[indices]
                    if self.projection_only or getattr(self, f"projection_only_{modality}"):
                        if self.proj_dim != 1:
                                query_outputs[modality] = getattr(self, f"{modality}_projection")(reordered_embeds.mean(1,keepdim=True)).view(bs*num[modality], self.num_query_token, -1)
                        else:
                            query_outputs[modality] = getattr(self, f"{modality}_projection")(reordered_embeds.view(reordered_embeds.shape[0],-1))
                        continue
                    query_output = getattr(self, f"{modality}_Qformer").bert(
                        text_Qformer.input_ids.repeat(num[modality], 1),
                        attention_mask=Qformer_atts[modality].repeat(num[modality], 1),
                        query_embeds=query_tokens[modality].repeat(num[modality], 1, 1),
                        encoder_hidden_states=reordered_embeds,
                        encoder_attention_mask=reordered_atts,
                        return_dict=True,
                    )
                    query_outputs[modality] = query_output
                else:
                    bs = embeds[modality].shape[0]
                    if self.projection_only or getattr(self, f"projection_only_{modality}"):
                        if self.proj_dim != 1:
                            query_outputs[modality] = getattr(self, f"{modality}_projection")(embeds[modality].mean(1, keepdim=True)).reshape(bs, self.num_query_token,-1)
                        else:
                            query_outputs[modality] = getattr(self, f"{modality}_projection")(embeds[modality]).reshape(bs, self.num_query_token,-1)
                        continue  
                    query_outputs[modality] = getattr(self, f"{modality}_Qformer").bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts[modality],
                        query_embeds=query_tokens[modality],
                        encoder_hidden_states=embeds[modality].to(torch.float32),
                        encoder_attention_mask=data_atts[modality],
                        return_dict=True,
                    )
        else:
            for modality in curr_modalities:
                if modality in Blip2VicunaXInstruct.SEQUENCIAL_MODALITIES and getattr(self, f'{modality}_enc_name') in Blip2VicunaXInstruct.SEQUENCIAL_ENCODERS:
                    num[modality] = len(embeds[modality])
                    bs = embeds[modality][0].shape[0]
                    indices = [j_+r for r,j in enumerate([[i*bs for i in range(num[modality])]]*bs) for j_ in j]
                    reordered_embeds = torch.cat(embeds[modality])[indices]
                    reordered_atts = torch.cat(data_atts[modality])[indices]
                    if self.projection_only or getattr(self, f"projection_only_{modality}"):
                        if self.proj_dim != 1:
                            query_outputs[modality] = getattr(self, f"{modality}_projection")(reordered_embeds.mean(1,keepdim=True)).view(bs*num[modality], self.num_query_token, -1)
                        else:
                            query_outputs[modality] = getattr(self, f"{modality}_projection")(reordered_embeds.view(reordered_embeds.shape[0],-1))
                        continue
                    query_output = getattr(self, f"{modality}_Qformer").bert(
                        query_embeds=query_tokens[modality].repeat(num[modality], 1, 1),
                        encoder_hidden_states=reordered_embeds,
                        encoder_attention_mask=reordered_atts,
                        return_dict=True,
                    )
                    query_outputs[modality] = query_output
                else:
                    bs = embeds[modality].shape[0]
                    if self.projection_only or getattr(self, f"projection_only_{modality}"):
                        with self.maybe_autocast():
                            if self.proj_dim != 1:
                                query_outputs[modality] = getattr(self, f"{modality}_projection")(embeds[modality].mean(1, keepdim=True)).reshape(bs, self.num_query_token,-1)
                            else:
                                query_outputs[modality] = getattr(self, f"{modality}_projection")(embeds[modality]).reshape(bs, self.num_query_token,-1)
                            continue  
                    query_outputs[modality] = getattr(self, f"{modality}_Qformer").bert(
                        query_embeds=query_tokens[modality],
                        encoder_hidden_states=embeds[modality].to(torch.float32),
                        encoder_attention_mask=data_atts[modality],
                        return_dict=True,
                    )
    
        inputs_llm = {}
        atts_llm = {}
        enumeration = {}

        for i,modality in enumerate(curr_modalities):
            if modality in Blip2VicunaXInstruct.SEQUENCIAL_MODALITIES and getattr(self, f'{modality}_enc_name') in Blip2VicunaXInstruct.SEQUENCIAL_ENCODERS:
                if self.projection_only or getattr(self, f"projection_only_{modality}"):
                    if self.proj_dim != 1:
                        inputs_llm[modality] = getattr(self, f"{modality}_llm_proj")(query_outputs[modality].unsqueeze(1)).reshape(bs*num[modality], self.num_query_token, -1)
                    else:
                        inputs_llm[modality] = getattr(self, f"{modality}_llm_proj")(query_outputs[modality].reshape(bs*num, self.num_query_token, -1))
                    inputs_llm[modality] = inputs_llm[modality].reshape(bs, num[modality], self.num_query_token, -1).view(bs, num[modality]*self.num_query_token, -1)
                    atts_llm[modality] =  torch.ones(inputs_llm[modality].size()[:-1], dtype=torch.long).to(self.device)
                    continue
                # num*bs, num query tokens, llm emb size
                inputs_llm[modality] = getattr(self, f"{modality}_llm_proj")(query_outputs[modality].last_hidden_state[:,:query_tokens[modality].size(1),:]) 
                # bs, num, num query tokens, llm emb size -> bs, num*num query tokens, llm emb size
                inputs_llm[modality] = inputs_llm[modality].reshape(bs, num[modality], self.num_query_token, -1).view(bs,  num[modality]*self.num_query_token, -1)
                atts_llm[modality] =  torch.ones(inputs_llm[modality].size()[:-1], dtype=torch.long).to(self.device)
            else:
                if self.projection_only or getattr(self, f"projection_only_{modality}"):
                    if self.proj_dim == 1:
                        inputs_llm[modality] = getattr(self, f"{modality}_llm_proj")(query_outputs[modality].mean(-1)).reshape(bs, self.num_query_token, -1)
                    else:
                        inputs_llm[modality] = getattr(self, f"{modality}_llm_proj")(query_outputs[modality].reshape(bs, self.num_query_token, -1))
                    atts_llm[modality] =  torch.ones(inputs_llm[modality].size()[:-1], dtype=torch.long).to(self.device)
                    continue
                inputs_llm[modality] = getattr(self, f"{modality}_llm_proj")(query_outputs[modality]['last_hidden_state'][:,:query_tokens[modality].size(1),:])
                atts_llm[modality] = torch.ones(inputs_llm[modality].size()[:-1], dtype=torch.long).to(self.device)
            if self.enumerate_inputs:
                enumeration[modality] = self.llm_tokenizer(
                [f"{'' if i == 0 else ' '}({chr(97+i)}) " for _ in prompt],
                return_tensors="pt",
                add_special_tokens=False if (i!= 0 or self.prefix) else True
                ).to(self.device)

        ## remove trailing whitespace 
        prompt = [p.strip() for p in prompt]

        if 'dialog' in samples:
            llm_tokens = self.llm_tokenizer(
                [f"{d} {p}" if d else p for d, p in zip(samples['dialog'], prompt)],
                padding="longest",
                return_tensors="pt",
                add_special_tokens= not self.clean_tokenization
            ).to(self.device)
        else:
            llm_tokens = self.llm_tokenizer(
                [f"{p}{self.postfix}" for p in prompt] if self.postfix else prompt,
                padding="longest",
                return_tensors="pt",
                add_special_tokens= not self.clean_tokenization
            ).to(self.device)
        bs = llm_tokens.input_ids.shape[0]

        att_list = []
        inp_list = []
        if self.prefix:
            att_list = [self.tokenized_prefix.attention_mask.repeat(bs, 1).to(self.device)]
            inp_list = [self.llm_model.get_input_embeddings()(self.tokenized_prefix.input_ids.to(self.device)).repeat(bs, 1, 1)]            

        if self.joint_video_audio:
            for pos in range(num['video']):
                if self.enumerate_inputs:
                    enumeration_pos = self.llm_tokenizer(
                        [f"{'' if pos == 0 else ' '}({chr(97+pos)}) " for _ in prompt],
                        return_tensors="pt",
                        add_special_tokens=False if (pos!= 0 or self.prefix) else True
                        ).to(self.device)
                    enumeration_inputs_llm = self.llm_model.get_input_embeddings()(enumeration_pos.input_ids)
                    enumeration_atts_llm = enumeration_pos.attention_mask.to(self.device)
                    inp_list.extend([enumeration_inputs_llm])
                    att_list.extend([enumeration_atts_llm])
                if self.use_cues:
                    for modality in ['video', 'audio']:
                        if self.clean_tokenization:
                            if self.prefix or pos > 1 or self.enumerate_inputs or modality == 'audio':
                                att_list.extend([torch.tensor(self.tokenized_cue[modality].attention_mask[:,1:]).to(self.device).repeat(atts_llm[modality].shape[0], 1), atts_llm[modality].view(bs,  num[modality], self.num_query_token)[:, pos, :]])
                                inp_list.extend([self.emb_cue[modality][:,1:].to(self.device).repeat(inputs_llm[modality].shape[0], 1, 1), inputs_llm[modality].view(bs,  num[modality], self.num_query_token, -1)[:, pos, :, :]])
                                continue
                        att_list.extend([torch.tensor(self.tokenized_cue[modality].attention_mask).to(self.device).repeat(atts_llm[modality].shape[0], 1), atts_llm[modality].view(bs,  num[modality], self.num_query_token)[:, pos, :]])
                        inp_list.extend([self.emb_cue[modality].to(self.device).repeat(inputs_llm[modality].shape[0], 1, 1), inputs_llm[modality].view(bs,  num[modality], self.num_query_token, -1)[:, pos, :, :]])
                else:
                    att_list.extend([atts_llm[modality].view(bs, num[modality], self.num_query_token)[:, pos, :]])
                    inp_list.extend([inputs_llm[modality].view(bs, num[modality], self.num_query_token, -1)[:, pos, :, :]])
        else:
            for modality in curr_modalities:
                if self.enumerate_inputs:
                    enumeration_inputs_llm = self.llm_model.get_input_embeddings()(enumeration[modality].input_ids.to(self.device))
                    enumeration_atts_llm = enumeration[modality].attention_mask.to(self.device)
                    inp_list.extend([enumeration_inputs_llm])
                    att_list.extend([enumeration_atts_llm])
                if self.use_cues:
                    if self.clean_tokenization or self.remove_start:
                        if (modality==curr_modalities[0] and not (self.prefix or self.enumerate_inputs)):
                            att_list.extend([torch.tensor(self.tokenized_cue[modality].attention_mask).to(self.device).repeat(atts_llm[modality].shape[0], 1), atts_llm[modality]])
                            inp_list.extend([self.emb_cue[modality].to(self.device).repeat(inputs_llm[modality].shape[0], 1, 1), inputs_llm[modality]])
                        else:
                            att_list.extend([torch.tensor(self.tokenized_cue[modality].attention_mask[:,1:]).to(self.device).repeat(atts_llm[modality].shape[0], 1), atts_llm[modality]])
                            inp_list.extend([self.emb_cue[modality][:,1:].to(self.device).repeat(inputs_llm[modality].shape[0], 1, 1), inputs_llm[modality]])
                    else:
                        att_list.extend([torch.tensor(self.tokenized_cue[modality].attention_mask).to(self.device).repeat(atts_llm[modality].shape[0], 1), atts_llm[modality]])
                        inp_list.extend([self.emb_cue[modality].to(self.device).repeat(inputs_llm[modality].shape[0], 1, 1), inputs_llm[modality]])

                else:
                    att_list.extend([atts_llm[modality]])
                    inp_list.extend([inputs_llm[modality]])

                if self.add_space:
                    space_tok =  self.llm_tokenizer(
                        [f" " for _ in prompt],
                        return_tensors="pt",
                        add_special_tokens=False
                        )
                    space_inputs_llm = self.llm_model.get_input_embeddings()(space_tok.input_ids.to(self.device))
                    space_atts_llm = space_tok.attention_mask.to(self.device)
                    inp_list.extend([space_inputs_llm])
                    att_list.extend([space_atts_llm])

        att_list.append(llm_tokens.attention_mask)
        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)
        inp_list.append(inputs_embeds)
       
        attention_mask = torch.cat(att_list, dim=1)
        inputs_embeds = torch.cat(inp_list, dim=1)

       
        with self.maybe_autocast():
            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
        outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)
        output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [o.strip() for o in output_text]
        return output_text
    
    @torch.no_grad()
    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=-1,
        **kwargs
        ):
        if samples == None or samples == {}:
            return None

        # get batch size
        bs = None
        if 'modalities' in samples:
            curr_modalities = samples['modalities'][0] if isinstance(samples['modalities'][0], list) else  samples['modalities']
        else:
            curr_modalities = [modality for modality in self.modalities if modality in samples]
        for modality in curr_modalities:
            data = samples[modality]
            if isinstance(data, torch.Tensor):
                bs = data.size(0)   
            else:
                bs = len(data)     
            break

        if "text_input" not in samples:
            samples["text_input"] = self.prompt
        if isinstance(samples["text_input"], str):
            samples["text_input"] = [samples["text_input"]] * bs
        text_input = samples['text_input']

        if not prompt and self.prompt:
            prompt=self.prompt
        if prompt:
            if prompt.count("{}") == 2:
                if 'ocr_tokens' in samples:
                    text_input = [
                        prompt.format(', '.join(samples['ocr_tokens'][i][:30]), samples["text_input"][i])
                    for i in range(len(samples["text_input"]))]
                elif 'choices' in samples:
                    text_input = []
                    for i in range(len(samples["text_input"])):
                        this_choices = [f"({string.ascii_lowercase[j]}) {ch}" for j, ch in enumerate(samples["choices"][i])]
                        this_choices = " ".join(this_choices)
                        text_input.append(prompt.format(samples["text_input"][i], this_choices))
            else:
                text_input = [prompt.format(question) for question in samples["text_input"]]
            samples["prompt"] = text_input

        if 'discrn' in samples and self.use_caption: ## discriminatory reasoning
            self.llm_tokenizer.padding_side = "left"

            text_input = samples['text_input'] if 'prompt' not in samples else samples['prompt']
            if self.postfix:
                text_input = [f'{t}{self.postfix}' for t in text_input]
            if self.enumerate_inputs:
                prompt = [f'{self.prefix}(a){Blip2VicunaXInstruct.MODALITY_TO_CUE[samples["modalities"][i][0]] if self.use_cues else " "}{samples["baseline_captions"][i][0]} (b){Blip2VicunaXInstruct.MODALITY_TO_CUE[samples["modalities"][i][1]] if self.use_cues else " "}{samples["baseline_captions"][i][1]} {text_input[i]}' for i in range(bs)]
            else:
                prompt = [f'{self.prefix}{Blip2VicunaXInstruct.MODALITY_TO_CUE[samples["modalities"][i][0]]}{samples["baseline_captions"][i][0] if self.use_cues else " "}{Blip2VicunaXInstruct.MODALITY_TO_CUE[samples["modalities"][i][1]] if self.use_cues else " "}{samples["baseline_captions"][i][1]} {text_input[i]}' for i in range(bs)]
            llm_tokens = self.llm_tokenizer(
                prompt,
                padding="longest",
                return_tensors="pt"
            ).to(self.device)

            with self.maybe_autocast():
                outputs = self.llm_model.generate(
                    inputs_embeds=self.llm_model.get_input_embeddings()(llm_tokens.input_ids),
                    attention_mask=llm_tokens.attention_mask,
                    do_sample=False,
                    num_beams=num_beams,
                    max_length=max_len,
                    min_length=min_len,
                    repetition_penalty=1.5,
                    # eos_token_id=self.eos_token_id,
                    length_penalty=length_penalty,
                )
            outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)
            output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            return output_text

        output_text = self.generate(
            samples,
            num_beams=num_beams,
            max_length=max_len,
            min_length=min_len,
            length_penalty=length_penalty
        )

        if "apply_lemmatizer" in samples.keys() and samples["apply_lemmatizer"]:
            output_text = self._lemmatize(output_text)
        
        #vizwiz
        output_text = [o if o != "" else "unanswerable" for o in output_text]

        return output_text

    def predict(
        self,
        samples,
        candidates=None,
        n_segments=1,
        max_length=10,
        min_length=1,
        length_penalty=-1.,
        special_qformer_input_prompt=False
        ):

        self.llm_tokenizer.padding_side = "left"

        if candidates == None:
            candidates = self.candidates
        else:
            self.candidates = candidates # for the output targets.
        
        if self.predict_with_gen:
            output = self.generate(samples,max_length=max_length,min_length=min_length,length_penalty=length_penalty)
            result = []
            for text in output:
                text = BlipCaptionProcessor().pre_caption(text)
                pred_label = ""  # default to an empty string
                for cand in candidates:
                    cand = BlipCaptionProcessor().pre_caption(cand)
                    if cand in text.split(" "):
                        pred_label = cand
                        break  # stop as soon as we find a match
                result.append(pred_label)
            return {"predictions":result, "target": samples["label"]}


        # If candidates is a list of lists, each sample has its candidates, then we need to iterate one by one
        if type(candidates[0]) == list:
            results = []

            for i in range(samples["image"].size(0)):
                this_sample = {
                    "image": samples["image"][i].unsqueeze(0),
                    "prompt": samples["prompt"],
                }

                if "text_input" in samples.keys():
                    this_sample["text_input"] = [samples["text_input"][i]]

                if 'context' in samples.keys():
                    this_sample['context'] = [samples["context"][i]]

                if 'history' in samples.keys():
                    this_sample['history'] = [samples["history"][i]]

                if 'caption' in samples.keys():
                    this_sample['caption'] = [samples["caption"][i]]

                this_result = self._predict_class(this_sample, candidates[i], n_segments, special_qformer_input_prompt)
                results.append(this_result)

            try:
                results = torch.cat(results, dim=0)
            except:
                results = [res.tolist()[0] for res in results]

            return results

        return self._predict_class(samples, candidates, n_segments, special_qformer_input_prompt)

    def _predict_class(
        self,
        samples,
        candidates,
        n_segments=1,
        special_qformer_input_prompt=False,
        ):
        if list(samples.keys()) == []:
            return None
    
        if "prompt" in samples:
            prompt = samples["prompt"]
        else:
            prompt = self.prompt
        
        candidates = [self.format_candidates_prompt.format(c) for c in candidates]

        if 'modalities' in samples:
            curr_modalities = samples['modalities'][0] if isinstance(samples['modalities'][0], list) else  samples['modalities']
        else:
            curr_modalities = [modality for modality in self.modalities if modality in samples]
        
        # get batch size
        for modality in curr_modalities:
            data = samples[modality]
            if isinstance(data, torch.Tensor):
                bs = data.size(0)
            else:
                bs = len(data)
            break

        if isinstance(prompt, str):
            prompt = [prompt] * bs
        else:
            assert len(prompt) == bs, "The number of prompts must be equal to the batch size."

        if "text_input" in samples.keys():
            if type(samples["text_input"][0]) == list:
                prompt = [prompt[i].format(*samples["text_input"][i]) for i in range(len(prompt))]
            else:
                prompt = [prompt[i].format(samples["text_input"][i]) for i in range(len(prompt))]

        # scienceqa
        if 'context' in samples.keys() and samples['context'] != '':
            prompt = [f'context: {samples["context"][i]}. {prompt[i]}' for i in range(len(prompt))]

        # visual dialog
        if 'history' in samples.keys() and samples['history'][0] != '':
            prompt = [f'dialog history: {samples["history"][i]}\n{prompt[i]}' for i in range(len(prompt))]

        if 'caption' in samples.keys() and samples['caption'][0] != '':
            prompt = [f'This image has the caption "{samples["caption"][i]}". {prompt[i]}' for i in range(len(prompt))]


                
        if 'discrn' in samples and self.use_caption: ## discriminatory reasoning
            if self.postfix:
                prompt = [f'{p}{self.postfix}' for p in prompt]
            if self.enumerate_inputs:
                prompt = [f'{self.prefix}(a){Blip2VicunaXInstruct.MODALITY_TO_CUE[samples["modalities"][i][0]] if self.use_cues else " "}{samples["baseline_captions"][i][0]} (b){Blip2VicunaXInstruct.MODALITY_TO_CUE[samples["modalities"][i][1]] if self.use_cues else " "}{samples["baseline_captions"][i][1]} {prompt[i]}' for i in range(bs)]
            else:
                prompt = [f'{self.prefix}{Blip2VicunaXInstruct.MODALITY_TO_CUE[samples["modalities"][i][0]]}{samples["baseline_captions"][i][0] if self.use_cues else " "}{Blip2VicunaXInstruct.MODALITY_TO_CUE[samples["modalities"][i][1]] if self.use_cues else " "}{samples["baseline_captions"][i][1]} {prompt[i]}' for i in range(bs)]
            text_input_tokens = self.llm_tokenizer(
                prompt,
                padding="longest",
                return_tensors="pt"
            ).to(self.device)
        else:
            if not self.projection_only and not getattr(self, f"projection_only_{modality}"):
                query_tokens = {}
                for modality in self.modalities:
                    if modality not in samples:
                            continue
                    query_tokens[modality] = getattr(self, f"{modality}_query_tokens").expand(bs, -1, -1)
            
            if self.qformer_text_input:
                if self.special_qformer_input_prompt or special_qformer_input_prompt:
                    
                    qformer_prompt = special_qformer_input_prompt if special_qformer_input_prompt else self.special_qformer_input_prompt
                    qformer_prompt = [qformer_prompt] * len(prompt)
                    if "text_input" in samples.keys():
                        if type(samples["text_input"][0]) == list:
                            qformer_prompt = [qformer_prompt[i].format(*samples["text_input"][i]) for i in range(len(qformer_prompt))]
                        else:
                            qformer_prompt = [qformer_prompt[i].format(samples["text_input"][i]) for i in range(len(qformer_prompt))]

                    text_Qformer = self.tokenizer(
                        qformer_prompt,
                        padding='longest',
                        truncation=True,
                        max_length=self.max_txt_len,
                        return_tensors="pt",
                    ).to(self.device)
                elif self.use_describe:
                    modality2prompt = {
                    "video": "a short description of the video",
                    "audio": "an audio that shows",
                    "image": "a short image caption",
                    "pc": "a 3d model of"
                    }
                    qformer_prompt = [modality2prompt[modality] for _ in samples['text_input']]

                    # qformer_prompt = [f'Describe the {Blip2VicunaXInstruct.MODALITY_TO_CUE[modality].replace(":", "").strip() if modality != "pc" else "3d model"}.' for _ in samples["text_input"]]
                    text_Qformer = self.tokenizer(
                        qformer_prompt,
                        padding='longest',
                        truncation=True,
                        max_length=self.max_txt_len,
                        return_tensors="pt",
                    ).to(self.device)
                else:
                    text_Qformer = self.tokenizer(
                        prompt,
                        padding='longest',
                        truncation=True,
                        max_length=self.max_txt_len,
                        return_tensors="pt",
                    ).to(self.device)
                
                Qformer_atts = {}
                query_atts = {}
                
                for modality in curr_modalities:
                    # B, Token Size
                    query_atts[modality] = torch.ones(query_tokens[modality].size()[:-1], dtype=torch.long).to(self.device)
                    # B, Token Size + Inp Size
                    Qformer_atts[modality] = torch.cat([query_atts[modality],text_Qformer.attention_mask],dim=1)
                
            embeds = {}
            data_atts = {}
            for modality in curr_modalities:
                data = samples[modality]
                ln = getattr(self, f"{modality}_ln")
                encoder = getattr(self, f"{modality}_encoder")
                if modality == "video" and "clip" in self.video_enc_name:
                    embeds[modality] = []
                    data_atts[modality] = []
                    for j in range(data.size(2)):
                        this_frame = data[:,:,j,:,:]
                        with self.maybe_autocast():
                            embeds[modality].append(ln(encoder(this_frame)))
                            if self.shared_qformer:
                                embeds[modality][j] = getattr(self, f"{modality}_encoder_projection")(embeds[modality][j])
                            data_atts[modality].append(torch.ones(embeds[modality][j].size()[:-1], dtype=torch.long).to(self.device))

                elif modality == 'audio' and 'beats' in self.audio_enc_name:
                    embeds[modality] = []
                    data_atts[modality] = []
                    for j in range(data.size(1)):
                        this_frame = data[:,j,:,:]
                        with self.maybe_autocast():
                            embeds[modality].append(ln(encoder(this_frame)))
                            if self.shared_qformer:
                                embeds[modality][j] = getattr(self, f"{modality}_encoder_projection")(embeds[modality][j])
                        data_atts[modality].append(torch.ones(embeds[modality][j].size()[:-1], dtype=torch.long).to(self.device))
                else:
                    with self.maybe_autocast():
                        embeds[modality] = ln(encoder(data))
                    if len(embeds[modality].size()) == 2:
                        # B, C, D
                        embeds[modality] = embeds[modality].unsqueeze(1)
                    # B, C
                    if self.shared_qformer:
                        embeds[modality] = getattr(self, f"{modality}_encoder_projection")(embeds[modality])
                    data_atts[modality] = torch.ones(embeds[modality].size()[:-1], dtype=torch.long).to(self.device)
                
            query_outputs = {}
            num = {}
            if self.qformer_text_input:
                for modality in curr_modalities:
                    if modality in Blip2VicunaXInstruct.SEQUENCIAL_MODALITIES and getattr(self, f'{modality}_enc_name') in Blip2VicunaXInstruct.SEQUENCIAL_ENCODERS:
                        num[modality] = len(embeds[modality])
                        bs = embeds[modality][0].shape[0]
                        indices = [j_+r for r,j in enumerate([[i*bs for i in range(num[modality])]]*bs) for j_ in j]
                        reordered_embeds = torch.cat(embeds[modality])[indices]
                        reordered_atts = torch.cat(data_atts[modality])[indices]
                        if self.projection_only or getattr(self, f"projection_only_{modality}"):
                            if self.proj_dim != 1:
                                    query_outputs[modality] = getattr(self, f"{modality}_projection")(reordered_embeds.mean(1,keepdim=True)).view(bs*num[modality], self.num_query_token, -1)
                            else:
                                query_outputs[modality] = getattr(self, f"{modality}_projection")(reordered_embeds.view(reordered_embeds.shape[0],-1))
                            continue
                        query_output = getattr(self, f"{modality}_Qformer").bert(
                            text_Qformer.input_ids.repeat(num[modality], 1),
                            attention_mask=Qformer_atts[modality].repeat(num[modality], 1),
                            query_embeds=query_tokens[modality].repeat(num[modality], 1, 1),
                            encoder_hidden_states=reordered_embeds,
                            encoder_attention_mask=reordered_atts,
                            return_dict=True,
                        )
                        query_outputs[modality] = query_output
                    else:
                        bs = embeds[modality].shape[0]
                        if self.projection_only or getattr(self, f"projection_only_{modality}"):
                            if self.proj_dim != 1:
                                query_outputs[modality] = getattr(self, f"{modality}_projection")(embeds[modality].mean(1, keepdim=True)).reshape(bs, self.num_query_token,-1)
                            else:
                                query_outputs[modality] = getattr(self, f"{modality}_projection")(embeds[modality]).reshape(bs, self.num_query_token,-1)
                            continue  
                        query_outputs[modality] = getattr(self, f"{modality}_Qformer").bert(
                            text_Qformer.input_ids,
                            attention_mask=Qformer_atts[modality],
                            query_embeds=query_tokens[modality],
                            encoder_hidden_states=embeds[modality].to(torch.float32),
                            encoder_attention_mask=data_atts[modality],
                            return_dict=True,
                        )
            else:
                for modality in curr_modalities:
                    if modality in Blip2VicunaXInstruct.SEQUENCIAL_MODALITIES and getattr(self, f'{modality}_enc_name') in Blip2VicunaXInstruct.SEQUENCIAL_ENCODERS:
                        num[modality] = len(embeds[modality])
                        bs = embeds[modality][0].shape[0]
                        indices = [j_+r for r,j in enumerate([[i*bs for i in range(num[modality])]]*bs) for j_ in j]
                        reordered_embeds = torch.cat(embeds[modality])[indices]
                        reordered_atts = torch.cat(data_atts[modality])[indices]
                        if self.projection_only or getattr(self, f"projection_only_{modality}"):
                            if self.proj_dim != 1:
                                query_outputs[modality] = getattr(self, f"{modality}_projection")(reordered_embeds.mean(1,keepdim=True)).view(bs*num[modality], self.num_query_token, -1)
                            else:
                                query_outputs[modality] = getattr(self, f"{modality}_projection")(reordered_embeds.view(reordered_embeds.shape[0],-1))
                            continue
                        query_output = getattr(self, f"{modality}_Qformer").bert(
                            query_embeds=query_tokens[modality].repeat(num[modality], 1, 1),
                            encoder_hidden_states=reordered_embeds,
                            encoder_attention_mask=reordered_atts,
                            return_dict=True,
                        )
                        query_outputs[modality] = query_output
                    else:
                        bs = embeds[modality].shape[0]
                        if self.projection_only or getattr(self, f"projection_only_{modality}"):
                            if self.proj_dim != 1:
                                query_outputs[modality] = getattr(self, f"{modality}_projection")(embeds[modality].mean(1, keepdim=True)).reshape(bs, self.num_query_token,-1)
                            else:
                                query_outputs[modality] = getattr(self, f"{modality}_projection")(embeds[modality]).reshape(bs, self.num_query_token,-1)
                            continue  
                        query_outputs[modality] = getattr(self, f"{modality}_Qformer").bert(
                            query_embeds=query_tokens[modality],
                            encoder_hidden_states=embeds[modality].to(torch.float32),
                            encoder_attention_mask=data_atts[modality],
                            return_dict=True,
                        )
            
            inputs_llm = {}
            atts_llm = {}
            enumeration = {}
            # from pdb import set_trace; set_trace()
            for i,modality in enumerate(curr_modalities):
                if modality in Blip2VicunaXInstruct.SEQUENCIAL_MODALITIES and getattr(self, f'{modality}_enc_name') in Blip2VicunaXInstruct.SEQUENCIAL_ENCODERS:
                    if self.projection_only or getattr(self, f"projection_only_{modality}"):
                        if self.proj_dim != 1:
                            inputs_llm[modality] = getattr(self, f"{modality}_llm_proj")(query_outputs[modality].unsqueeze(1)).reshape(bs*num[modality], self.num_query_token, -1)
                        else:
                            inputs_llm[modality] = getattr(self, f"{modality}_llm_proj")(query_outputs[modality].reshape(bs*num, self.num_query_token, -1))
                        inputs_llm[modality] = inputs_llm[modality].reshape(bs, num[modality], self.num_query_token, -1).view(bs, num[modality]*self.num_query_token, -1)
                        atts_llm[modality] =  torch.ones(inputs_llm[modality].size()[:-1], dtype=torch.long).to(self.device)
                        continue
                    # num*bs, num query tokens, llm emb size
                    inputs_llm[modality] = getattr(self, f"{modality}_llm_proj")(query_outputs[modality].last_hidden_state[:,:query_tokens[modality].size(1),:]) 
                    # bs, num, num query tokens, llm emb size -> bs, num*num query tokens, llm emb size
                    inputs_llm[modality] = inputs_llm[modality].reshape(bs, num[modality], self.num_query_token, -1).view(bs, num[modality]*self.num_query_token, -1)
                    atts_llm[modality] =  torch.ones(inputs_llm[modality].size()[:-1], dtype=torch.long).to(self.device)
                    
                else:
                    if self.projection_only or getattr(self, f"projection_only_{modality}"):
                        if self.proj_dim == 1:
                            inputs_llm[modality] = getattr(self, f"{modality}_llm_proj")(query_outputs[modality].mean(-1)).reshape(bs, self.num_query_token, -1)
                        else:
                            inputs_llm[modality] = getattr(self, f"{modality}_llm_proj")(query_outputs[modality].reshape(bs, self.num_query_token, -1))
                        atts_llm[modality] =  torch.ones(inputs_llm[modality].size()[:-1], dtype=torch.long).to(self.device)
                        continue
                    inputs_llm[modality] = getattr(self, f"{modality}_llm_proj")(query_outputs[modality]['last_hidden_state'][:,:query_tokens[modality].size(1),:])
                    atts_llm[modality] = torch.ones(inputs_llm[modality].size()[:-1], dtype=torch.long).to(self.device)
                if self.enumerate_inputs:
                    enumeration[modality] = self.llm_tokenizer(
                        [f"{'' if i == 0 else ' '}({chr(97+i)}) " for _ in prompt],
                        return_tensors="pt",
                        add_special_tokens=False if (i!= 0 or self.prefix) else True
                        ).to(self.device)
                    
            att_list = []
            inp_list = []
            if self.prefix:
                att_list = [self.tokenized_prefix.attention_mask.repeat(bs, 1).to(self.device)]
                inp_list = [self.llm_model.get_input_embeddings()(self.tokenized_prefix.input_ids.to(self.device)).repeat(bs, 1, 1)]            
        
            for modality in curr_modalities:
                if self.enumerate_inputs:
                    enumeration_inputs_llm = self.llm_model.get_input_embeddings()(enumeration[modality].input_ids.to(self.device))
                    enumeration_atts_llm = enumeration[modality].attention_mask.to(self.device)
                    inp_list.extend([enumeration_inputs_llm])
                    att_list.extend([enumeration_atts_llm])
                if self.use_cues:
                    if self.clean_tokenization or self.remove_start:
                        if (modality==curr_modalities[0] and not (self.prefix or self.enumerate_inputs)):
                            att_list.extend([torch.tensor(self.tokenized_cue[modality].attention_mask).to(self.device).repeat(atts_llm[modality].shape[0], 1), atts_llm[modality]])
                            inp_list.extend([self.emb_cue[modality].to(self.device).repeat(inputs_llm[modality].shape[0], 1, 1), inputs_llm[modality]])
                        else:
                            att_list.extend([torch.tensor(self.tokenized_cue[modality].attention_mask[:,1:]).to(self.device).repeat(atts_llm[modality].shape[0], 1), atts_llm[modality]])
                            inp_list.extend([self.emb_cue[modality][:,1:].to(self.device).repeat(inputs_llm[modality].shape[0], 1, 1), inputs_llm[modality]])
                    else:
                        att_list.extend([torch.tensor(self.tokenized_cue[modality].attention_mask).to(self.device).repeat(atts_llm[modality].shape[0], 1), atts_llm[modality]])
                        inp_list.extend([self.emb_cue[modality].to(self.device).repeat(inputs_llm[modality].shape[0], 1, 1), inputs_llm[modality]])

                else:
                    att_list.extend([atts_llm[modality]])
                    inp_list.extend([inputs_llm[modality]])

                if self.add_space:
                    space_tok =  self.llm_tokenizer(
                        [f" " for _ in prompt],
                        return_tensors="pt",
                        add_special_tokens=False
                        )
                    space_inputs_llm = self.llm_model.get_input_embeddings()(space_tok.input_ids.to(self.device))
                    space_atts_llm = space_tok.attention_mask.to(self.device)
                    inp_list.extend([space_inputs_llm])
                    att_list.extend([space_atts_llm])



            atts_llm = torch.cat(att_list, dim=1)
            empty_targets = torch.ones(atts_llm.size(), dtype=torch.long).to(self.device).fill_(-100)
            inputs_llm = torch.cat(inp_list, dim=1)


            self.llm_tokenizer.padding_side = "right"
            self.llm_tokenizer.truncation_side = 'left'


            text_input_tokens = self.llm_tokenizer(
                [f"{p}{self.postfix}" for p in prompt] if self.postfix else prompt,
                padding="longest",
                return_tensors="pt",
                add_special_tokens= not self.clean_tokenization
            ).to(self.device)

        self.llm_tokenizer.truncation_side = 'right'
        n_cands = len(candidates)
        with self.maybe_autocast():
            all_losses = []
            for n in range(n_segments):
                seg_len = n_cands // n_segments
                if n == (n_segments - 1):
                    seg_len = n_cands - seg_len * (n_segments - 1)

                start_i = n * (n_cands // n_segments)
                end_i = start_i + seg_len
                this_output_tokens = self.llm_tokenizer(
                    candidates[start_i:end_i],
                    return_tensors="pt",
                    padding="longest",
                    # truncation=True,
                    # max_length=self.max_output_txt_len,
                ).to(self.device)

                this_input_tokens_ids = text_input_tokens.input_ids.repeat_interleave(seg_len, dim=0)
                this_input_tokens_atts = text_input_tokens.attention_mask.repeat_interleave(seg_len, dim=0)

                this_output_tokens_ids = this_output_tokens.input_ids.repeat(bs, 1)
                this_output_tokens_atts = this_output_tokens.attention_mask.repeat(bs, 1)

                this_llm_tokens, this_input_targets_len = self.concat_text_input_output(
                    this_input_tokens_ids,
                    this_input_tokens_atts,
                    this_output_tokens_ids,
                    this_output_tokens_atts
                )

                this_llm_input_ids = this_llm_tokens['input_ids']
                this_llm_atts = this_llm_tokens['attention_mask']

                inputs_embeds = self.llm_model.get_input_embeddings()(this_llm_input_ids)

                if self.use_caption:
                    inputs_embeds = torch.cat([inputs_embeds], dim=1)
                    attention_mask = torch.cat([this_llm_atts], dim=1)
                else:
                    inputs_embeds = torch.cat([inputs_llm.repeat_interleave(seg_len, dim=0), inputs_embeds], dim=1)
                    attention_mask = torch.cat([atts_llm.repeat_interleave(seg_len, dim=0), this_llm_atts], dim=1)


                this_targets = this_llm_input_ids.masked_fill(this_llm_input_ids == self.llm_tokenizer.pad_token_id, -100)
        
                for i, l in enumerate(this_input_targets_len):
                    this_targets[i][:l] = -100
                
                if self.use_caption:
                    torch.cat([this_targets], dim=1)
                else:
                    this_targets = torch.cat([empty_targets.repeat_interleave(seg_len, dim=0), this_targets], dim=1)


                outputs = self.llm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=this_targets,
                    reduction="none",
                )

                loss = outputs.loss

                loss = loss.reshape(bs, seg_len)
                all_losses.append(loss)

        all_losses = torch.cat(all_losses, dim=-1)
        all_losses = -all_losses
        output_class_ranks = torch.argsort(all_losses, dim=-1)
        return {"predictions": all_losses, "targets": torch.tensor([self.candidates.index(l) for l in samples["label"]])}

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer
    
    def get_optimizer_params(self, weight_decay, lr_scale=1):
        return BaseModel.get_optimizer_params(self, weight_decay, lr_scale=lr_scale)

    @classmethod
    def from_config(cls, cfg):
        image_model = cfg.get("image_model","eva_clip_g")
        pc_model = cfg.get("pc_model","ulip2_pointbert")
        video_model = cfg.get("video_model","eva_clip_g")
        audio_model = cfg.get("audio_model","beats")

        pretrained_image_qformer = cfg.get("pretrained_image_qformer",None)
        pretrained_pc_qformer = cfg.get("pretrained_pc_qformer",None)
        pretrained_video_qformer = cfg.get("pretrained_video_qformer",None)
        pretrained_audio_qformer = cfg.get("pretrained_audio_qformer",None)

        load_attention_image_qformer = cfg.get("load_attention_image_qformer",False)
        load_attention_pc_qformer = cfg.get("load_attention_pc_qformer",False)
        load_attention_video_qformer = cfg.get("load_attention_video_qformer",False)
        load_attention_audio_qformer = cfg.get("load_attention_audio_qformer",False)
 
        load_qformer_type_image=cfg.get('load_qformer_type_image', "")
        load_qformer_type_pc=cfg.get('load_qformer_type_pc', "")
        load_qformer_type_video=cfg.get('load_qformer_type_video', "")
        load_qformer_type_audio=cfg.get('load_qformer_type_audio',"")

        load_projection_image=cfg.get('load_projection_image', True)
        load_projection_pc=cfg.get('load_projection_pc', True)
        load_projection_video=cfg.get('load_projection_video', True)
        load_projection_audio=cfg.get('load_projection_audio', True)

        load_projection_type_image=cfg.get('load_projection_type_image', "")
        load_projection_type_pc=cfg.get('load_projection_type_pc', "")
        load_projection_type_video=cfg.get('load_projection_type_video', "")
        load_projection_type_audio=cfg.get('load_projection_type_audio', "")

        load_ln_type_image=cfg.get('load_ln_type_image', "")
        load_ln_type_pc=cfg.get('load_ln_type_pc', "")
        load_ln_type_video=cfg.get('load_ln_type_video', "")
        load_ln_type_audio=cfg.get('load_ln_type_audio', "")

        image_encoder_kwargs = cfg.get("image_encoder_kwargs", {"image_size": 224, "drop_path_rate": 0, "use_grad_checkpoint": False})
        pc_encoder_kwargs = cfg.get("pc_encoder_kwargs",{})
        video_encoder_kwargs = cfg.get("video_encoder_kwargs",{})
        audio_encoder_kwargs = cfg.get("audio_encoder_kwargs",{})

        image_precision = cfg.get("image_precision","fp16")
        pc_precision = cfg.get("pc_precision","fp16")
        video_precision = cfg.get("video_precision","fp16")
        audio_precision = cfg.get("audio_precision","fp16")
  
        freeze_image = cfg.get("freeze_image",True)
        freeze_pc = cfg.get("freeze_pc",True)
        freeze_video = cfg.get("freeze_video",True)
        freeze_audio = cfg.get("freeze_audio",True)
        num_query_token = cfg.get("num_query_token")
       
        llm_model = cfg.get("llm_model")
        freeze_pc = cfg.get("freeze_pc", True)
        freeze_video = cfg.get("freeze_video", True)
        freeze_audio = cfg.get("freeze_audio", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 128)
        max_output_txt_len = cfg.get("max_output_txt_len", 256)

        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        qformer_text_input = cfg.get("qformer_text_input", True)
        modalities = cfg.get("modalities", ["image"])
        use_cues = cfg.get("use_cues", True)
        shared_qformer = cfg.get("shared_qformer",False)
        pretrained_shared_qformer = cfg.get("pretrained_shared_qformer", None)
        load_attention_shared_qformer = cfg.get("load_attention_shared_qformer", None)
        load_qformer_type_shared= cfg.get('load_qformer_type_shared',"")
        load_projection_shared= cfg.get('load_projection_shared',False)
        load_projection_type_shared= cfg.get('load_projection_type_shared',"")
        shared_qformer_num_features=cfg.get("shared_qformer_num_features", 512)
        encoder_projection_type_image=cfg.get("encoder_projection_type_image","")
        encoder_projection_type_video=cfg.get("encoder_projection_type_video","")
        encoder_projection_type_audio=cfg.get("encoder_projection_type_audio","")
        encoder_projection_type_pc=cfg.get("encoder_projection_type_pc","")

        llm_text_input = cfg.get("llm_text_input", True)
        lora = cfg.get("lora", False)
        prefix = cfg.get("prefix", "")
        postfix = cfg.get("postfix", "")

        cached_audio= cfg.get("cached_audio", False)
        cached_image= cfg.get("cached_image", False)
        cached_video= cfg.get("cached_video", False)
        cached_pc= cfg.get("cached_pc", False)

        num_features_audio=cfg.get('num_features_audio', 768)
        num_features_image=cfg.get('num_features_image', 1408)
        num_features_video=cfg.get('num_features_video', 14080)
        num_features_pc=cfg.get('num_features_depth', 512)

        joint_video_audio=cfg.get('joint_video_audio', False)
        use_caption=cfg.get('use_caption', False)
        use_describe=cfg.get('use_describe', False)
        predict_with_gen = cfg.get('predict_with_gen', False)
        format_candidates_prompt = cfg.get('format_candidates_prompt', "{}")
        special_qformer_input_prompt = cfg.get('special_qformer_input_prompt', False)
        enumerate_inputs = cfg.get('enumerate_inputs', False)
        add_space = cfg.get('add_space', True)
        projection_only = cfg.get('projection_only', False)

        lora_model = cfg.get('lora_model', '')

        projection_only_audio= cfg.get('projection_only_audio', False)
        projection_only_pc=  cfg.get('projection_only_pc', False)
        projection_only_video=  cfg.get('projection_only_video', False)
        projection_only_image=  cfg.get('projection_only_image', False)

        projection_path_audio=cfg.get('projection_path_audio', False)
        projection_path_pc=cfg.get('projection_path_pc', False)
        projection_path_video=cfg.get('projection_path_video', False)
        projection_path_image=cfg.get('projection_path_image', False)
        remove_start=cfg.get('remove_start', False)
        proj_dim=cfg.get('proj_dim', 1)
        clean_tokenization=cfg.get('clean_tokenization', False)

        logging.info("Model Config Arguments:")
        logging.info(OmegaConf.to_yaml(cfg))

        model = cls(
            image_model=image_model,
            pc_model=pc_model,
            video_model=video_model,
            audio_model=audio_model,

            pretrained_image_qformer=pretrained_image_qformer,
            pretrained_pc_qformer=pretrained_pc_qformer,
            pretrained_video_qformer=pretrained_video_qformer,
            pretrained_audio_qformer=pretrained_audio_qformer,

            load_attention_image_qformer=load_attention_image_qformer,
            load_attention_pc_qformer=load_attention_pc_qformer,
            load_attention_video_qformer=load_attention_video_qformer,
            load_attention_audio_qformer=load_attention_audio_qformer,
 
            load_qformer_type_image=load_qformer_type_image,
            load_qformer_type_pc=load_qformer_type_pc,
            load_qformer_type_video=load_qformer_type_video,
            load_qformer_type_audio=load_qformer_type_audio,
   
            load_projection_image=load_projection_image,
            load_projection_pc=load_projection_pc,
            load_projection_video=load_projection_video,
            load_projection_audio=load_projection_audio,

            load_projection_type_image=load_projection_type_image,
            load_projection_type_pc=load_projection_type_pc,
            load_projection_type_video=load_projection_type_video,
            load_projection_type_audio=load_projection_type_audio,

            load_ln_type_image=load_ln_type_image,
            load_ln_type_pc=load_ln_type_pc,
            load_ln_type_video=load_ln_type_video,
            load_ln_type_audio=load_ln_type_audio,
  
            image_encoder_kwargs = image_encoder_kwargs,
            pc_encoder_kwargs = pc_encoder_kwargs,
            video_encoder_kwargs = video_encoder_kwargs,
            audio_encoder_kwargs = audio_encoder_kwargs,

            image_precision=image_precision,
            pc_precision=pc_precision,
            video_precision=video_precision,
            audio_precision=audio_precision,

            freeze_image=freeze_image,
            freeze_pc=freeze_pc,
            freeze_video=freeze_video,
            freeze_audio=freeze_audio,

            num_query_token=num_query_token,
            llm_model=llm_model,
            lora_model=lora_model,
            lora = lora,
            prompt=prompt,
            max_txt_len=max_txt_len,
            max_output_txt_len=max_output_txt_len,
            apply_lemmatizer=apply_lemmatizer,
            qformer_text_input=qformer_text_input,
            modalities=modalities,
            use_cues=use_cues,
            llm_text_input=llm_text_input,
            shared_qformer=shared_qformer,
            pretrained_shared_qformer = pretrained_shared_qformer,
            load_attention_shared_qformer = load_attention_shared_qformer,
            shared_qformer_num_features=shared_qformer_num_features,
            load_qformer_type_shared= load_qformer_type_shared,
            load_projection_shared= load_projection_shared,

            encoder_projection_type_image=encoder_projection_type_image, 
            encoder_projection_type_video=encoder_projection_type_video, 
            encoder_projection_type_audio=encoder_projection_type_audio, 
            encoder_projection_type_pc=encoder_projection_type_pc,

            projection_path_audio=projection_path_audio,
            projection_path_pc=projection_path_pc,
            projection_path_video=projection_path_video,
            projection_path_image=projection_path_image,

            load_projection_type_shared= load_projection_type_shared,

            prefix=prefix,
            postfix=postfix,

            cached_audio=cached_audio,
            cached_image=cached_image,
            cached_video=cached_video,
            cached_pc=cached_pc,

            num_features_audio=num_features_audio,
            num_features_image=num_features_image,
            num_features_video=num_features_video,
            num_features_pc=num_features_pc,

            joint_video_audio=joint_video_audio,
            use_caption=use_caption,
            use_describe=use_describe,
            predict_with_gen=predict_with_gen,
            format_candidates_prompt=format_candidates_prompt,
            special_qformer_input_prompt=special_qformer_input_prompt,
            enumerate_inputs=enumerate_inputs,
            add_space=add_space,
            projection_only=projection_only,

            projection_only_audio= projection_only_audio,
            projection_only_pc=  projection_only_pc,
            projection_only_video= projection_only_video,
            projection_only_image=  projection_only_image,
            remove_start= remove_start,
            proj_dim=proj_dim,
            clean_tokenization=clean_tokenization
        )

        stage1_url_or_filename = cfg.get("stage1_url_or_filename","")

        if stage1_url_or_filename:
            model.load_from_pretrained(stage1_url_or_filename)

        model.load_checkpoint_from_config(cfg)
        return model
    
    @classmethod
    def init_ln(cls, num_features, load_ln_path=False, load_ln_type=""):
        ln = LayerNorm(num_features)
        if load_ln_path and load_ln_type:
            url_or_filename=load_ln_path
            logging.info(f"Loading pretrained layer norm weights from {url_or_filename} of type {load_ln_type}")
            if is_url(url_or_filename):
                cached_file = download_cached_file(
                    url_or_filename, check_hash=False, progress=True
                )
                checkpoint = torch.load(cached_file, map_location="cpu")
            elif os.path.isfile(url_or_filename):
                checkpoint = torch.load(url_or_filename, map_location="cpu")
            else:
                raise RuntimeError("checkpoint url or path is invalid")
            
            if load_ln_type:
                load_ln_type = f"{load_ln_type}_ln" if "vision" not in load_ln_type else "ln_vision"
            loaded_state_dict = {}
            if 'model' in checkpoint:
                checkpoint = checkpoint['model'] 
            for k in checkpoint.keys():
                if load_ln_type in k:
                    loaded_state_dict['.'.join(k.split('.')[1:])] = checkpoint[k]
            ln.load_state_dict(loaded_state_dict, strict=False)
        
        return ln
    
    @classmethod
    def init_encoder_projection(cls, enc_num_features, shared_qformer_num_features, load_proj_path=False, load_proj_type=""):
        encoder_projection = nn.Linear(enc_num_features, shared_qformer_num_features)
        if load_proj_path and load_proj_type:
            url_or_filename=load_proj_path
            logging.info(f"Loading shared Qformer encoder projection weights from {url_or_filename} of type {load_proj_type}")
            if is_url(url_or_filename):
                cached_file = download_cached_file(
                    url_or_filename, check_hash=False, progress=True
                )
                checkpoint = torch.load(cached_file, map_location="cpu")
            elif os.path.isfile(url_or_filename):
                checkpoint = torch.load(url_or_filename, map_location="cpu")
            else:
                raise RuntimeError("checkpoint url or path is invalid")
            
            if load_proj_type:
                load_proj_type = f"{load_proj_type}_"
            loaded_state_dict = {}
            if 'model' in checkpoint:
                checkpoint = checkpoint['model'] 
            for k in checkpoint.keys():
                if load_proj_type+'encoder_projection' in k:
                    loaded_state_dict['.'.join(k.split('.')[1:])] = checkpoint[k]
            encoder_projection.load_state_dict(loaded_state_dict, strict=False)
        
        return encoder_projection
    
    @classmethod
    def init_vicuna_projection(cls, input_size, output_size, load_projection_path=False, load_projection_type="", projection_key=None):
        proj = nn.Linear(input_size, output_size)
        if load_projection_path:
            url_or_filename=load_projection_path
            logging.info(f"Loading pretrained projection weights from {url_or_filename} of type {load_projection_type} with key {projection_key if projection_key else load_projection_type+'_llm_proj.'}")
            if is_url(url_or_filename):
                cached_file = download_cached_file(
                    url_or_filename, check_hash=False, progress=True
                )
                checkpoint = torch.load(cached_file, map_location="cpu")
            elif os.path.isfile(url_or_filename):
                checkpoint = torch.load(url_or_filename, map_location="cpu")
            else:
                raise RuntimeError("checkpoint url or path is invalid")
            if load_projection_type:
                load_projection_type = f"{load_projection_type}_"
            loaded_state_dict = {}
            if 'model' in checkpoint:
                checkpoint = checkpoint['model'] 
            for k in checkpoint.keys():
                if projection_key:
                    if projection_key in k:
                        loaded_state_dict['.'.join(k.split('.')[1:])] = checkpoint[k]
                else:
                    if load_projection_type+'llm_proj.' in k:
                        loaded_state_dict['.'.join(k.split('.')[1:])] = checkpoint[k]
            proj.load_state_dict(loaded_state_dict, strict=False)
        
        return proj

    @classmethod
    def init_Qformer(cls, num_query_token, modality_width, cross_attention_freq=2, pretrained_qformer=None, load_attention=False, load_qformer_type=""):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = modality_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        encoder_config.vocab_size += 1 # for special token [DEC]
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

        if pretrained_qformer:
            url_or_filename=pretrained_qformer
            logging.info(f"Loading pretrained qformer weights and query tokens from {url_or_filename} of type {load_qformer_type}")
            if is_url(url_or_filename):
                cached_file = download_cached_file(
                    url_or_filename, check_hash=False, progress=True
                )
                checkpoint = torch.load(cached_file, map_location="cpu")
            elif os.path.isfile(url_or_filename):
                checkpoint = torch.load(url_or_filename, map_location="cpu")
            else:
                raise RuntimeError("checkpoint url or path is invalid")
            
            if load_qformer_type:
                load_qformer_type = f"{load_qformer_type}_"
            loaded_state_dict = {}
            if 'model' in checkpoint:
                checkpoint = checkpoint['model'] 
            for k in checkpoint.keys():
                if load_qformer_type+'Qformer.' in k:
                    if not load_attention and 'attention' in k:
                        continue
                    loaded_state_dict['.'.join(k.split('.')[1:])] = checkpoint[k]
            Qformer.load_state_dict(loaded_state_dict, strict=False)
            query_tokens.data = checkpoint[load_qformer_type+'query_tokens']
        
        return Qformer, query_tokens
    
    def get_state_dict(self, url_or_filename, **kwargs):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        if "model" in checkpoint.keys():
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        return state_dict
    
    def load_from_pretrained(self, url_or_filename, **kwargs):
        state_dict = self.get_state_dict(url_or_filename)
        self.load_state_dict(state_dict, strict=False)
        logging.info("load checkpoint from %s" % url_or_filename)

    def load_checkpoint(self, url_or_filename, **kwargs):
        """
        Load from a finetuned checkpoint.

        This should expect no mismatch in the model keys and the checkpoint keys.
        """
        state_dict = self.get_state_dict(url_or_filename)
        self.load_state_dict(state_dict, strict=True)
        logging.info("load checkpoint from %s" % url_or_filename)
    
    def load_state_dict(self, state_dict, strict=True):
        # from pdb import set_trace; set_trace()
        unexpected_keys = []
        missing_keys = []
        if self.shared_qformer and not self.projection_only:
            ## Load Q-Former if it is not loaded from config
            if not getattr(self, "pretrained_shared_qformer"):
                shared_qformer_state_dict = {'.'.join(k.split('.')[1:]):v for k,v in state_dict.items() if "shared_Qformer" == k.split('.')[0]}
                msg = self.shared_Qformer.load_state_dict(shared_qformer_state_dict, strict=strict)
                missing_keys.extend(msg.missing_keys)
                ## Load query tokens
                if "shared_query_tokens" not in state_dict:
                    missing_keys.append("shared_query_tokens")
                else:
                    self.shared_query_tokens = state_dict["shared_query_tokens"]
                missing_keys.extend(msg.missing_keys)
                unexpected_keys.extend(msg.unexpected_keys)

                for modality in self.modalities:
                    # Map shared Qformer by reference to all modalities. 
                    setattr(self, f"{modality}_Qformer", self.shared_Qformer) 
                    getattr(self, f"{modality}_query_tokens").data =  state_dict[f"shared_query_tokens"]
                    # load encoder projections
                    modality_encoder_projection_dict = {'.'.join(k.split('.')[1:]):v for k,v in state_dict.items() if f"{modality}_encoder_projection" in k.split('.')[0]}
                    msg = getattr(self, f"{modality}_encoder_projection").load_state_dict(modality_encoder_projection_dict, strict=strict)
                    missing_keys.extend(msg.missing_keys)
                    unexpected_keys.extend(msg.unexpected_keys)
                    # load modality layer norm
                    if getattr(self,f"load_ln_type_{modality}") == "vision":
                        modality_ln_dict = {'.'.join(k.split('.')[1:]):v for k,v in state_dict.items() if f"ln_vision" in k.split('.')[0]}
                    else:
                        modality_ln_dict = {'.'.join(k.split('.')[1:]):v for k,v in state_dict.items() if f"{modality}_ln" in k.split('.')[0]}
                    msg = getattr(self, f"{modality}_ln").load_state_dict(modality_ln_dict, strict=strict)
                    missing_keys.extend(msg.missing_keys)
                    unexpected_keys.extend(msg.unexpected_keys)
            
            ## Load Shared LLM projection if not loaded by config
            if not getattr(self, "load_projection_shared"):  
                shared_llm_projection_dict = {'.'.join(k.split('.')[1:]):v for k,v in state_dict.items() if f"shared_llm_proj" in k.split('.')[0]}
                msg = self.shared_llm_proj.load_state_dict(shared_llm_projection_dict, strict=strict)    
                missing_keys.extend(msg.missing_keys)
                unexpected_keys.extend(msg.unexpected_keys)
                for modality in self.modalities:   
                    ## Map to modality projections by reference
                    msg = setattr(self, f"{modality}_llm_proj", self.shared_llm_proj)
        else:
            for modality in self.modalities:
                ## Load Q-Former if not loaded from config
                if not getattr(self, f"pretrained_{modality}_qformer") or ((self.projection_only or getattr(self, f"projection_only_{modality}")) and not getattr(self, f"projection_path_{modality}")):

                    if self.projection_only or getattr(self, f"projection_only_{modality}") :
                        if not getattr(self, f"projection_path_{modality}"):
                            logging.info(f"Loaded {modality} projection")
                            modality_qformer_state_dict = {'.'.join(k.split('.')[1:]):v for k,v in state_dict.items() if f"{modality}_projection" == k.split('.')[0]}
                            msg = getattr(self, f"{modality}_projection").load_state_dict(modality_qformer_state_dict, strict=strict)
                            missing_keys.extend(msg.missing_keys)
                            unexpected_keys.extend(msg.unexpected_keys)
                    else:
                        modality_qformer_state_dict = {'.'.join(k.split('.')[1:]):v for k,v in state_dict.items() if f"{modality}_Qformer" == k.split('.')[0]}
                        msg = getattr(self, f"{modality}_Qformer").load_state_dict(modality_qformer_state_dict, strict=strict)
                        missing_keys.extend(msg.missing_keys)
                        unexpected_keys.extend(msg.unexpected_keys)
                    ## Load query tokens
                    if not self.projection_only and not getattr(self, f"projection_only_{modality}"):
                        if f"{modality}_query_tokens" not in state_dict:
                            missing_keys.append(f"{modality}_query_tokens")
                        else:
                            logging.info(f"Loaded {modality} query tokens")
                            getattr(self, f"{modality}_query_tokens").data =  state_dict[f"{modality}_query_tokens"]
                    # load modality layer norm if not loaded from config
                    if getattr(self,f"load_ln_type_{modality}") == "vision":
                        logging.info(f"Loaded {modality} vision ln")
                        modality_ln_dict = {'.'.join(k.split('.')[1:]):v for k,v in state_dict.items() if f"ln_vision" in k.split('.')[0]}
                    else:
                        modality_ln_dict = {'.'.join(k.split('.')[1:]):v for k,v in state_dict.items() if f"{modality}_ln" in k.split('.')[0]}
                    msg = getattr(self, f"{modality}_ln").load_state_dict(modality_ln_dict, strict=strict)
                    missing_keys.extend(msg.missing_keys)
                    unexpected_keys.extend(msg.unexpected_keys)
                ## Load LLM projections if not loaded from config
                if not getattr(self, f"load_projection_{modality}") or  (getattr(self, f"projection_only_{modality}") or self.projection_only):
                    if not getattr(self, f"projection_path_{modality}"):
                        logging.info(f"Loaded {modality} llm  projection")
                        modality_llm_projection_dict = {'.'.join(k.split('.')[1:]):v for k,v in state_dict.items() if f"{modality}_llm_proj" in k.split('.')[0]}
                        msg = getattr(self, f"{modality}_llm_proj").load_state_dict(modality_llm_projection_dict, strict=strict)
                        missing_keys.extend(msg.missing_keys)
                        unexpected_keys.extend(msg.unexpected_keys)
        
        ## llm model is loaded from pretrained
        lora_state_dict = {'.'.join(k.split('.')[1:]):v for k,v in state_dict.items() if f"llm_model" in k.split('.')[0]}

        if not self.lora or len(lora_state_dict) == 0:
            unexpected_keys = [k for k in unexpected_keys if k.split('.')[0] != 'llm_model']
        else:
            msg = self.llm_model.load_state_dict({'.'.join(k.split('.')[1:]):v for k,v in state_dict.items() if f"llm_model" in k.split('.')[0]}, strict=False)
            missing_keys.extend(["llm_model."+k for k in msg.missing_keys])
        missing_keys = [k for k in missing_keys if 'encoder' not in k.split('.')[0]]
        missing_keys = [k for k in missing_keys if k.split('.')[0] != 'llm_model']
        return _IncompatibleKeys(missing_keys, unexpected_keys)
    

    def before_evaluation(self, dataset, task_type, **kwargs):
        if task_type == MultimodalClassificationTask:
            self.candidates = dataset.classnames
            print(self.candidates)