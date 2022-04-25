import time
import datetime

import torch
import torch.nn.functional as F
import torch.distributed as dist

from tasks.base_task import BaseTask

from common.registry import registry

import utils.blip_utils as utils

@registry.register_task('retrieval')
class RetrievalTask(BaseTask):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
    
    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        return cls(cfg=run_cfg)
    
    def evaluation(self, model, data_loader, **kwargs):
        config = self.config

        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Evaluation:'    
        
        print('Computing features for evaluation...')
        start_time = time.time()  

        texts = data_loader.dataset.text   
        num_text = len(texts)
        text_bs = 256
        text_ids = []
        text_embeds = []  
        text_atts = []
        for i in range(0, num_text, text_bs):
            text = texts[i: min(num_text, i+text_bs)]
            text_input = model.tokenizer(text, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(model.device) 
            text_output = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')  
            text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:,0,:]))
            text_embeds.append(text_embed)   
            text_ids.append(text_input.input_ids)
            text_atts.append(text_input.attention_mask)
        
        text_embeds = torch.cat(text_embeds,dim=0)
        text_ids = torch.cat(text_ids,dim=0)
        text_atts = torch.cat(text_atts,dim=0)
        text_ids[:,0] = model.tokenizer.enc_token_id
        
        image_feats = []
        image_embeds = []
        for image, img_id in data_loader: 
            image = image.to(model.device) 
            image_feat = model.visual_encoder(image)   
            image_embed = model.vision_proj(image_feat[:,0,:])            
            image_embed = F.normalize(image_embed,dim=-1)      
            
            image_feats.append(image_feat.cpu())
            image_embeds.append(image_embed)
        
        image_feats = torch.cat(image_feats,dim=0)
        image_embeds = torch.cat(image_embeds,dim=0)
        
        sims_matrix = image_embeds @ text_embeds.t()
        score_matrix_i2t = torch.full((len(data_loader.dataset.image),len(texts)),-100.0).to(device)
        
        num_tasks = utils.get_world_size()
        rank = utils.get_rank() 
        step = sims_matrix.size(0)//num_tasks + 1
        start = rank*step
        end = min(sims_matrix.size(0),start+step)

        for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)): 
            topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)

            encoder_output = image_feats[start+i].repeat(config['k_test'],1,1).to(model.device)
            encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(model.device)
            output = model.text_encoder(text_ids[topk_idx], 
                                        attention_mask = text_atts[topk_idx],
                                        encoder_hidden_states = encoder_output,
                                        encoder_attention_mask = encoder_att,                             
                                        return_dict = True,
                                    )
            score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
            score_matrix_i2t[start+i,topk_idx] = score + topk_sim
            
        sims_matrix = sims_matrix.t()
        score_matrix_t2i = torch.full((len(texts),len(data_loader.dataset.image)),-100.0).to(device)
        
        step = sims_matrix.size(0)//num_tasks + 1
        start = rank*step
        end = min(sims_matrix.size(0),start+step)    
        
        for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)): 
            
            topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
            encoder_output = image_feats[topk_idx].to(model.device)
            encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(model.device)
            output = model.text_encoder(text_ids[start+i].repeat(config['k_test'],1), 
                                        attention_mask = text_atts[start+i].repeat(config['k_test'],1),
                                        encoder_hidden_states = encoder_output,
                                        encoder_attention_mask = encoder_att,                             
                                        return_dict = True,
                                    )
            score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
            score_matrix_t2i[start+i,topk_idx] = score + topk_sim

        if utils.is_dist_avail_and_initialized():
            dist.barrier()   
            torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM) 
            torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)        
            
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Evaluation time {}'.format(total_time_str)) 

        return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()
