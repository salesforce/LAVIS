#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/anthonytmh/lavis-pnpvqa/blob/pnp_vqa/projects/pnp-vqa/pnp_vqa.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# ## Img2Prompt-VQA: Inference Demo

# In[2]:


# install requirements
import sys
# if 'google.colab' in sys.modules:
 #   print('Running in Colab.')
  #  get_ipython().system('git clone https://github.com/salesforce/LAVIS')
   # get_ipython().run_line_magic('cd', 'LAVIS')
   # get_ipython().system('pip install .')
   # get_ipython().system('pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.0.0/en_core_web_sm-3.0.0.tar.gz')
#else:
 #   get_ipython().system('pip install omegaconf')
  #  get_ipython().run_line_magic('cd', '../..')
   # get_ipython().system('pip install .')
  # 'pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.0.0/en_core_web_sm-3.0.0.tar.gz')


# In[3]:


import torch
import requests
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

from lavis.common.gradcam import getAttMap
from lavis.models import load_model_and_preprocess


# ### Load an example image and question

# In[ ]:


#img_url = 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/projects/pnp-vqa/demo.png' 
#raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')   
raw_image = Image.open("./demo.png").convert("RGB")
#display(raw_image.resize((400, 300)))
question = "What item s are spinning which can be used to control electric?"
print(question)


# In[ ]:


# setup device to use
device = 'cpu'
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


# ### Load Img2Prompt-VQA model

# In[ ]:


model, vis_processors, txt_processors = load_model_and_preprocess(name="img2prompt_vqa", model_type="base", is_eval=True, device=device)


# ### Preprocess image and text inputs

# In[ ]:


image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
question = txt_processors["eval"](question)

samples = {"image": image, "text_input": [question]}


# In[ ]:





# ### Img2Prompt-VQA utilizes 4 submodels to perform VQA:
# #### 1. Image-Question Matching 
# Compute the relevancy score of image patches with respect to the question using GradCAM

# In[ ]:


samples = model.forward_itm(samples=samples)


# In[ ]:


# Gradcam visualisation
dst_w = 720
w, h = raw_image.size
scaling_factor = dst_w / w

resized_img = raw_image.resize((int(w * scaling_factor), int(h * scaling_factor)))
norm_img = np.float32(resized_img) / 255
gradcam = samples['gradcams'].reshape(24,24)

avg_gradcam = getAttMap(norm_img, gradcam, blur=True)

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.imshow(avg_gradcam)
ax.set_yticks([])
ax.set_xticks([])
print('Question: {}'.format(question))


# #### 2. Image Captioning
# Generate question-guided captions based on the relevancy score

# In[ ]:


samples = model.forward_cap(samples=samples, num_captions=50, num_patches=20)
print('Examples of question-guided captions: ')
print(samples['captions'][0][:5])


# #### 3. Question Generation
# Generate synthetic questions using the captions

# In[ ]:


samples = model.forward_qa_generation(samples)
print('Sample Question: {} \nSample Answer: {}'.format(samples['questions'][:5], samples['answers'][:5]))


# #### 4. Prompt Construction
# Prepare the prompts for LLM

# ### Generate answer by calling `predict_answers()` directly
# 

# In[ ]:
Img2Prompt = model.prompts_construction(samples)


from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
def load_model(model_selection):
    model = AutoModelForCausalLM.from_pretrained(model_selection)
    tokenizer = AutoTokenizer.from_pretrained(model_selection, use_fast=False)
    return model,tokenizer
def postprocess_Answer(text):
    for i, ans in enumerate(text):
        for j, w in enumerate(ans):
            if w == '.' or w == '\n':
                ans = ans[:j].lower()
                break
    return ans

model,tokenizer = load_model('facebook/opt-6.7b')

Img2Prompt_input = tokenizer(Img2Prompt, padding='longest', truncation=True, return_tensors="pt").to(
    device)

assert (len(Img2Prompt_input.input_ids[0])+20) <=2048
# print(len(question_input.attention_mask[0]))

outputs_list  = []
outputs = model.generate(input_ids=Img2Prompt_input.input_ids,
                             attention_mask=Img2Prompt_input.attention_mask,
                             max_length=20+len(Img2Prompt_input.input_ids[0]),
                             return_dict_in_generate=True,
                             output_scores = True
                             )
outputs_list.append(outputs)




pred_answer = tokenizer.batch_decode(outputs.sequences[:, len(Img2Prompt_input.input_ids[0]):])
pred_answer = postprocess_Answer(pred_answer)
print({"question": question, "answer": pred_answer})

#pred_answers, caption, gradcam = model.predict_answers(samples, num_captions=50, num_patches=20)
#print('Question: {} \nPredicted answer: {}'.format(question, pred_answers[0]))

