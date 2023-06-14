# %%
import torch

from PIL import Image
from lavis.models import load_model_and_preprocess

# %%
torch.cuda.is_available()

# %%
model, vis_preprocess, txt_preprocess = load_model_and_preprocess("blip_diffusion", "base", device="cuda")

# %%
class_names = [txt_preprocess["eval"]("dog")]
prompt = [txt_preprocess["eval"]("A dog swimming underwater")]
ctx_begin_pos = [2]

image = Image.open("../images/dog.png").convert("RGB")
input_images = vis_preprocess["eval"](image).unsqueeze(0).cuda()

# %%
samples = {
    "input_images": input_images,
    "src_subject": class_names,
    "prompt": prompt,
    "ctx_begin_pos": ctx_begin_pos,
}

# %%
iter_seed = 42
guidance_scale = 7.5
num_inference_steps = 50
negative_prompt = "over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"

output = model.generate(
    samples,
    seed=iter_seed,
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps,
    neg_prompt=negative_prompt,
    height=512,
    width=512,
)

# text.input_ids,
# query_embeds=query_tokens,
# attention_mask=attention_mask,
# encoder_hidden_states=image_embeds_frozen,
# encoder_attention_mask=image_atts,

# tensor([[ 101, 3899,  102]], device='cuda:0')
# torch.Size([1, 16, 768])
# tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
#        device='cuda:0')
# torch.float32
# device(type='cuda', index=0)
# torch.Size([1, 257, 1024])
# torch.Size([1, 257])

