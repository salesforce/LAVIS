from app import device, load_demo_image
from app.utils import load_model_cache, get_pending_jobs, create_uniq_user_job_name
from app import job_output_path, finished_job_path, pending_job_path
from lavis.processors import load_processor
from PIL import Image

import random
import numpy as np
import torch
import os, shutil, time

job_type = 'caption'

if torch.cuda.is_available():
    torch.cuda.set_device(0)
    device = "cuda"
else:
    device = "cpu"

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(int(seed))
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = False
        cudnn.deterministic = True

def back_end():
    vis_processor = load_processor("blip_image_eval").build(image_size=384)
    blip_large_model = load_model_cache(
                "blip_caption",
                model_type=f"large_coco",
                is_eval=True,
                device=device,
            )
    blip_base_model = load_model_cache(
                "blip_caption",
                model_type=f"base_coco",
                is_eval=True,
                device=device,
            )
    os.makedirs(os.path.join(finished_job_path, job_type), exist_ok=True)
    while True:
        pending_jobs = get_pending_jobs(job_type)
        for job in pending_jobs:
            while True:
                with open(job) as f:
                    content = f.readline().rstrip(' \n')
                    if len(content.split('\t')) == 5: break
            time_stamp, blip_type, sampling_method, num_captions, seed = content.split('\t')
            outpath = os.path.join(job_output_path, job_type)
            os.makedirs(outpath, exist_ok=True)
            img_file = outpath+'/{}_raw_image.pt'.format(create_uniq_user_job_name(time_stamp, sampling_method))
            while True:
                if os.path.exists(img_file):
                    break
            time.sleep(1)
            img = torch.load(outpath+'/{}_raw_image.pt'.format(create_uniq_user_job_name(time_stamp, sampling_method)),map_location=torch.device(device))
            if blip_type == 'large':
                model = blip_large_model
            else:
                model = blip_base_model
            use_nucleus_sampling = False
            if sampling_method == 'Nucleus sampling':
                use_nucleus_sampling = True
            setup_seed(int(seed))
            captions = generate_caption(model, img, use_nucleus_sampling, int(num_captions))
            caption_result = outpath+'/{}_result.txt'.format(create_uniq_user_job_name(time_stamp, sampling_method))
            with open(caption_result,'w') as f:
                for caption in captions:
                    f.write(caption+'\n')
            shutil.move(job, os.path.join(finished_job_path, job_type))
            os.remove(img_file)


def generate_caption(
    model, image, use_nucleus_sampling=False, num_captions = 1, num_beams=3, max_length=40, min_length=5
):
    samples = {"image": image}

    captions = []
    if use_nucleus_sampling:
        #for _ in range(5):
        captions = model.generate(
                samples,
                use_nucleus_sampling=True,
                max_length=max_length,
                min_length=min_length,
                top_p=0.9,
                num_captions=num_captions
        )
        #captions.append(caption[0])
    else:
        caption = model.generate(
            samples,
            use_nucleus_sampling=False,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            num_captions=1
        )
        captions.append(caption[0])
    return captions
if __name__ == "__main__":
    back_end()
