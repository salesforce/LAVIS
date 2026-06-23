FROM nvcr.io/nvidia/pytorch:21.06-py3 

COPY requirements-app.txt requirements_gpu.txt

ENV DEBIAN_FRONTEND=noninteractive 

RUN apt-get update && apt-get install -y --no-install-recommends \
    locales \
    wget \
    build-essential \
    vim \
    htop \
    curl \
    git less ssh cmake \
    zip unzip gzip bzip2 \
    python3-tk gcc g++ libpq-dev

RUN apt -y install openssh-server openssh-client
# BLIP-specific commands
RUN apt-get install -y libxtst6
RUN pip3 uninstall -y torch
RUN pip3 uninstall -y torchtext
RUN pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip3 install omegaconf
RUN pip3 install ipython
RUN pip3 install pycocoevalcap
RUN pip3 install pycocotools
RUN pip3 install timm==0.4.12
RUN pip3 install fairscale==0.4.4
RUN apt install -y default-jre
RUN apt install -y openjdk-11-jre-headless
RUN apt install -y openjdk-8-jre-headless
RUN pip uninstall opencv-python
RUN pip uninstall opencv-contrib-python
RUN pip uninstall opencv-contrib-python-headless


RUN  pip3 install -r requirements_gpu.txt


COPY . /lavis_app
WORKDIR /lavis_app

RUN wget https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt
RUN mv  sd-v1-4.ckpt /lavis_app/stable-diffusion/sd-v1-4.ckpt

ENV PYTHONPATH="${PYTHONPATH}:./:/lavis_app:/lavis_app/stable-diffusion"

EXPOSE 8080
RUN chmod +x /lavis_app/run_scripts/start_lavis_app.sh
ENTRYPOINT ["/lavis_app/run_scripts/start_lavis_app.sh" ]


