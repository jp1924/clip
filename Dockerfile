from nvcr.io/nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV PATH /usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:$LD_LIBRARY_PATH

WORKDIR /root
USER root

# 서버 관련 패키지 설치
RUN apt-get update && \
    apt-get install -y --no-install-recommends apt-utils && \
    apt-get install -y ffmpeg wget net-tools build-essential git curl && \
    apt-get install -y python3.10 python3.10-dev python3.10-venv python3-pip &&

RUN pip install -U pip wheel setuptools && \
    pip install transformers==4.37.2 accelerate==0.26.1 datasets==2.16.1 evaluate==0.4.1 && \
    pip install bitsandbytes==0.41.3.post2 scipy==1.12.0 sentencepiece==0.1.99 peft==0.8.2 deepspeed==0.13.1 wandb==0.16.3 && \
    pip install setproctitle glances[gpu] && \
    pip install black flake8 && \
    pip install torch==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install flash-attn==2.5.2 