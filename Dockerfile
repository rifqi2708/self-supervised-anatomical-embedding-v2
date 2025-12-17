FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel
WORKDIR /root

RUN apt update && apt install -y git wget curl unzip nano libgl1 libglib2.0-0
RUN python3 -m pip install --upgrade pip

# Repo
RUN git clone https://github.com/alibaba-damo-academy/self-supervised-anatomical-embedding-v2.git prj
RUN pip install gdown && \
    gdown 1LH9E5D273kOJXrUmBv_s2hXuOZV-dR65 -O weights.zip && \
    unzip weights.zip && \
    mv Self-supervised_Anatomical_Embeddings/checkpoints prj && \
    mv Self-supervised_Anatomical_Embeddings/data prj && \
    rm weights.zip


# Code tunnel
RUN curl -Lk 'https://update.code.visualstudio.com/1.85.2/cli-alpine-x64/stable' --output vscode_cli.tar.gz && \
    tar -xf vscode_cli.tar.gz && \
    rm vscode_cli.tar.gz

# Dependencies
RUN pip install SimpleITK --only-binary=:all:
RUN pip install torchvision==0.10.0 torchio nibabel numpy matplotlib scikit-image statsmodels
RUN pip install openmim && mim install mmcv-full==1.5.0 && pip install mmdet==2.20.0

# To run in Runpod
CMD ["sleep", "infinity"]

