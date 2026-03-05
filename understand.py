from sam.apis.train import train_detector
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmcv import Config
from sam import *
from torch import optim

cfg = Config.fromfile('configs/sam/sam_NIHLN.py')
cfg.model

model = build_detector(cfg.model)

# Download weights and data
# git clone https://github.com/alibaba-damo-academy/self-supervised-anatomical-embedding-v2.git prj
# pip install gdown && \
#     gdown 1LH9E5D273kOJXrUmBv_s2hXuOZV-dR65 -O weights.zip && \
#     unzip weights.zip && \
#     mv Self-supervised_Anatomical_Embeddings/checkpoints . && \
#     mv Self-supervised_Anatomical_Embeddings/data . && \
#     rm -r Self-supervised_Anatomical_Embeddings weights.zip 

# Prepare the data
# ipython misc/lymphnode_preprocess_crop_multi_process.py

ds = build_dataset(cfg.data.train)


import torch
from mmcv.parallel import DataContainer

# 1. Get two samples from the dataset
data1 = ds[0][0]
data2 = ds[0][1]

def unbox(dc):
    """Extracts the tensor or list from an MMCV DataContainer."""
    return dc.data if isinstance(dc, DataContainer) else dc

# 2. Manually Batch (Stack) the tensors
# We combine sample 1 and sample 2 into a batch of size 2
batch_img = torch.stack([unbox(data1['img']), unbox(data2['img'])])
batch_meshgrid = torch.stack([unbox(data1['meshgrid']), unbox(data2['meshgrid'])])
batch_valid = torch.stack([unbox(data1['valid']), unbox(data2['valid'])])

# 3. Handle img_metas (It must be a list of dicts)
# unbox(data1['img_metas']) usually returns a single dict
batch_metas = [unbox(data1['img_metas']), unbox(data2['img_metas'])]

# 4. Run the model
# model = model.train().float()
# losses = model.forward_train(
#     img=batch_img,
#     img_metas=batch_metas,
#     meshgrid=batch_meshgrid,
#     valid=batch_valid
# )

# print(losses)

# Build optimizer
epochs = 2
opt = optim.Adam(model.parameters(), lr=1e-4)

# Training loop using manual batching
for epoch in range(epochs):
    opt.zero_grad()
    
    losses = model.forward_train(
        img=batch_img,
        img_metas=batch_metas,
        meshgrid=batch_meshgrid,
        valid=batch_valid
    )
    
    # losses is a dict, sum all loss components
    loss =  losses['loss']
    loss.backward()
    opt.step()
    
    print(f'Epoch {epoch}, Loss: {loss.item():.4f}')



# About the MMCV dataset and dataloader
# from mmdet.datasets.pipelines import Compose
# p = Compose(cfg.data.train['pipeline'])


# from mmdet.datasets import build_dataset, build_dataloader

# dl = build_dataloader(ds, 10, 4, 0) # dataset, batch_size, n_workers, n_gpus

# l = list(dl)

# len(l) # 1
# len(l[0]) # 2
# len(l[0][0]) # 4 dict
# l[0][0].keys() # dict_keys(['img_metas', 'img', 'meshgrid', 'valid'])

# l[0][0]['img'].shape # AttributeError: 'DataContainer' object has no attribute 'shape'
# l[0][0]['img'].data # a list of one item
# l[0][0]['img'].data[0].shape # torch.Size([10, 1, 32, 96, 96])

# model(img=l[0][0]['img'].data[0], img_metas=l[0][0]['img_metas'], meshgrid=l[0][0]['meshgrid'].data[0], valid=l[0][0]['valid'].data[0]) # Out[79]: {'loss': tensor(11.3252, grad_fn=<AddBackward0>)}

