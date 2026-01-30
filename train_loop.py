from sam.models.frameworks.sam import Sam
from mmcv.utils.config import ConfigDict

model = Sam(
    backbone= {'type': 'ResNet3d',
  'pretrained2d': True,
  'pretrained': 'torchvision://resnet18',
  'depth': 18,
  'in_channels': 1,
  'spatial_strides': (2, 2, 2, 2),
  'temporal_strides': (1, 1, 1, 2),
  'conv1_kernel': (3, 7, 7),
  'conv1_stride_t': 1,
  'conv1_stride_s': 1,
  'pool1_stride_t': 1,
  'with_pool1': False,
  'with_pool2': True,
  'conv_cfg': {'type': 'Conv3d'},
  'inflate': ((0, 0), (0, 0), (1, 1), (1, 1)),
  'norm_eval': False,
  'zero_init_residual': False},

   neck = {'type': 'FPN3d',
  'end_level': 3,
  'in_channels': [64, 128, 256],
  'out_channels': 128,
  'num_outs': 3,
  'conv_cfg': {'type': 'Conv3d'}},

    read_out_head = {'type': 'FPN3d',
  'end_level': 1,
  'in_channels': [512],
  'out_channels': 128,
  'num_outs': 1,
  'conv_cfg': {'type': 'Conv3d'}},

  # To allow attribute style, wrap train_cfg under ConfigDict
   train_cfg = ConfigDict({'pre_select_pos_number': 2000,
  'after_select_pos_number': 100,
  'pre_select_neg_number': 2000,
  'after_select_neg_number': 500,
  'positive_distance': 2.0,
  'ignore_distance': 20.0,
  'coarse_positive_distance': 25.0,
  'coarse_ignore_distance': 5.0,
  'coarse_z_thres': 6.0,
  'coarse_pre_select_neg_number': 250,
  'coarse_after_select_neg_number': 200,
  'coarse_global_select_number': 1000,
  'temperature': 0.5}),
 test_cfg = {'save_path': '/data/results/result-dlt/',
  'output_embedding': True}

)
from mmcv.runner import load_checkpoint
checkpoint = load_checkpoint(model, 'checkpoints/SAM.pth', map_location='cpu')

from sam.datasets.dataset3dsam import Dataset3dsam


ds = Dataset3dsam(
    data_dir = 'data/processed_data/NIH_lymph_node/nii-crop/',
    index_file = 'data/processed_data/NIH_lymph_node/ind_files/lymphnode_abd_filename.csv',
    pipeline= [{'type': 'LoadTioImage'},
  {'type': 'CropBackground'},
  {'type': 'ComputeAugParam_sample'},
  {'type': 'MultiBranch',
   'view1': [{'type': 'ExtraAttrs', 'tag': 'view1'},
    {'type': 'Crop'},
    {'type': 'Resample'},
    {'type': 'Crop', 'switch': 'fix'},
    {'type': 'RescaleIntensity'},
    {'type': 'RandomNoise3d'},
    {'type': 'GenerateMeshGrid'},
    {'type': 'GenerateMetaInfo'},
    {'type': 'DefaultFormatBundle3d'},
    {'type': 'Collect3d',
     'keys': ['img', 'meshgrid', 'valid'],
     'meta_keys': ('filename', 'tag', 'crop_info')}],
   'view2': [{'type': 'ExtraAttrs', 'tag': 'view2'},
    {'type': 'Crop'},
    {'type': 'Resample'},
    {'type': 'Crop', 'switch': 'fix'},
    {'type': 'RescaleIntensity'},
    {'type': 'RandomNoise3d'},
    {'type': 'GenerateMeshGrid'},
    {'type': 'GenerateMetaInfo'},
    {'type': 'DefaultFormatBundle3d'},
    {'type': 'Collect3d',
     'keys': ['img', 'meshgrid', 'valid'],
     'meta_keys': ('filename', 'tag', 'crop_info')}]}]
)



import torch
from mmcv.parallel.data_container import DataContainer

def unbox(dc): return dc.data if isinstance(dc, DataContainer) else dc


data1 = ds[0][0]
data2 = ds[0][1]


batch_img = torch.stack([unbox(data1['img']), unbox(data2['img'])])
batch_meshgrid = torch.stack([unbox(data1['meshgrid']), unbox(data2['meshgrid'])])
batch_valid = torch.stack([unbox(data1['valid']), unbox(data2['valid'])])
batch_metas = [unbox(data1['img_metas']), unbox(data2['img_metas'])]


model = model.train().float()


import torch.optim as optim
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10):
    for d in ds:
        
        data0, data1 = d

        batch_img = torch.stack([unbox(data1['img']), unbox(data2['img'])])
        batch_meshgrid = torch.stack([unbox(data1['meshgrid']), unbox(data2['meshgrid'])])
        batch_valid = torch.stack([unbox(data1['valid']), unbox(data2['valid'])])
        batch_metas = [unbox(data1['img_metas']), unbox(data2['img_metas'])]
        
        optimizer.zero_grad()

        # --- FORWARD PASS ---
        # We call forward_train directly to get the loss dictionary
        losses = model.forward_train(
            img=batch_img,
            img_metas=batch_metas,
            meshgrid=batch_meshgrid,
            valid=batch_valid
        )

        # --- LOSS CALCULATION ---
        # losses is a dict: {'loss_1': tensor, 'loss_2': tensor}
        # We must sum them into one scalar for .backward()
        total_loss = sum(_loss.mean() for _loss in losses.values())

        # --- BACKWARD & OPTIMIZE ---
        total_loss.backward()
        optimizer.step()
        print(total_loss)
