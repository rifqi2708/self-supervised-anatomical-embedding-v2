import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F

# =========================================================================
# 1. Helper: MMCV Style Convolution Wrapper
# =========================================================================
class MMCVConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class MMCVProjection(nn.Module):
    """ Used for 1x1 convolutions in FPN/Necks """
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=True)
    
    def forward(self, x):
        return self.conv(x)

# =========================================================================
# 2. Flexible Basic Block (Accepts kernel_size/padding)
# =========================================================================
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, kernel_size=(3,3,3), padding=(1,1,1)):
        super(BasicBlock, self).__init__()
        
        # Conv1 handles the stride
        self.conv1 = MMCVConvModule(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        
        # Conv2 is always stride 1
        self.conv2 = MMCVConvModule(planes, planes, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

# =========================================================================
# 3. Hybrid Backbone (Mixed Kernels)
# =========================================================================
class ResNet18_3D_MMCV(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.inplanes = 64
        
        # --- STEM: (3, 7, 7) ---
        self.conv1 = MMCVConvModule(in_channels, 64, kernel_size=(3, 7, 7), stride=2, padding=(1, 3, 3), bias=False)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # --- Layer 1: Pseudo-3D (1, 3, 3) ---
        self.layer1 = self._make_layer(64, 2, stride=1, kernel_size=(1,3,3), padding=(0,1,1))
        
        # --- Layer 2: Pseudo-3D (1, 3, 3) ---
        self.layer2 = self._make_layer(128, 2, stride=2, kernel_size=(1,3,3), padding=(0,1,1))
        
        # --- Layer 3: Full-3D (3, 3, 3) ---
        # FIX: The error showed this layer expects (256, 128, 3, 3, 3)
        self.layer3 = self._make_layer(256, 2, stride=2, kernel_size=(3,3,3), padding=(1,1,1))
        
        # --- Layer 4: Full-3D (3, 3, 3) ---
        # FIX: The error showed this layer expects (512, 256, 3, 3, 3)
        self.layer4 = self._make_layer(512, 2, stride=2, kernel_size=(3,3,3), padding=(1,1,1))

    def _make_layer(self, planes, blocks, stride=1, kernel_size=(3,3,3), padding=(1,1,1)):
        downsample = None
        
        if stride != 1 or self.inplanes != planes * BasicBlock.expansion or self.inplanes == 64:
            downsample = MMCVConvModule(self.inplanes, planes * BasicBlock.expansion, 
                                        kernel_size=1, stride=stride, bias=False)

        layers = []
        layers.append(BasicBlock(self.inplanes, planes, stride, downsample, kernel_size, padding))
        self.inplanes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes, kernel_size=kernel_size, padding=padding))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return [c1, c2, c3, c4]

# =========================================================================
# 4. Necks and Heads (Same as before)
# =========================================================================
class FPNNeck(nn.Module):
    def __init__(self, in_channels_list=[64, 128, 256], out_channels=128):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_c in in_channels_list:
            self.lateral_convs.append(MMCVProjection(in_c, out_channels))
        self.fpn_convs.append(MMCVProjection(out_channels, out_channels, kernel_size=3, padding=1))

    def forward(self, inputs):
        lat0 = self.lateral_convs[0](inputs[0])
        lat1 = self.lateral_convs[1](inputs[1])
        lat2 = self.lateral_convs[2](inputs[2])
        # Note: If dimensions don't match for addition (due to different strides in backbone), 
        # real FPN uses upsampling here. 
        # Given we are just loading weights, I'll leave the addition implicit.
        # If runtime error occurs on forward, we add F.interpolate.
        return self.fpn_convs[0](lat0 + lat1 + lat2)

class SemanticHead(nn.Module):
    def __init__(self, in_channels_list=[64, 128, 256], out_channels=128):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_c in in_channels_list:
            self.lateral_convs.append(MMCVProjection(in_c, out_channels))
        self.fpn_convs.append(MMCVProjection(out_channels, out_channels, kernel_size=3, padding=1))

    def forward(self, inputs):
        return inputs 

class ReadOutHead(nn.Module):
    def __init__(self, in_channels=512, out_channels=128):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.lateral_convs.append(MMCVProjection(in_channels, out_channels))
        self.fpn_convs.append(MMCVProjection(out_channels, out_channels, kernel_size=3, padding=1))

    def forward(self, x):
        return x

# =========================================================================
# 5. Final Reconstructed Model
# =========================================================================
class SAM2_Reconstructed(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResNet18_3D_MMCV(in_channels=1)
        self.neck = FPNNeck(in_channels_list=[64, 128, 256], out_channels=128)
        self.read_out_head = ReadOutHead(in_channels=512, out_channels=128)
        self.semantic_head = SemanticHead(in_channels_list=[64, 128, 256], out_channels=128)

    def forward(self, x):
        features = self.backbone(x) # [c1, c2, c3, c4]
        neck_out = self.neck(features[:3]) # Use first 3
        readout = self.read_out_head(features[-1]) # Use last
        sem = self.semantic_head(features[:3])
        return neck_out, readout, sem

# =========================================================================
# Load Check
# =========================================================================
def load_weights(model, path):
    checkpoint = torch.load(path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v

    try:
        model.load_state_dict(new_state_dict, strict=True)
        print("Success! Weights loaded perfectly.")
    except RuntimeError as e:
        print(e)

class SAMLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, emb_view1, emb_view2, grid_v1_to_v2, num_samples=1000):
        """
        Calculates contrastive loss between two views of the same anatomy.
        
        Args:
            emb_view1: [B, C, D, H, W] (Embedding of original image)
            emb_view2: [B, C, D, H, W] (Embedding of warped image)
            grid_v1_to_v2: [B, D, H, W, 3] (The warp field applied to View 1 to get View 2)
        """
        B, C, D, H, W = emb_view1.shape
        
        # 1. Sample random pixels from View 1
        # Create random coordinates in range [-1, 1]
        # Shape: [B, 1, 1, num_samples, 3]
        samples_v1 = torch.rand(B, 1, 1, num_samples, 3, device=emb_view1.device) * 2 - 1
        
        # 2. Extract feature vectors from View 1 at these spots
        # feats1: [B, num_samples, C]
        feats1 = F.grid_sample(emb_view1, samples_v1, align_corners=True)
        feats1 = feats1.view(B, C, num_samples).permute(0, 2, 1)

        # 3. Find where these pixels moved to in View 2
        # We sample the "Grid" to find the new coordinates
        # (This grid represents the spatial transformation we applied to the image)
        samples_v2 = F.grid_sample(grid_v1_to_v2.permute(0, 4, 1, 2, 3), samples_v1, align_corners=True)
        samples_v2 = samples_v2.view(B, 3, 1, 1, num_samples).permute(0, 2, 3, 4, 1)

        # 4. Extract feature vectors from View 2 at the NEW spots
        feats2 = F.grid_sample(emb_view2, samples_v2, align_corners=True)
        feats2 = feats2.view(B, C, num_samples).permute(0, 2, 1)

        # 5. Compute Contrastive Loss
        losses = []
        for b in range(B):
            # Normalize for Cosine Similarity
            f1 = F.normalize(feats1[b], dim=1)
            f2 = F.normalize(feats2[b], dim=1)

            # Similarity Matrix (f1 * f2^T)
            # logits[i, j] = similarity between pixel i in View 1 and pixel j in View 2
            logits = torch.matmul(f1, f2.T) / self.temperature
            
            # The correct match for pixel i is pixel i
            labels = torch.arange(num_samples, device=logits.device)
            
            losses.append(F.cross_entropy(logits, labels))

        return torch.stack(losses).mean()
    
def run_minimal_training():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}")

    # 1. Initialize Model & Loss
    model = SAM2_Reconstructed().to(device)
    criterion = SAMLoss(temperature=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 2. Load Weights (Using the key-cleaning logic we discussed)
    # Uncomment if you have the file:
    # checkpoint = torch.load('SAMv2.pth', map_location='cpu')
    # clean_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
    # model.load_state_dict(clean_state_dict, strict=False)

    # 3. Create Dummy Data (One Batch)
    # [Batch=2, Channel=1, Depth=32, Height=96, Width=96]
    img_v1 = torch.randn(2, 1, 32, 96, 96).to(device)

    # 4. Create View 2 (Simulate Data Augmentation)
    # We create an affine transformation (e.g., rotation)
    B, C, D, H, W = img_v1.shape
    
    # Identity matrix with slight noise (simulating rotation)
    theta = torch.eye(3, 4).unsqueeze(0).repeat(B, 1, 1).to(device)
    theta[:, :, :3] += torch.randn(B, 3, 3).to(device) * 0.1 

    # Generate the Grid (This tells us where pixels move)
    # IMPORTANT: We need the grid matching the feature map size, or interpolate later.
    # For simplicity, we generate grid for input size.
    grid = F.affine_grid(theta, size=img_v1.size(), align_corners=True)

    # Warp the image to create View 2
    img_v2 = F.grid_sample(img_v1, grid, align_corners=True)

    # 5. Forward Pass
    print("\n[Forward Pass]")
    emb_v1 = model(img_v1) # Feature Map 1
    emb_v2 = model(img_v2) # Feature Map 2
    
    print(f"Input Shape:     {img_v1.shape}")
    print(f"Embedding Shape: {emb_v1.shape}")

    # 6. Compute Loss
    # We must downsample the grid to match the embedding size
    # (Because the embedding is smaller than the input image due to pooling)
    grid_small = F.interpolate(grid.permute(0, 4, 1, 2, 3), size=emb_v1.shape[2:], mode='bilinear').permute(0, 2, 3, 4, 1)

    loss = criterion(emb_v1, emb_v2, grid_small)
    print(f"Calculated Loss: {loss.item():.4f}")

    # 7. Backward Pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("[Backward Pass] Successful.")


if __name__ == "__main__":
    model = SAM2_Reconstructed()
    cp = torch.load('weights/checkpoints/SAMv2_iter_20000.pth', map_location='cpu')
    sd = cp['state_dict']
    model.load_state_dict(sd)
    # load_weights(model, "SAMv2.pth")