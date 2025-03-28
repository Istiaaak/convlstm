import torch
import torch.nn as nn
import numpy as np

class Patchify(nn.Module):
    def __init__(self, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x_2d = x.permute(0,2,1,3,4).reshape(B*T, C, H, W)
        patches = self.unfold(x_2d)  
        new_HW = H // self.patch_size
        patches = patches.reshape(B*T, C*(self.patch_size**2), new_HW, new_HW)
        patches = patches.view(B, T, -1, new_HW, new_HW).permute(0,2,1,3,4)
        return patches

class Unpatchify(nn.Module):
    def __init__(self, patch_size=4, out_size=64):
        super().__init__()
        self.patch_size = patch_size
        self.out_size = out_size
        self.fold = nn.Fold(
            output_size=(out_size, out_size),
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        B, newC, T, newH, newW = x.shape
        x_2d = x.permute(0,2,1,3,4).reshape(B*T, newC, newH*newW)
        out_2d = self.fold(x_2d) 
        out = out_2d.view(B, T, 1, self.out_size, self.out_size)
        return out


def collate_train_val(batch, device, patch_size=4):
    from torch import tensor
    patcher = Patchify(patch_size=patch_size).to(device)

    batch = np.array(batch)
    batch = tensor(batch).unsqueeze(1).float() / 255.0 
    batch = batch.to(device)


    rand = np.random.randint(10,20)
    input_  = batch[:, :, rand-10:rand]  
    target_ = batch[:, :, [rand]]

    input_patch = patcher(input_)
    return input_patch, target_

def collate_test(batch, device):
    from torch import tensor
    data = np.array(batch)
    data = tensor(data).unsqueeze(1).float() / 255.0
    data = data.to(device)
    
    target = data[:,:,10:]  
    return data, target
