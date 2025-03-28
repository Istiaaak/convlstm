import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import Seq2Seq
from utils import Patchify, Unpatchify, collate_test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(title):
    model = Seq2Seq(
        num_channels=16,
        hidden_dims=[128,64,64],
        kernel_sizes=[(5,5),(5,5),(5,5)],
        frame_size=(16,16),
        dropout=0
    ).to(device)

    weights_path = "model_(5x5)-5x5-128-5x5-64-5x5-64_RMSProp_lr0.0005_drop0_wd0.pth"
    model.load_state_dict(torch.load(weights_path, map_location=device), strict=True)
    print(f"Modèle chargé avec succès depuis {weights_path}")

    model.eval()

    moving_mnist = np.load('./dataset/mnist_test_seq.npy').transpose(1,0,2,3)
    test_data = moving_mnist[9000:10000]
    test_loader = DataLoader(
        test_data, batch_size=2, shuffle=True,
        collate_fn=lambda b: collate_test(b, device=device)
    )

    patcher   = Patchify(patch_size=4).to(device)
    unpatcher = Unpatchify(patch_size=4, out_size=64).to(device)


    data, target = next(iter(test_loader))

    B, _, T, H, W = data.shape
    output_np = np.zeros((B, 10, 64, 64), dtype=np.uint8)

    with torch.no_grad():
        for t in range(10):
            input_ = data[:, :, t:t+10]
            input_patch = patcher(input_.to(device))
            pred_patch = model(input_patch)
            pred = unpatcher(pred_patch.unsqueeze(2))
            pred_bin = (pred > 0.5).float() * 255.0
            output_np[:, t] = pred_bin.squeeze(2).squeeze(1).cpu().numpy().astype(np.uint8)


    i = 0 
    gt_frames = []
    pred_frames = []
    for t in range(10):
        gt = (target[i, 0, t].cpu().numpy() * 255).astype(np.uint8)
        pr = output_np[i, t]
        gt_frames.append(gt)
        pred_frames.append(pr)

    fig, axes = plt.subplots(2, 10, figsize=(20, 4))
    for t in range(10):
        axes[0, t].imshow(gt_frames[t], cmap='gray', vmin=0, vmax=255)
        axes[0, t].set_title(f"Réalité {t+10}")
        axes[0, t].axis("off")
        
        axes[1, t].imshow(pred_frames[t], cmap='gray', vmin=0, vmax=255)
        axes[1, t].set_title(f"Prédiction {t+10}")
        axes[1, t].axis("off")

    plt.tight_layout()
    
    out_dir = "images"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{title}.png")
    plt.savefig(out_path)
    plt.close(fig)

    print(f"Image sauvegardée dans le dossier '{out_dir}'")

if __name__ == "__main__":
    main(title="rmsprop")
