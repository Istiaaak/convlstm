import os
import numpy as np
import torch
from torch.optim import Adam, RMSprop
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import Seq2Seq
from utils import Patchify, Unpatchify, collate_train_val

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(model, loader, optimizer, criterion, unpatcher):
    model.train()
    running_loss = 0
    for input_patch, target in loader:
        output_patch = model(input_patch)
        output = unpatcher(output_patch.unsqueeze(2))
        loss = criterion(output.flatten(), target.flatten())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(loader.dataset)

def val_one_epoch(model, loader, criterion, unpatcher):
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for input_patch, target in loader:
            output_patch = model(input_patch)
            output = unpatcher(output_patch.unsqueeze(2))
            loss = criterion(output.flatten(), target.flatten())
            running_loss += loss.item()
    return running_loss / len(loader.dataset)

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.stop = False
        self.best_state_dict = None

    def update(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

    def restore_weights(self, model):
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict, strict=True)


def main(save):
    moving_mnist = np.load('./dataset/mnist_test_seq.npy').transpose(1,0,2,3)
    train_data = moving_mnist[:8000]
    val_data   = moving_mnist[8000:9000]

    train_loader = DataLoader(
        train_data, batch_size=16, shuffle=True,
        collate_fn=lambda b: collate_train_val(b, device=device)
    )
    val_loader = DataLoader(
        val_data, batch_size=16, shuffle=True,
        collate_fn=lambda b: collate_train_val(b, device=device)
    )

    architectures = [
        {
            "name": "(5x5)-5x5-128-5x5-64-5x5-64",
            "hidden_dims": [128, 64, 64],
            "kernel_sizes": [(5,5), (5,5), (5,5)]
        }
    ]


    hyperparams_list = [
        {
            "optim": "Adam",
            "lr": 1e-5,
            "weight_decay":2e-5,
            "dropout": 0
        }
    ]

    n_epochs = 200
    results = []

    patcher   = Patchify(patch_size=4).to(device)
    unpatcher = Unpatchify(patch_size=4, out_size=64).to(device)

    os.makedirs("run_finale", exist_ok=True)

    for arch in architectures:
        arch_name    = arch["name"]
        hidden_dims  = arch["hidden_dims"]
        kernel_sizes = arch["kernel_sizes"]

        for hp in hyperparams_list:
            config_name = f"{arch_name}_{hp['optim']}_lr{hp['lr']}_drop{hp['dropout']}_wd{hp['weight_decay']}"
            print(f"\n=== Lancement config: {config_name} ===")

            model = Seq2Seq(
                num_channels=16,
                hidden_dims=hidden_dims,
                kernel_sizes=kernel_sizes,
                frame_size=(16,16),
                dropout=hp["dropout"]
            ).to(device)

            if hp["optim"].lower() == "adam":
                optimizer = Adam(   
                    model.parameters(),
                    lr=hp["lr"],
                    weight_decay=hp["weight_decay"]
                )
            else:
                optimizer = RMSprop(
                    model.parameters(),
                    lr=hp["lr"],
                    weight_decay=hp["weight_decay"]
                )

            criterion = torch.nn.BCELoss(reduction='sum')

            log_dir = os.path.join("run_finale", config_name)
            writer = SummaryWriter(log_dir=log_dir)

            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )

            early_stopper = EarlyStopping(patience=10, min_delta=0.0)

            for epoch in range(1, n_epochs+1):
                train_loss = train_one_epoch(model, train_loader, optimizer, criterion, unpatcher)
                val_loss   = val_one_epoch(model,   val_loader,   criterion, unpatcher)

                print(f"  Epoch {epoch}/{n_epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

                writer.add_scalar("Loss/Train", train_loss, epoch)
                writer.add_scalar("Loss/Val",   val_loss,   epoch)


                scheduler.step(val_loss)


                early_stopper.update(val_loss, model)
                if early_stopper.stop:
                    print("EARLY STOPPING déclenché.")
                    break


            early_stopper.restore_weights(model)

            writer.close()
            if save == True:

                model_filename = f"model_{config_name}.pth"
                torch.save(model.state_dict(), model_filename)
                print(f"Modèle sauvegardé sous {model_filename}")

            final_train_loss = train_loss
            final_val_loss   = val_loss
            results.append({
                "arch": arch_name,
                "optim": hp["optim"],
                "lr": hp["lr"],
                "dropout": hp["dropout"],
                "weight_decay": hp["weight_decay"],
                "train_loss": final_train_loss,
                "val_loss": final_val_loss
            })

    for r in results:
        print(f" - Arch={r['arch']} | {r['optim']} lr={r['lr']} drop={r['dropout']}, wd={r['weight_decay']}"
              f" => train_loss={r['train_loss']:.4f}, val_loss={r['val_loss']:.4f}")



if __name__ == "__main__":
    main(save=True)
