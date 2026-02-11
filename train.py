import torch
import yaml
import sys
import os
from tqdm import tqdm

from utils.seed import set_seed
from utils.dataloader import get_loaders
from utils.plots import plot_curves
from models.baseline import BaselineCNN
from models.improved import ImprovedCNN

device = "cpu"

config = yaml.safe_load(open("configs/config.yaml"))

set_seed(config["seed"])

train_loader, val_loader, _ = get_loaders(config["batch_size"])

loss_function = torch.nn.CrossEntropyLoss(
    label_smoothing=config["label_smoothing"]
)

def run_training(model, model_name):

    print(f"\nStarting training for {model_name}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    train_loss_list = []
    val_loss_list = []

    for epoch in range(config["epochs"]):

        model.train()
        running_loss = 0

        for x, y in tqdm(train_loader):

            output = model(x)
            loss = loss_function(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss_list.append(running_loss)
        val_loss_list.append(running_loss)

        print(f"{model_name} | epoch {epoch+1} | loss {running_loss:.3f}")

    os.makedirs("outputs/models", exist_ok=True)

    torch.save(model.state_dict(), f"outputs/models/{model_name}.pt")

    plot_curves(train_loss_list, val_loss_list, model_name)

selected = "all"
if len(sys.argv) > 1:
    selected = sys.argv[1]

if selected in ["baseline", "all"]:
    run_training(BaselineCNN(), "baseline")

if selected in ["improved", "all"]:
    run_training(ImprovedCNN(), "improved")
