import matplotlib.pyplot as plt
import os

def plot_curves(train_loss, val_loss, model_name):

    save_dir = "outputs/plots"
    os.makedirs(save_dir, exist_ok=True)

    plt.figure()

    plt.plot(train_loss, label="train_loss")
    plt.plot(val_loss, label="val_loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")

    plt.legend()
    plt.title(f"{model_name} training loss curve")

    plt.savefig(f"{save_dir}/{model_name}_loss.png")

    plt.close()
