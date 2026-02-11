import matplotlib.pyplot as plt

def show_gradcam(image, heatmap):
    plt.imshow(image.permute(1,2,0))
    plt.imshow(heatmap, alpha=0.5, cmap="jet")
    plt.savefig("outputs/gradcam.png")
