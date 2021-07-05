import matplotlib.patches as patches
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import json
import os



def exceeds(from_x, to_x, from_y, to_y, H, W):
    """Check whether the extracted patch will exceed
    the boundaries of the image of size `T`.
    """
    if (from_x < 0) or (from_y < 0) or (to_x > H) or (to_y > W):
        return True
    return False


def fix(from_x, to_x, from_y, to_y, H, W, size):

    """
    Check whether the extracted patch will exceed
    the boundaries of the image of size `T`.
    If it will exceed, make a list of the offending reasons and fix them
    """

    offenders = []

    if (from_x < 0):
        offenders.append("negative x")
    if from_y < 0:
        offenders.append("negative y")
    if from_x > H:
        offenders.append("from_x exceeds h")            
    if to_x > H:
        offenders.append("to_x exceeds h")
    if from_y > W:
        offenders.append("from_y exceeds w")
    if to_y > W:
        offenders.append("to_y exceeds w")            


    if ("from_y exceeds w" in offenders) or ("to_y exceeds w" in offenders):
        from_y, to_y = W - size, W

    if ("from_x exceeds h" in offenders) or ("to_x exceeds h" in offenders):
        from_x, to_x = H - size, H     

    elif ("negative x" in offenders):
        from_x, to_x = 0, 0 + size

    elif ("negative y" in offenders):
        from_y, to_y = 0, 0 + size            

    return from_x, to_x, from_y, to_y


def denormalize(dims, coords):
    
    W, H = dims
    x, y = coords[0]

    W = int(0.5 * (x + 1) * W)
    H = int(0.5 * (y + 1) * H)

    return torch.tensor([[W, H]], dtype = torch.long)


def bounding_box(x, y, size, color="w"):
    rect = patches.Rectangle(
        (x, y), size, size, linewidth=1, edgecolor=color, fill=False
    )
    return rect


def reset(hidden_size, batch_size, device):
    h_t = torch.zeros(batch_size, hidden_size, dtype = torch.float, device = device, requires_grad = True)
    l_t = torch.FloatTensor(batch_size, 2).uniform_(-1, 1).to(device)
    l_t.requires_grad = True
    return h_t, l_t


# https://github.com/pytorch/examples/blob/master/imagenet/main.py
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def resize_array(x, size):
    # 3D and 4D tensors allowed only
    assert x.ndim in [3, 4], "Only 3D and 4D Tensors allowed!"

    # 4D Tensor
    if x.ndim == 4:
        res = []
        for i in range(x.shape[0]):
            img = array2img(x[i])
            img = img.resize((size, size))
            img = np.asarray(img, dtype="float32")
            img = np.expand_dims(img, axis=0)
            img /= 255.0
            res.append(img)
        res = np.concatenate(res)
        res = np.expand_dims(res, axis=1)
        return res

    # 3D Tensor
    img = array2img(x)
    img = img.resize((size, size))
    res = np.asarray(img, dtype="float32")
    res = np.expand_dims(res, axis=0)
    res /= 255.0
    return res


def img2array(data_path, desired_size=None, expand=False, view=False):
    """
    Util function for loading RGB image into a numpy array.

    Returns array of shape (1, H, W, C).
    """
    img = Image.open(data_path)
    img = img.convert("RGB")
    if desired_size:
        img = img.resize((desired_size[1], desired_size[0]))
    if view:
        img.show()
    x = np.asarray(img, dtype="float32")
    if expand:
        x = np.expand_dims(x, axis=0)
    x /= 255.0
    return x


def array2img(x):
    """
    Util function for converting anumpy array to a PIL img.

    Returns PIL RGB img.
    """
    x = np.asarray(x)
    x = x + max(-np.min(x), 0)
    x_max = np.max(x)
    if x_max != 0:
        x /= x_max
    x *= 255
    return Image.fromarray(x.astype("uint8"), "RGB")


def plot_images(images, gd_truth):

    images = images.squeeze()
    assert len(images) == len(gd_truth) == 9

    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)

    for i, ax in enumerate(axes.flat):
        # plot the image
        ax.imshow(images[i], cmap="Greys_r")

        xlabel = "{}".format(gd_truth[i])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


def prepare_dirs(config):
    for path in [config.data_dir, config.ckpt_dir, config.logs_dir]:
        if not os.path.exists(path):
            os.makedirs(path)


def save_config(config):
    model_name = "ram_{}_{}x{}_{}".format(
        config.num_glimpses, config.patch_size, config.patch_size, config.glimpse_scale
    )
    filename = model_name + "_params.json"
    param_path = os.path.join(config.ckpt_dir, filename)

    print("[*] Model Checkpoint Dir: {}".format(config.ckpt_dir))
    print("[*] Param Path: {}".format(param_path))

    with open(param_path, "w") as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)

        
        
def load_inputs(impath):
    to_tens = transforms.ToTensor()
    return to_tens(Image.open(impath).convert('RGB')).unsqueeze(0)


def get_png_names(directory):
    images = []
    for i in os.listdir(directory):
        try:
            if os.path.isdir(os.path.join(directory, i)):
                new_path = os.path.join(directory, i, "pngs")
                image = os.listdir(new_path)[0]
                images.append(os.path.join(directory, i, "pngs", image))
        except:
            pass
    return images


def get_ys(image_names, mig_data):
    y_class, y_mig = [], []
    for i in image_names:
        dta = mig_data[mig_data["muni_id"] == i.split("/")[5]]
        if len(dta) != 0:
            y_class.append(dta['class'].values[0])
            y_mig.append(dta['num_migrants'].values[0])
    return y_class, y_mig