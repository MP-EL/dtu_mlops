import os
from datetime import datetime
import torch
import matplotlib.pyplot as plt
import numpy as np


base_dir = os.getcwd()
print("base_dir: ", base_dir)

def save_model(model):
    """Saves a machinelearning model to the models folder.

    Args:
        model ([type]): [Model to be saved.
    """
    
    dirname = base_dir + "/models/" + datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
    os.makedirs(dirname)
    torch.save(model, dirname + "/model.pt")
    print(f'Saved model in {dirname}/ as model.pt')

def save_losses(loss, make_image = True, make_txt = True):
    """Saves the losses.

    Args:
        loss ([type]): [description]
        make_image (bool, optional): [description]. Defaults to True.
        make_txt (bool, optional): [description]. Defaults to True.
    """
    
    dirname = base_dir + "/reports/figures/" + datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
    os.makedirs(dirname)
    if make_image is True:
        plt.plot(loss)
        plt.ylabel("Training loss")
        plt.xlabel("Epoch")
        plt.savefig(dirname + "/Training_loss.png")
        print(f'Saved loss figure in {dirname}/ as Training_loss.png')
        
    if make_txt is True:
        with open(dirname + "/loss.txt", "w") as txt_file:
            for line in loss:
                txt_file.write(str(line) + "\n")
        print(f'Saved loss file in {dirname}/ as loss.txt')

def unpack_npz(dir):
    """Unpacking npz files to tensors.

    Args:
        directory ([str]): directory of npz file.

    Returns:
        images ([Tensor]): images from npz file
        labels ([Tensor]): labels from npz file
    """
    
    print("Unpacking npz")
    tmp = np.load(base_dir + dir)
    images = torch.tensor(tmp['images'])
    labels = torch.tensor(tmp['labels'])
    print("Done unpacking")
    return images, labels