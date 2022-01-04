import os
from datetime import datetime
import torch
import matplotlib.pyplot as plt
import numpy as np

from torch import nn, optim
from torch.autograd import Variable
# from torch.utils.data.dataset import TensorDataset
from torch.utils.data import TensorDataset, DataLoader


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

def unpack_npz(dir, batch_size = 64, shuffle = True):
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
    
    tmp_dataset = TensorDataset(images, labels)
    tmp_dataloader = DataLoader(tmp_dataset, batch_size=batch_size, shuffle=shuffle)
    print("Done unpacking")
    return tmp_dataloader

def test_network(net, trainloader):

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # Create Variables for the inputs and targets
    inputs = Variable(images)
    targets = Variable(images)

    # Clear the gradients from all Variables
    optimizer.zero_grad()

    # Forward pass, then backward pass, then update weights
    output = net.forward(inputs)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()

    return True


def imshow(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax

def view_recon(img, recon):
    ''' Function for displaying an image (as a PyTorch Tensor) and its
        reconstruction also a PyTorch Tensor
    '''

    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)
    axes[0].imshow(img.numpy().squeeze())
    axes[1].imshow(recon.data.numpy().squeeze())
    for ax in axes:
        ax.axis('off')
        ax.set_adjustable('box-forced')

def view_classify(img, ps, version="MNIST"):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    if version == "MNIST":
        ax2.set_yticklabels(np.arange(10))
    elif version == "Fashion":
        ax2.set_yticklabels(['T-shirt/top',
                            'Trouser',
                            'Pullover',
                            'Dress',
                            'Coat',
                            'Sandal',
                            'Shirt',
                            'Sneaker',
                            'Bag',
                            'Ankle Boot'], size='small');
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()
    plt.show()
