import argparse
import sys

import torch
from torch import nn
import numpy as np
import os
from datetime import datetime

# from ..models.model import MyAwesomeModel
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from util import unpack_npz

# from torchvision.models.feature_extraction import get_graph_node_names
# from torchvision.models.feature_extraction import create_feature_extractor
# from torchvision.models.detection.mask_rcnn import MaskRCNN
# from torchvision.models.detection.backbone_utils import LastLevelMaxPool
# from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork

base_dir = os.getcwd()
print("base_dir: ", base_dir)
activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def visualize():
    print("Evaluating until hitting the ceiling")
    parser = argparse.ArgumentParser(description='Training arguments')
    parser.add_argument('--model', type=str, help="link to the model")
    parser.add_argument('--images', type=str, help="link to images used for testing")
    
    # add any additional argument that you want
    args = parser.parse_args()
    print(args)
    
    model = torch.load(args.model)
    
    criterion = nn.NLLLoss()
    
    test_loader = unpack_npz("/" + args.images)
    
    # model.conv3.register_forward_hook(get_activation("conv3"))
    # output = model(test_images[1].float().unsqueeze(dim=2).unsqueeze(dim=3))
    # aslkdj = (activation["conv3"])
    
    dirname = base_dir + "/reports/figures/" + datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
    os.makedirs(dirname)
    i = 0
    for images, labels in test_loader:
        model.conv3.register_forward_hook(get_activation("conv3"))
        output = model(images[1].float().unsqueeze(dim=0).unsqueeze(dim=1))
        aslkdj = (activation["conv3"])
        
        X_embedded = TSNE(n_components=2, learning_rate="auto", init="random").fit_transform(np.array(aslkdj[0][0]))

        plt.scatter(X_embedded[:,0], X_embedded[:,1], marker='o')
        
        plt.savefig(dirname + "/Training_loss_" + str(i) + ".png")   
        i += 1 

if __name__ == '__main__':
    visualize()