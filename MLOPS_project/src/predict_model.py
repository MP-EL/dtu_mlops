import argparse
import sys

import torch
from torch import nn
import numpy as np
from util import unpack_npz, view_classify

Show_pics = False

def evaluate():
    print("Evaluating until hitting the ceiling")
    parser = argparse.ArgumentParser(description='Training arguments')
    parser.add_argument('--model', type=str, help="link to the model")
    parser.add_argument('--images', type=str, help="link to images used for testing")
    # add any additional argument that you want
    args = parser.parse_args()
    print(args)
    
    # TODO: Implement evaluation logic here
    model = torch.load(args.model)
    # _, _, test = mnist()
    
    criterion = nn.NLLLoss()
    
    test_loader = unpack_npz("/" + args.images)
    
    # new_test_images = torch.flatten(test_images, start_dim=1, end_dim=2)
    
    tot_test_loss = 0
    accuracy = 0
    accuracy_tot = []
    i = 0
    with torch.no_grad():
        for images, labels in test_loader:
            log_ps = model.forward(images.float().unsqueeze(dim=1))        
            loss = criterion(log_ps,labels)
            # print(loss)
            tot_test_loss += loss.item()
            # print(loss.item())
            ps = torch.exp(log_ps)
            # print(ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)    
            i += 1
            if (i % 500 == 0) and (Show_pics is True):
                view_classify(images.view(1,28,28),ps)
                
            accuracy = torch.mean(equals.type(torch.FloatTensor))
            print(f'Accuracy: {accuracy.item()*100:.2f}%')
            accuracy_tot.append(accuracy)
    
    test_loss = tot_test_loss / len(test_loader)
    
    
    ## TODO: Implement the validation pass and print out the validation accuracy
    print(f'Total accuracy: {np.mean(accuracy_tot)*100:.2f}%')
    print(f'Test Loss: {test_loss:.2f}')
    
if __name__ == '__main__':
    evaluate()