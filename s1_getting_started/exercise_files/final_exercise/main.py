import argparse
import sys

from matplotlib import pyplot as plt


from torch import nn, optim

import torch
from torch._C import Block

from data import mnist
from model import MyAwesomeModel
from sklearn.model_selection import KFold


class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.1)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement training loop here
        model = MyAwesomeModel()
        train_images, train_labels, _ = mnist()
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.003)
        epochs = 1
        steps = 0

        for e in range(epochs):
            tot_train_loss = 0
            for images, labels in zip(train_images, train_labels):
                optimizer.zero_grad()
                
                
                log_ps = model.forward(images.float().unsqueeze(dim=2).unsqueeze(dim=3))
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()
                
                tot_train_loss += loss.item()
                
            print("loss: ", tot_train_loss / len(train_images))
        
        torch.save(model, 'trained_model_cnn.pt')
        
    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement evaluation logic here
        model = torch.load(args.load_model_from)
        _, _, test = mnist()
        
        criterion = nn.NLLLoss()
        
        test_images = torch.tensor(test['images'])
        test_labels = torch.tensor(test['labels'])
        
        train_losses, test_losses = [], []
        
        # new_test_images = torch.flatten(test_images, start_dim=1, end_dim=2)
        
        tot_test_loss = 0
        accuracy = 0
        equals = []
        with torch.no_grad():
            for images, labels in zip(test_images, test_labels):
                log_ps = model.forward(images.float().unsqueeze(dim=2).unsqueeze(dim=3))        
                loss = criterion(log_ps,labels)
                # print(loss)
                tot_test_loss += loss.item()
                # print(loss.item())
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=0)
                equals.append(top_class == labels.view(*top_class.shape))
        
        equals = torch.tensor(equals)
        test_loss = tot_test_loss / len(test_images)
        
        accuracy = torch.mean(equals.type(torch.FloatTensor))
        ## TODO: Implement the validation pass and print out the validation accuracy
        print(f'Accuracy: {accuracy.item()*100}%')
        print(f'Test Loss: {test_loss}')

if __name__ == '__main__':
    TrainOREvaluate()
    
    
    
    
    
    
    
    
    