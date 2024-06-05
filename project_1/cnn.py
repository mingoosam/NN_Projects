import torch
import torch.nn as nn #oop
import torch.nn.functional as F #functions
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt

#from sklearn import datasets
from tqdm import tqdm

import sklearn.metrics as sk_m
from sklearn.metrics import plot_confusion_matrix

from IPython import embed

import os
import pickle
import shutil

class Pytorch_Dataset(torch.utils.data.Dataset):
    
    def __init__(self, data, transforms):
        
        super(Pytorch_Dataset, self).__init__()
        
        self.data = data
        self.transforms = transforms
        
        self.create_dataset()
        
    def create_dataset(self):
        
        self.dataset = []
        for sample, label in zip(self.data.samples, self.data.labels):
            self.dataset.append([sample, label])

    def transform_samples(self, samples):
        
        return self.transforms(samples)
    
    def transform_labels(self, labels):
        
        return torch.tensor(labels)
        
    def __getitem__(self, index):
        
        # do transforms here :)
        
        sample, label = self.dataset[index]
        
        if(not(isinstance(sample, torch.Tensor))):
            sample = self.transform_samples(sample).float()
            
        if(not(isinstance(label, torch.Tensor))):
            label = self.transform_labels(label).float()
        
        return sample, label                            
        
    def __len__(self):
        
        return len(self.dataset)

def create_folder(path):
    
    if(os.path.exists(path)):  # if the path already exits
        #shutil.rmtree(path)  # remove the folder
        return
    else:
        os.makedirs(path) # create the folder
        
class Dataset:
    
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

def one_hot_encode(all_labels, num_classes):
    
    ohv_labels = np.zeros((len(all_labels), num_classes))
    
    for i, current_label in enumerate(all_labels):
        ohv_labels[i][current_label] = 1
        
    return ohv_labels
    
def get_dataset(which_dataset):
    
    if(which_dataset == "mnist".lower()):
        
        train = datasets.MNIST(root='../DATA', train=True, download=True)
        test = datasets.MNIST(root='../DATA', train=False, download=True)
        
        train_idx = ((train.targets == 1) + (train.targets == 8)).nonzero().flatten()
        test_idx = ((test.targets == 1) + (test.targets == 8)).nonzero().flatten()
        
        train.data, train.targets = train.data[train_idx], train.targets[train_idx]
        test.data, test.targets = test.data[test_idx], test.targets[test_idx]
        
    if(which_dataset == "fashion".lower()):
        
        train = datasets.FashionMNIST(root='../DATA', train=True, download=True)
        test = datasets.FashionMNIST(root='../DATA', train=False, download=True)
    
    train.data, test.data = train.data.numpy(), test.data.numpy()
    train.targets, test.targets = train.targets.numpy(), test.targets.numpy()
    
    # use one hot encoding for MSE - turn off for torch.nn.CrossEnropy()
    train.targets = one_hot_encode(train.targets, 10)
    test.targets = one_hot_encode(test.targets, 10)
    
    train, test = Dataset(train.data, train.targets), Dataset(test.data, test.targets)
    
    sample_transforms = transforms.Compose([transforms.ToTensor(), 
                                           transforms.Normalize((0.5), (0.5))])
    
    train, test = Pytorch_Dataset(train, sample_transforms), Pytorch_Dataset(test, sample_transforms)
    
    # batch_size: how many samples at a time we pass to the model (generally 8 to 64). helps us generalize (each time it optimizes, optimizations that "stick around" tend to be general, rather than overfit, features). also helps training time
    # shuffle: helps with generalization. don't want to learn all 1s, then all 2s, and so on.
    train = torch.utils.data.DataLoader(train, batch_size=16, shuffle=True)
    test = torch.utils.data.DataLoader(test, batch_size=1, shuffle=False)
                                           
    return train, test

class Net(torch.nn.Module): #inherits from nn.Module
    
    def __init__(self, lr, num_features, num_classes, loss, network, kernel_size, num_filters):
        
        super(Net, self).__init__() #initialize nn.Module
        
        self.alpha = lr
        self.num_features = num_features
        self.num_classes = num_classes
        self.loss_choice = loss
        self.net = network
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        #embed()
        # MLP
        if(self.net.lower() == "mlp"):
            #embed()
            self.network = torch.nn.Sequential(torch.nn.Linear(num_features, 512),
                                               torch.nn.ReLU(),
                                               torch.nn.Linear(512, 256),
                                               torch.nn.ReLU(),
                                               torch.nn.Linear(256, 128),
                                               torch.nn.ReLU(),
                                               torch.nn.Linear(128, 64),
                                               torch.nn.ReLU(),
                                               torch.nn.Linear(64, num_classes))
            
#             self.network = torch.nn.Sequential(torch.nn.Linear(num_features, 64),
#                                                torch.nn.ReLU(),
#                                                torch.nn.Linear(64, 64),
#                                                torch.nn.ReLU(),
#                                                torch.nn.Linear(64, 32),
#                                                torch.nn.ReLU(),
#                                                torch.nn.Linear(32, 16),
#                                                torch.nn.ReLU(),
#                                                torch.nn.Linear(16, num_classes))

        # CNN
        if(self.net.lower() == "cnn"):

            self.extract = torch.nn.Sequential(nn.Conv2d(in_channels = 1, out_channels = 2, kernel_size = kernel_size, stride = 1, padding = 1),
                                               nn.ReLU(inplace=True),
                                               nn.MaxPool2d(2),
                                               nn.Conv2d(in_channels = 2, out_channels = 4, kernel_size = kernel_size, stride = 1, padding = 1),
                                               nn.ReLU(inplace=True),
                                               nn.MaxPool2d(2),
#                                                nn.Conv2d(in_channels = 4, out_channels = 4, kernel_size = kernel_size, stride = 1, padding = 1),
#                                                nn.ReLU(inplace=True),
#                                                nn.MaxPool2d(2)
                                                )
            # in_channels = 1 for grayscale
            # out_channels -> number of shared weights / features
            # kernel size -> nxn kernel size
            # stride -> how many pixels we move at a time
            # padding -> adds 1 pixesl of zeros to each side of each dimension to maintain spatial dimensions for our kernel size
            self.decimate = torch.nn.Sequential(nn.Linear(4 * (7*7), 12),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(12, num_classes)
                                                )
        
        # RBFN
        
        
    def init_optimizer(self):
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr = self.alpha)
        
        #self.optimizer = torch.optim.SGD(self.parameters(), lr = self.alpha)

    def objective(self, preds, labels): # this is the loss function
        
        #preds = F.log_softmax(preds, dim = 1) # dim 1: distribute across output layer of tensors. like np's axis param
    
        loss = torch.nn.MSELoss()
        
        return loss(preds, labels)
    
    def forward(self, x): # you can complicate the network in the forward() method.
        
        if(self.net.lower() == "mlp"):
            x = self.network(x)
        
            return x
        
        if(self.net.lower() == 'cnn'):
            features = self.extract(x)
            features = features.view(features.size()[0], -1)
            output = self.decimate(features)
            
            return output
    
def train(model, train_dataset, test_dataset, num_epochs = 50, rate = 5):
    
    training_loss, cf_matrices = [], []

    model.init_optimizer()

    for epoch in range(num_epochs):

        # Train network

        epoch_loss = 0
        
        for i, data, in enumerate(tqdm(train_dataset, desc = "Train Epoch %s" % epoch)):
            
            sample, label = data
            
            # for CNN:
            sample = sample.type('torch.FloatTensor')
            label = label.type('torch.LongTensor')
            
            sample = sample.to("mps")
            label = label.to("mps")
            
            # only for MLP:
            #sample = sample.view(-1, 784)
            
            preds = model(sample)
            
            loss = model.objective(preds, label.float())

            epoch_loss = epoch_loss + loss.item()

            model.optimizer.zero_grad() # zero the gradients after every batch
            
            loss.backward()
            
            model.optimizer.step() # adjust the weights

        epoch_loss = epoch_loss / (i + 1)
        
        print(epoch_loss)

        training_loss.append(epoch_loss)

        # Validate network
        
        if(epoch % rate == 0):
            
            model.eval()
            
            acc = 0
            all_labels, all_preds = [], []
            for i, (sample, label) in enumerate(tqdm(test_dataset, desc = "Test Epoch %s" % epoch)):
                
                #sample = sample.view(-1, 784)
                
                sample = sample.to("mps")

                logits = model(sample)
                pred = torch.argmax(logits)

                label = np.argmax(label.numpy())
                
                all_preds.append(int(pred.detach().cpu().numpy()))
                all_labels.append(label)
                
                
                if(pred == label):
                    acc += 1
                    
            acc = acc / (i + 1)
            
            print("Valid Accuracy %s" % acc)
                
            ##get metrics
            training_metrics = {}
            cf_matrix = sk_m.confusion_matrix(all_labels, all_preds)
            
            #epoch_accuracy = calculate_accuracy(np.asarray(all_preds), np.asarray(all_labels))

            cf_matrices.append(cf_matrix)
            print(f"confusion matrix appended. epoch {epoch}")
            model.train()
            
        training_metrics = {}
        training_metrics["labels"] = all_labels
        training_metrics["preds"] = all_preds
        training_metrics["mats"] = cf_matrices
            
    return training_loss, training_metrics

if __name__ == "__main__":
    which_dataset = "mnist" # "mnist" or "fashion"
    which_network = "mlp"
    
    loss_choice = "mse"
    
    batch_size = 16
    num_features = 784
    num_classes = 10
    
    alpha = 1e-4 
    
    kernel_size = 1
    num_filters = 8
    
    train_dataset, test_dataset = get_dataset(which_dataset)
    total = 0
    counter = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
    
    for data in train_dataset:
        Xs, ys = data
        for y in ys:
            counter[(int(torch.nonzero(y)))] += 1
    
    print(counter)
    
    new_counter = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
    for data in test_dataset:
        Xs, ys = data
        for y in ys:
            new_counter[(int(torch.nonzero(y)))] += 1
    
    print(new_counter)
    
    model = Net(alpha, num_features, num_classes, loss_choice, which_network, kernel_size, num_filters).to("mps")

    train_loss, train_metrics = train(model, train_dataset, test_dataset)

    embed()
