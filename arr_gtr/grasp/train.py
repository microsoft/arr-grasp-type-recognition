# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
import copy
import time
import os
from pathlib import Path
from arr_gtr.grasp.grasp_type_utils import grasp_types


class ModelTrainingModule(nn.Module):

    def __init__(self, data_directory, pretrained_model_path=None, batch_size=128):
        super(ModelTrainingModule, self).__init__()

        print('preparing model...')
        model = models.resnet101(pretrained=True)
       
        for param in model.parameters():
            param.requires_grad = False
    
        num_features = model.fc.in_features
        num_class = len(grasp_types)
        model.fc = nn.Linear(num_features, num_class)

        if pretrained_model_path is not None:
            if torch.cuda.is_available():
                model.load_state_dict(torch.load(pretrained_model_path))
            else:
                model.load_state_dict(torch.load(pretrained_model_path, map_location=torch.device('cpu')))

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(self.device)
        self.model = model
        print('done')

        transform_dict = {
                'train': transforms.Compose(
                    [transforms.Resize((256,256)),
                    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomAffine(degrees=0, translate=(10/256, 10/256)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
                    ]),
                'test': transforms.Compose(
                    [transforms.Resize((256,256)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
                    ])}
        data_train = torchvision.datasets.ImageFolder(root=data_directory+'/train', transform=transform_dict["train"])
        data_val = torchvision.datasets.ImageFolder(root=data_directory+'/valid', transform=transform_dict["test"])
        
        train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)
        val_loader   = torch.utils.data.DataLoader(data_val,   batch_size=batch_size, shuffle=False)
        self.dataloaders  = {"train":train_loader, "val":val_loader}
        print(len(data_train))
        print(len(data_val))
        self.train_lossList = []
        self.train_accList = []
        self.val_lossList = []
        self.val_accList = []
        self.best_acc = None
        self.best_model_wts = None

    def train(self, num_epochs=200, save_path='./out/model.pth'):

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        since = time.time()
        finish = 0

        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        self.best_acc = 0.0

        for epoch in range(num_epochs):
            if(finish==3):
                break
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0
                total = 0

                # Iterate over data.
                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    #print(torch.sum(preds==labels.data))
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    total += inputs.size(0)

                print(total)
                epoch_loss = running_loss / total
                epoch_acc = running_corrects.double() / total
                
                if phase == "train":
                    self.train_lossList.append(epoch_loss)
                    self.train_accList.append(epoch_acc)
                else:
                    self.val_lossList.append(epoch_loss)
                    self.val_accList.append(epoch_acc)
                    if len(self.val_accList) >= 3:
                        if max(self.val_accList) >= self.val_accList[-1]:
                            finish += 1
                            if finish == 3:
                                break
                        else:
                            finish = 0
                    
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > self.best_acc:
                    self.best_acc = epoch_acc
                    self.best_model_wts = copy.deepcopy(self.model.state_dict())

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(self.best_acc))

        # load best model weights
        self.model.load_state_dict(self.best_model_wts)
        directory = os.path.dirname(save_path)
        Path(directory).mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), save_path)


    def visualize_result(self):
        fig, axes = plt.subplots(1, 2, tight_layout=True)
        axes[0].grid()
        axes[1].grid()
        l1, = axes[0].plot(self.train_lossList)
        l2, = axes[0].plot(self.val_lossList)
        axes[0].legend([l1, l2],["Train", "Validation"])
        axes[0].set_title("Loss")
        axes[0].set_xlabel('Epoch')
        if torch.cuda.is_available():
            tmp_train_accList = [a.to(torch.device("cpu")) for a in self.train_accList]
            tmp_val_accList = [a.to(torch.device("cpu")) for a in self.val_accList]
            l1, = axes[1].plot(tmp_train_accList)
            l2, = axes[1].plot(tmp_val_accList)
        else:
            l1, = axes[1].plot(self.train_accList)
            l2, = axes[1].plot(self.val_accList)
        axes[1].legend([l1, l2],["Train", "Validation"])
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel('Epoch')
        plt.show()


    def imshow(self):
        # obtain ine iamge
        dataiter = iter(self.dataloaders["train"])
        images, labels = dataiter.next()
        print(' '.join('%5s' % grasp_types[labels[j]] for j in range(8)))
        img = torchvision.utils.make_grid(images)
        img = img.mul(torch.FloatTensor([0.229, 0.224, 0.225]).view(3, 1, 1))
        img = img.add(torch.FloatTensor([0.485, 0.456, 0.406]).view(3, 1, 1))
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()