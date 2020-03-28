from torchvision import models, transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
import argparse
from dataset import MyDataSet
import time
import os
import numpy as np

def train_model(model, data_sizes, num_epochs, scheduler, dataloaders,criterion, optimizer, ):
    device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        begin_time = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('*'*20)

        for phase in ['train', 'val']:
            count_batch = 0
            if phase == 'train':
                scheduler.step()
                model.train(True)
            else:
                model.train(False)
            running_loss = 0.0
            running_corrects = 0.0
            for i, data in enumerate(dataloaders[phase]):
                count_batch += 1

                inputs, labels = data
                
                #print(labels)
                #print(inputs, labels)

                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                    #inputs = inputs.cuda()
                    #labels = labels.cuda()
                    #labels = Variable(torch.from_numpy(np.array(labels)).long()).cuda()
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                optimizer.zero_grad()
                outputs = model(inputs)
                out = torch.argmax(outputs.data, 1)
                #print(torch.argmax(outputs.data, 1))
                _, preds = torch.max(outputs.data, 1)

                #print('labels:', labels)
                #print('preds:', preds)
                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                running_loss += loss.data
                running_corrects += torch.sum(preds == labels.data).to(torch.float32)

                if count_batch % 10==0:
                    #print('batch_size * count_batch:', batch_size * count_batch)
                    batch_loss = running_loss / (batch_size * count_batch)
                    batch_acc = running_corrects / (batch_size * count_batch)
                    print('{} Epoch [{}] Batch Loss: {:.4f} Acc:{:.4f} Time: {:.4f}s'.format(
                        phase, epoch, batch_loss, batch_acc, time.time()-begin_time
                    ))
                    begin_time = time.time()
        epoch_loss = running_loss / data_sizes[phase]
        epoch_acc = running_corrects / data_sizes[phase]
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

        if phase == 'train':
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            torch.save(model, os.path.join(model_path, 'resnet_epoch{}.pkl').format(epoch))

        if phase == 'val' and epoch_acc > best_acc:

            best_acc = epoch_acc
            best_model_wts = model.state_dict()

        time_elapsed = time.time() - since
        print('Training completed in {:.0f}mins {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:.4f}'.format(best_acc))

        model.load_state_dict(best_model_wts)
    return(model)







if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type = int)
    parser.add_argument("--model_path", default='model', type=str)
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--lr', default=0.005, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--num_epochs', default=5, type=int)
    parser.add_argument('--step_size', default=2, type=int)
    args = parser.parse_args()


    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            #transforms.Scale(256),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    use_gpu = torch.cuda.is_available()

    batch_size = args.batch_size
    num_classes = args.num_classes
    model_path = args.model_path
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    num_epochs = args.num_epochs

    # load train/val dataset
    image_datasets = {x:MyDataSet(
        root_dir='./data',
        train_val=x,
        transform=data_transforms
    ) for x in ['train', 'val']}

    dataloaders = dict()
    dataloaders['train'] = DataLoader(
        image_datasets['train'],
        batch_size=batch_size,
        shuffle=True
    )
    dataloaders['val'] = DataLoader(
        image_datasets['val'],
        batch_size=batch_size,
        shuffle=True
    )
    # dataloaders = {x:DataLoader(
    #     image_datasets[x],
    #     batch_size=batch_size,
    #     shuffle=True
    # ) for x in ['train', 'val']
    # }
    data_sizes = {x:len(image_datasets[x]) for x in ['train', 'val']}

    #print(len(dataloaders['train']))

    # define the model
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    if use_gpu:
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),lr = 0.005, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5,gamma=0.2)
    model = train_model(model=model,
                        data_sizes = data_sizes,
                        dataloaders=dataloaders,
                        num_epochs = num_epochs,
                        scheduler=exp_lr_scheduler,
                        criterion=criterion,
                        optimizer= optimizer)


    torch.save(model, os.path.join(model_path, 'best_resnet.pkl'))

