import torch
from torch.utils.data import DataLoader, Dataset
import glob
import os
from PIL import Image

from torchvision import models, transforms


class MyDataSet(Dataset):

    def __init__(self, root_dir='./data', train_val='train', transform=None):

        self.data_path = os.path.join(root_dir, train_val)
        self.image_names = glob.glob(self.data_path + '/*/*.jpg')
        self.data_transform = transform
        self.train_val = train_val
        #print(self.image_names[0])

    def __len__(self):

        return(len(self.image_names))

    def __getitem__(self, item):
        img_path = self.image_names[item]
        #print(img_path)
        img = Image.open(img_path)
        # print(img.size)
        image = img
        label = img_path.split('/')[-2]
        label = int(label)

        #label = torch.Tensor(label)


        if self.data_transform is not None:
            try:
                image = self.data_path
                image = self.data_transform[self.train_val](img)
            except:
                print('can not load image:{}'.format(img_path))
        return image, label

if __name__ == '__main__':

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            # transforms.Scale(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    image_datasets = {x: MyDataSet(
        root_dir='/home/yinliang/works/pytorch_learn/PK/data',
        train_val=x,
        transform=data_transforms
    ) for x in ['train', 'val']}

    # img, label = image_datasets['train'].__getitem__(13000)
    # print(img, label)

    dataloaders = dict()
    dataloaders['train'] = DataLoader(
        image_datasets['train'],
        batch_size=32,
        shuffle=True
    )
    dataloaders['val'] = DataLoader(
        image_datasets['val'],
        batch_size=32,
        shuffle=True
    )
    # dataloaders = {x: DataLoader(
    #     image_datasets[x],
    #     batch_size=32,
    #     shuffle=True
    # ) for x in ['train', 'val']
    # }

    data1 = iter(dataloaders['val'])

    for i in range(1):
        print(next(data1))

