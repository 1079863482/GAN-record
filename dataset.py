import os
import torch.utils.data as data
from torch.utils.data import DataLoader
import PIL.Image as Image
import numpy as np
import torchvision.transforms as transforms

class trainDataset(data.Dataset):
    def __init__(self,train_path):
        self.img_path = []
        for name in os.listdir(train_path):
            img_path = os.path.join(train_path,name)
            self.img_path.append(img_path)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5,),(0.5,))
        ])
    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        img_path = self.img_path[index]
        img_data = np.array(Image.open(img_path))
        img_data = self.transform(img_data)
        return img_data

if __name__=="__main__":
    train_path = r"E:\faces"
    traindata = trainDataset(train_path)
    train = DataLoader(traindata,batch_size=10,shuffle=True)
    for x in train:
        print(x)

