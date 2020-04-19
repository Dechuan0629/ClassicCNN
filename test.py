import  torch
from    torch.utils.data import DataLoader
from    torchvision import datasets
from    torchvision import transforms
from    torch import nn, optim
from torch.utils.data import Dataset
from    lenet5 import Lenet5
from resnet import ResNet18
import os
from    PIL import Image
import matplotlib.pyplot as plt
from torchvision.models import resnet152,vgg16,alexnet,resnet18,resnet34,densenet161
from torch.nn import functional as F
from AlexNet import AlexNet
from VGG16 import VGG16
import pandas as pd
from    utils import Flatten
from tqdm import tqdm
import numpy as np
from Inceptionv4 import inceptionv4


path = 'F://test//imgs//test//'
device = torch.device('cuda')
# model = ResNet18().to(device)
# model1 = VGG16().to(device)
trained_model = densenet161(pretrained=False)
trained_model.classifier = nn.Linear(in_features=2208, out_features=10, bias=True)

model = trained_model.to(device)

trained_model1 = resnet152(pretrained=False)
trained_model1.fc  = nn.Linear(2048,10,bias=True)

trained_model2 = resnet152(pretrained=False)
model2 = nn.Sequential(*list(trained_model2.children())[:-1],  # [b, 512, 1, 1]
                      Flatten(),  # [b, 512, 1, 1] => [b, 512]
                      nn.Linear(2048, 10)
                      ).to(device)

# trained_model1 = resnet34(pretrained=False)
# trained_model1.fc = nn.Linear(512, 10, bias=True)
#
# trained_model2 = vgg16(pretrained=False)
# trained_model2.classifier = nn.Sequential(
#     nn.Linear(in_features=25088, out_features=4096, bias=True),
#     nn.ReLU(inplace=True),
#     nn.Dropout(p=0.5, inplace=False),
#     nn.Linear(in_features=4096, out_features=4096, bias=True),
#     nn.ReLU(inplace=True),
#     nn.Dropout(p=0.5, inplace=False),
#     nn.Linear(in_features=4096, out_features=10, bias=True)
# )
#
# trained_model3 = alexnet(pretrained=False)
# trained_model3.classifier = nn.Sequential(
#     nn.Dropout(p=0.5, inplace=False),
#     nn.Linear(in_features=9216, out_features=4096, bias=True),
#     nn.ReLU(inplace=True),
#     nn.Dropout(p=0.5, inplace=False),
#     nn.Linear(in_features=4096, out_features=4096, bias=True),
#     nn.ReLU(inplace=True),
#     nn.Linear(in_features=4096, out_features=10, bias=True)
# )
#
# trained_model4 = resnet152(pretrained=False)
# model4 = nn.Sequential(*list(trained_model4.children())[:-1],
#                       Flatten(),
#                       nn.Linear(2048, 10)
#                       ).to(device)


model1 = trained_model1.to(device)
# model1 = trained_model1.to(device)
# model2 = trained_model2.to(device)
# model3 = trained_model3.to(device)

class TestLoad(Dataset):

    def __init__(self, root, resize):
        super(TestLoad, self).__init__()

        self.root = root
        self.resize = resize
        imgs = []
        self.imgs_name = []
        for name in os.listdir(os.path.join(root)):
            imgs.append(root+name)
            self.imgs_name.append(name)
        self.images = imgs
    def __len__(self):

        return len(self.images)

    def __getitem__(self, idx):
        # idx~[0~len(images)]
        # self.images, self.labels
        # img: 'pokemon\\bulbasaur\\00000000.png'
        # label: 0
        img= self.images[idx]
        name = self.imgs_name[idx]
        tf = transforms.Compose([
            lambda x:Image.open(x).convert('RGB'), # string path= > image data
            transforms.Resize((229, 229)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        img = tf(img)

        return img,name

# def denormalize(x):
#
#     mean = [0.485, 0.456, 0.406]
#     std = [0.229, 0.224, 0.225]
#     print(type(x))
#     # x_hat = (x-mean)/std
#     # x = x_hat*std = mean
#     # x: [c, h, w]
#     # mean: [3] => [3, 1, 1]
#     mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
#     std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
#     # print(mean.shape, std.shape)
#     x = x.cpu() * std + mean
#
#     return x
#
# def img_load():
#     tf = transforms.Compose([
#         lambda x: Image.open(x).convert('RGB'),  # string path= > image data
#         transforms.Resize((224, 224)),
#         transforms.ToTensor()
#     ])


def main():
    model.load_state_dict(torch.load('best_checkpoint_transfered_dense161-L2-92.9.model'))

    model1.load_state_dict(torch.load('best_checkpoint_transfered_resnet152-92-L2.model'))
    #
    model2.load_state_dict(torch.load('best_checkpoint_transfered_resnet152-92.model'))
    #
    # model3.load_state_dict(torch.load('best_checkpoint_transfered_alex-81.8.model'))
    #
    # model4.load_state_dict(torch.load('best_checkpoint_transfered_resnet152-92.model'))


    img_test = DataLoader(TestLoad(path,229),shuffle=False,batch_size=1)
    test = pd.DataFrame(columns=['img','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9'])
    model.eval()
    model1.eval()
    model2.eval()
    # model3.eval()
    # model4.eval()
    for i,(x,name) in tqdm(enumerate(img_test)):
        x = x.to(device)
        # temp = []
        # pred = np.zeros(10).tolist()
        logits0  = model(x)
        logits1 = model1(x)
        logits2 = model2(x)
        em_logits = (logits0+logits1+logits2)/3
        # logits3 = model3(x)
        # logits4 = model4(x)
        # temp.append(F.softmax(logits0)[0].argmax().item())
        # temp.append(F.softmax(logits1)[0].argmax().item())
        # temp.append(F.softmax(logits2)[0].argmax().item())
        # temp.append(F.softmax(logits3)[0].argmax().item())
        # temp.append(F.softmax(logits4)[0].argmax().item())
        # pred[max(temp,key=temp.count)] = 1
        pred = F.softmax(em_logits)
        pred = pred.cpu().detach().numpy()[0].tolist()
        test.loc[i] = list(name)+pred
    test.to_csv('test_dense161+resnet152+L2.csv', index=False, sep=',')





if __name__ == '__main__':
    main()

