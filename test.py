import  torch
from    torch.utils.data import DataLoader
from    torchvision import datasets
from    torchvision import transforms
from    torch import nn, optim
from torch.utils.data import Dataset
from    lenet5 import Lenet5
from resnet import ResNet18
import os
from PIL import Image
from tqdm import  tqdm
import time
from torchvision.models import resnet152,vgg16
from utils import Flatten
from AlexNet import AlexNet
from VGG16 import VGG16



path = 'D://驾驶行为//imgs//test_set//'
device = torch.device('cuda')
# model = ResNet18().to(device)
# model1 = VGG16().to(device)
trained_model1 = resnet152()
model1 = nn.Sequential(*list(trained_model1.children())[0:-1],
                     Flatten(),
                      nn.Linear(2048, 10)
                     ).to(device)

trained_model = vgg16()
trained_model.classifier = nn.Sequential(
    nn.Linear(in_features=25088, out_features=4096, bias=True),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5, inplace=False),
    nn.Linear(in_features=4096, out_features=4096, bias=True),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5, inplace=False),
    nn.Linear(in_features=4096, out_features=10, bias=True)
)
model = trained_model.to(device)
# class TestLoad(Dataset):
#
#     def __init__(self, root, resize):
#         super(TestLoad, self).__init__()
#
#         self.root = root
#         self.resize = resize
#         imgs = []
#         self.imgs_name = []
#         for name in os.listdir(os.path.join(root)):
#             imgs.append(root+name)
#             self.imgs_name.append(name)
#         self.images = imgs
#     def __len__(self):
#
#         return len(self.images)
#
#     def __getitem__(self, idx):
#         # idx~[0~len(images)]
#         # self.images, self.labels
#         # img: 'pokemon\\bulbasaur\\00000000.png'
#         # label: 0
#         img= self.images[idx]
#         name = self.imgs_name[idx]
#         tf = transforms.Compose([
#             lambda x:Image.open(x).convert('RGB'), # string path= > image data
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
#         ])
#
#         img = tf(img)
#
#         return img,name

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
    import visdom
    batch_size = 32
    test = datasets.ImageFolder(path, transform=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ]))
    img_test = DataLoader(test,shuffle=False,batch_size=8)
    # model.load_state_dict(torch.load('best_checkpoint_transfered_vgg16_L2.model'))
    model1.load_state_dict(torch.load('best_checkpoint_transfered_resnet152.model'))
    criteon = nn.CrossEntropyLoss().to(device)
    count = 0
    loss = 0
    with torch.no_grad():
        total_correct = 0
        total_num = 0
        # model.eval()
        model1.eval()
        for x,label in tqdm(img_test):
            x, label = x.to(device), label.to(device)

            # logits = model(x)
            logits1 = model1(x)
            # em_logits = 0.62*logits1+0.38*logits

            pred = logits1.argmax(dim=1)
            loss += criteon(logits1,label)
            count+=1
            correct = torch.eq(pred, label).float().sum().item()
            total_correct+=correct
            total_num+=x.size(0)
        print("acc:",total_correct/total_num);
        print("loss:",loss.item()/count)


if __name__ == '__main__':
    main()

