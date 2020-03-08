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
from torchvision.models import resnet152
from utils import Flatten
from AlexNet import AlexNet
from VGG16 import VGG16



path = 'D://驾驶行为//imgs//test//'
device = torch.device('cuda')
model = VGG16().to(device)
# trained_model = resnet152(pretrained=True)
# model = nn.Sequential(*list(trained_model.children())[0:-1],
#                       Flatten(),
#                       nn.Linear(2048, 10)
#                       ).to(device)

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
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        img = tf(img)

        return img,name

def denormalize(x):

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    print(type(x))
    # x_hat = (x-mean)/std
    # x = x_hat*std = mean
    # x: [c, h, w]
    # mean: [3] => [3, 1, 1]
    mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
    std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
    # print(mean.shape, std.shape)
    x = x.cpu() * std + mean

    return x

def img_load():
    tf = transforms.Compose([
        lambda x: Image.open(x).convert('RGB'),  # string path= > image data
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])


def main():
    import visdom
    viz = visdom.Visdom()
    img = TestLoad(path,224)
    img_test = DataLoader(img,shuffle=False,batch_size=1)
    model.load_state_dict(torch.load('best_checkpoint_vgg.model'))
    re = []
    for x,name in tqdm(img_test):
        print(x.shape)
        x= x.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
            pred = pred.cpu().numpy()
            pred = list(pred)
            for i in range(len(pred)):
                if pred[i] == 0:
                    pred[i] = 'safe driving'
                elif pred[i] == 1:
                    pred[i] = 'texting - right'
                elif pred[i] == 2:
                    pred[i] = 'talking on the phone - right'
                elif pred[i] == 3:
                    pred[i] = 'texting - left'
                elif pred[i] == 4:
                    pred[i] = 'talking on the phone - left'
                elif pred[i] == 5:
                    pred[i] = 'operating the radio'
                elif pred[i] == 6:
                    pred[i] = 'drinking'
                elif pred[i] == 7:
                    pred[i] = 'reaching behind'
                elif pred[i] == 8:
                    pred[i] = 'hair and makeup'
                elif pred[i] == 9:
                    pred[i] = 'talking to passenger'

            list3 = dict(zip(name,pred))
            print('1---',list3)
            viz.images(denormalize(x),nrow=1,win='batch',opts=dict(title=pred[0]))

            time.sleep(5)

if __name__ == '__main__':
    main()

