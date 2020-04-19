import  torch
from    torch.utils.data import DataLoader
from    torchvision import datasets
from    torchvision import transforms
from    torch import nn, optim
import numpy
from    lenet5 import Lenet5
#from resnet import ResNet18
from torchvision.models import  resnet152
import visdom
from utils import Flatten
from torch.nn import functional as F


def main():
    path = 'D://驾驶行为//imgs//train//'
    path1 = 'D://驾驶行为//imgs//test_set//'
    batch_size = 8
    viz = visdom.Visdom()
    cifar_train = DataLoader(datasets.ImageFolder(path, transform=transforms.Compose([
                            transforms.Resize((224,224)),
                            transforms.RandomRotation(30),
                             transforms.FiveCrop((172,172)),
                            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                            transforms.Lambda(lambda crops: torch.stack(
                                [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop) for crop in crops])),
                            ])),
                             shuffle=True,
                             batch_size=batch_size)

    val_train = DataLoader(datasets.ImageFolder(path, transform=transforms.Compose([
                            transforms.Resize((224,224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
                             ])),
                            shuffle=False,
                            batch_size=8)

    cifar_test = DataLoader(datasets.ImageFolder(path1, transform=transforms.Compose([
                            transforms.Resize((224,224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
                             ])),
                            shuffle=False,
                            batch_size=8)
    print(len(cifar_train),len(cifar_test))

    device = torch.device('cuda')

    trained_model = resnet152(pretrained=False)
    model = nn.Sequential(*list(trained_model.children())[:-1], #[b, 512, 1, 1]
                          Flatten(), # [b, 512, 1, 1] => [b, 512]
                          nn.Linear(2048, 10)
                          ).to(device)

    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.00001,
        weight_decay=0.0005)
    print(model)
    best_acc,best_epoch = 0,0
    model.load_state_dict(torch.load('temp/best_checkpoint_transfered_resnet152-epoch-14-.model'))
    global_step = 0
    for epoch in range(20):
        if (epoch+1) % 5 == 0:
            for p in optimizer.param_groups:
                p['lr'] *= 0.1
        model.train()
        for batchidx, (x, label) in enumerate(cifar_train):
            # [b, 3, 32, 32]
            # [b]
            bs,crops,c,h,w = x.size()
            x = x.view(-1,c,h,w)
            x, label = x.to(device), label.to(device)

            logits = model(x)
            logits = logits.view(bs,crops,-1).mean(1)
            # logits: [b, 10]
            # label:  [b]
            # loss: tensor scalar
            loss = criteon(logits, label)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([loss.item()], [global_step], win='loss', update='append')
            global_step+=1

        print(epoch, 'loss:', loss.item())
        model.eval()
        with torch.no_grad():
            # test
            total_correct = 0
            total_num = 0
            for x, label in val_train:

                x, label = x.to(device), label.to(device)

                # [b, 10]
                logits = model(x)
                # [b]
                pred = logits.argmax(dim=1)
                # [b] vs [b] => scalar tensor
                correct = torch.eq(pred, label).float().sum().item()
                total_correct += correct
                total_num += x.size(0)
                # print(correct)

            acc = total_correct / total_num
        print("train acc :",acc)

        model.eval()
        with torch.no_grad():
            # test
            total_correct = 0
            total_num = 0
            for x, label in cifar_test:
                # [b, 3, 32, 32]
                # [b]
                x, label = x.to(device), label.to(device)

                # [b, 10]
                logits = model(x)
                # [b]
                pred = logits.argmax(dim=1)
                # [b] vs [b] => scalar tensor
                correct = torch.eq(pred, label).float().sum().item()
                total_correct += correct
                total_num += x.size(0)
                # print(correct)

            acc = total_correct / total_num
            if acc>best_acc:
                best_epoch = epoch
                best_acc = acc
                if epoch == 0:continue
            torch.save(model.state_dict(),'temp/best_checkpoint_transfered_resnet152-epoch-'+str(epoch+15)+'-.model')
        print('epoch:',epoch, 'test acc:', acc)
        print('best epoch;',best_epoch,'best acc:',best_acc)


if __name__ == '__main__':
    main()
