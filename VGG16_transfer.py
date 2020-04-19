import  torch
from    torch.utils.data import DataLoader
from    torchvision import datasets
from    torchvision import transforms
from    torch import nn, optim
import numpy as np
from torchvision.models import  vgg16_bn
import visdom



def main():
    path = 'D://驾驶行为//imgs//train//'
    path1 = 'D://驾驶行为//imgs//test_set//'
    batch_size = 16
    viz = visdom.Visdom()

    cifar_train = DataLoader(datasets.ImageFolder(path, transform=transforms.Compose([
                             transforms.Resize((224,224)),
                             transforms.RandomCrop((180,180)),
                             transforms.RandomRotation(20),
                             transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
                             transforms.RandomGrayscale(0.4),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])
                            ])),
                             shuffle=True,
                             batch_size=batch_size)

    val_train = DataLoader(datasets.ImageFolder(path, transform=transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
                             ])),
                            shuffle=False,
                            batch_size=batch_size,
                            num_workers=8)

    cifar_test = DataLoader(datasets.ImageFolder(path1, transform=transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
                             ])),
                            shuffle=False,
                            batch_size=batch_size,
                            num_workers=8)


    device = torch.device('cuda')

    trained_model = vgg16_bn(pretrained=True)
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
    print(model)
    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.00001,weight_decay=0.0005)
    best_acc,best_epoch = 0,0
    # model.load_state_dict(torch.load('best_checkpoint_transfered_vgg16_L2-77.model'))
    global_step = 0
    for epoch in range(20):
        if (epoch+1)%5 == 0:
            for p in optimizer.param_groups:
                p['lr'] *= 0.1
        model.train()
        total_loss = 0
        for batchidx, (x, label) in enumerate(cifar_train):
            # [b, 3, 32, 32]
            # [b]
            x,lable = x.to(device),label.to(device)

            logits = model(x)
            # logits: [b, 10]
            # label:  [b]
            # loss: tensor scalar
            loss = criteon(logits, lable)
            total_loss+=loss.item()
            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([loss.item()], [global_step], win='loss', update='append')
            global_step+=1

        print(epoch, 'loss:', total_loss/len(cifar_train))
        model.eval()
        with torch.no_grad():
            # test
            total_correct = 0
            total_num = 0
            for x, label in val_train:
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
            torch.save(model.state_dict(),'temp/best_checkpoint_transfered_vgg16_bn-epoch-'+str(epoch)+'-.model')
        print(epoch, 'test acc:', acc)
        print('best epoch',best_epoch,'best acc',best_acc)


if __name__ == '__main__':
    main()
