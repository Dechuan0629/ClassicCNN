import  torch
from    torch import nn
from    torch.nn import functional as F


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        self.conv_unit = nn.Sequential(
            # x: [b, 3, 224, 224] => [b, 64, 224, 224]
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0), #[b,64,224,224] =>[b,64,112,112]
            # x: [b, 64 ,112, 112] => [b, 128, 112, 112]
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0), #[b,128,112,112] =>[b,128,56,56]
            # x: [b, 128, 56, 56] => [b, 256, 56, 56]
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # [b,256,56,56] =>[b,256,28,28]
            # x: [b, 256, 28, 28] => [b, 512, 28, 28]
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # [b,512,28,28] =>[b,512,14,14]
            # x: [b, 512, 14, 14] => [b, 512, 14, 14]
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)#[b,512,14,14] =>[b,512,7,7]
            #
        )
        # fc unit
        self.fc_unit = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096,10)
        )

        # # use Cross Entropy Loss
        # self.criteon = nn.CrossEntropyLoss()

    def forward(self, x):
        """

        :param x: [b, 3, 224, 224]
        :return:
        """
        batchsz = x.size(0)
        # [b, 3, 224, 224] => [b, 512, 7, 7]
        x = self.conv_unit(x)
        # [b, 512, 7, 7] => [b, 512*7*7]
        x = x.view(batchsz,512*7*7)
        # [b, 512*7*7] => [b, 10]
        logits = self.fc_unit(x)

        # # [b, 10]
        # pred = F.softmax(logits, dim=1)
        # loss = self.criteon(logits, y)

        return logits
