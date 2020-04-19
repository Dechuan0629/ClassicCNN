import  torch
from    torch import nn
from    torch.nn import functional as F



class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.conv_unit = nn.Sequential(
            # x: [b, 3, 227, 227] => [b, 96, 55, 55]
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0), #[b,96,55,55] =>[b,96,27,27]
            # x: [b, 96, 27, 27] => [b, 256, 27, 27]
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0), #[b,256,27,27] =>[b,256,13,13]
            # x: [b, 256, 13, 13] => [b, 384, 13, 13]
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # x: [b, 384, 13, 13] => [b, 384, 13, 13]
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # x: [b, 384, 13, 13] => [b, 256, 13, 13]
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0)#[b,256,13,13] =>[b,256,6,6]
            #
        )
        # fc unit
        self.fc_unit = nn.Sequential(
            nn.Linear(256*6*6, 4096),
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

        :param x: [b, 3, 227, 227]
        :return:
        """
        batchsz = x.size(0)
        # [b, 3, 227, 227] => [b, 256, 6, 6]
        x = self.conv_unit(x)
        # [b, 256, 6, 6] => [b, 256*6*6]
        x = x.view(batchsz,256*6*6)
        # [b, 256*6*6] => [b, 10]
        logits = self.fc_unit(x)

        # # [b, 10]
        # pred = F.softmax(logits, dim=1)
        # loss = self.criteon(logits, y)

        return logits


