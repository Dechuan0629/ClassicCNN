import  torch
import  os, glob
import  random, csv
from    torch.utils.data import Dataset, DataLoader
from    torchvision import transforms
from    PIL import Image
import cv2



class Driver(Dataset):
    def __init__(self, root):
        super(Driver, self).__init__()

        self.root = root

        self.name2label = {} # "sq...":0
        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root, name)):
                continue
            self.name2label[name] = len(self.name2label.keys())

        # print(self.name2label)
        # image, label
        self.images, self.labels = self.load_csv('images.csv')

    def load_csv(self, filename):
        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            for name in self.name2label.keys():
                # 'pokemon\\mewtwo\\00001.png
                images += glob.glob(os.path.join(self.root, name, '*.png'))
                images += glob.glob(os.path.join(self.root, name, '*.jpg'))
                images += glob.glob(os.path.join(self.root, name, '*.jpeg'))

            # 1167, 'pokemon\\bulbasaur\\00000000.png'

            random.shuffle(images)
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images: # 'pokemon\\bulbasaur\\00000000.png'
                    name = img.split(os.sep)[-2][-2:]
                    label = self.name2label[name]
                    # 'pokemon\\bulbasaur\\00000000.png', 0
                    writer.writerow([img, label])

        # read from csv file
        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                # 'pokemon\\bulbasaur\\00000000.png', 0
                img, label = row
                label = int(label)

                images.append(img)
                labels.append(label)

        assert len(images) == len(labels)

        return images, labels


    def __len__(self):

        return len(self.images)

    def __getitem__(self, idx):
        # idx~[0~len(images)]
        # self.images, self.labels
        # img: 'pokemon\\bulbasaur\\00000000.png'
        # label: 0
        img, label = self.images[idx], self.labels[idx]
        img_init = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        img_init = list(cv2.resize(img_init,(320,240)))
        tf = transforms.Compose([
            lambda x:Image.open(x).convert('L'), # string path= > image data
            transforms.FiveCrop((240,320)),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        ])
        img = tf(img)
        img_left_up = img[0]
        img_right_down = img[3]
        img = [[],[],[]]
        img[0] = img_init
        img[1] = img_left_up[0]
        img[2] = img_right_down[0]
        img = torch.tensor(img)
        tf_1 = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        label = torch.tensor(label)

        return img, label
