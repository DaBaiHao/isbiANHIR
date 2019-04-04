import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
import pandas as pd
import os
import matplotlib.pyplot as plt
from skimage import io, transform
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



def read_csv_file():


    DATASET_MEDIUM_DIR = 'can_be_train.csv'
    Image.MAX_IMAGE_PIXELS = None

    imgs_dirs = []
    dataset_read_result = pd.read_csv(DATASET_MEDIUM_DIR)
    i = 0
    for each_img_dir, \
        each_landmarks_dir, \
        each_target_image, \
        each_target_landmarks, \
        each_status in zip(dataset_read_result['Source image'],
                           dataset_read_result['Source landmarks'],
                           dataset_read_result['Target image'],
                           dataset_read_result['Target landmarks'],
                           dataset_read_result['status']):
        each_img_dir = 'images/' + each_img_dir
        each_landmarks_dir = 'landmarks/' + each_landmarks_dir
        each_target_image = 'images/' + each_target_image
        each_target_landmarks = 'landmarks/' + each_target_landmarks

        dataset_read_result.set_value(index=i, col='Source image', value=each_img_dir)
        dataset_read_result.set_value(index=i, col='Source landmarks', value=each_landmarks_dir)
        dataset_read_result.set_value(index=i, col='Target image', value=each_target_image)
        dataset_read_result.set_value(index=i, col='Target landmarks', value=each_target_landmarks)

        imgs_dirs.append(each_img_dir)
        i = i + 1

    print(dataset_read_result['Source image'][1])
    print(imgs_dirs[1])
    return dataset_read_result


dataset_read_result = read_csv_file()

# the first 10
source_image_array = dataset_read_result['Source image']
target_image_array = dataset_read_result['Target image']
source_image_landmarks = dataset_read_result['Source landmarks']
target_image_landmarks = dataset_read_result['Target landmarks']

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self):
        # 使用super()方法调用基类的构造器，即nn.Module.__init__(self)
        super(Generator, self).__init__()
        # The first layer
        # Input channels = 3, output channels = 6 ,,5x5 square convolution kernel
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool_1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool_2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        self.reg_1 = torch.nn.Linear(128 * 16 * 16, 700)
        self.reg_1_1 = torch.nn.Linear(700, 70)
        # self.reg_1_2 = torch.nn.Linear(70, 70)

        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2)
        self.bn_deconv1 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.reg_2 = torch.nn.Linear(64 * 17 * 17, 700)
        self.reg_2_1 = torch.nn.Linear(700, 70)
        # self.reg_2_2 = torch.nn.Linear(70, 70)
        # 64 input features, 10 output features for our 10 defined classes
        # self.fc2 = torch.nn.Linear(78, 78)

        ########### end of the first
        self.conv1_y = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn1_y = nn.BatchNorm2d(32)
        self.pool_1_y = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.conv2_y = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2_y = nn.BatchNorm2d(64)
        self.pool_2_y = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.conv3_y = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        self.reg_1_y = torch.nn.Linear(128 * 16 * 16, 70)
        self.reg_1_1_y = torch.nn.Linear(70, 70)

        self.deconv1_y = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2)
        self.bn_deconv1_y = nn.BatchNorm2d(64)
        self.conv4_y = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.bn4_y = nn.BatchNorm2d(64)
        self.reg_2_y = torch.nn.Linear(64 * 17 * 17, 70)
        self.reg_2_1_y = torch.nn.Linear(70, 70)

    def forward(self, x, y):
        # Size changes from (3, height, weight) to (6, height, weight)
        x = F.relu(self.bn1(self.conv1(x)))
        # Size changes from (6, height, weight) to (6, height/2, weight/2)
        x = self.pool_1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool_2(x)
        x = F.relu(self.conv3(x))
        # Reshape data to input to the input layer of the neural net
        # Size changes from (6, height/2, weight/2) to (1, 6*height/2*weight/2)
        # Recall that the -1 infers this dimension from the other given dimension
        x_1 = x.view(-1, 128 * 16 * 16)
        # Computes the activation of the first fully connected layer
        # Size changes from (1, 4608) to (1, 78)
        x_1 = self.reg_1(x_1)
        x_1 = self.reg_1_1(x_1)

        x = F.relu(self.bn_deconv1(self.deconv1(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x_2 = x.view(-1, 64 * 17 * 17)
        x_2 = self.reg_2(x_2)
        x_2 = self.reg_2_1(x_2)
        #### end of the first image

        ### the first guess using the samiliar network
        y = F.relu(self.bn1(self.conv1(y)))
        y = self.pool_1(y)
        y = F.relu(self.bn2(self.conv2(y)))
        y = self.pool_2(y)
        y = F.relu(self.conv3(y))

        y_1 = y.view(-1, 128 * 16 * 16)
        y_1 = self.reg_1(y_1)
        y_1 = self.reg_1_1(y_1)

        y = F.relu(self.bn_deconv1(self.deconv1(y)))
        y = F.relu(self.bn4(self.conv4(y)))
        y_2 = y.view(-1, 64 * 17 * 17)
        y_2 = self.reg_2(y_2)
        y_2 = self.reg_2_1(y_2)

        # x = F.relu(self.fc2(x))
        return x_1, x_2, y_1, y_2


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.reg_1_1 = torch.nn.Linear(70, 700)
        self.reg_1_2 = torch.nn.Linear(70, 700)
        self.reg_2_1 = torch.nn.Linear(70, 700)
        self.reg_2_2 = torch.nn.Linear(70, 700)
        self.cos_1 = nn.CosineSimilarity(eps=1e-6)
        self.cos_2 = nn.CosineSimilarity(eps=1e-6)

        self.dense_1 = torch.nn.Linear(1, 1)
        self.dense_2 = torch.nn.Linear(1, 1)

        self.sigmoid1 = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()
        self.sigmoid3 = nn.Sigmoid()

    def forward(self, x_1, x_2, y_1, y_2):
        x_1 = self.reg_1_1(x_1)
        x_2 = self.reg_1_2(x_2)
        y_1 = self.reg_2_1(y_1)
        y_2 = self.reg_2_2(y_2)
        x = self.cos_1(x_1, x_2)
        y = self.cos_2(y_1, y_2)

        x = self.dense_1(x)
        y = self.dense_2(y)

        # print(x)
        output = self.sigmoid1(x) + self.sigmoid2(y)
        output = self.sigmoid3(output)
        return output


transform_1 = transforms.Compose([transforms.Resize([512, 512]),
                                  transforms.ToTensor()])

loader = torch.utils.data.DataLoader(source_image_array, batch_size=1, shuffle=True, num_workers=2)
generator = Generator()
discriminator = Discriminator()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

optimizerG = optim.Adam(generator.parameters(), lr=1e-4)
optimizerD = optim.Adam(discriminator.parameters(), lr=1e-4)

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

loss = nn.MSELoss()  # MSELoss可换为
criterion = nn.BCELoss()

print("Starting Training Loop...")
for epoch in range(1000):
    print("Begin ", epoch, "epoch")
    for i, data in enumerate(loader, 0):
        # print("Begin ",epoch,"epoch")
        inputs = data
        # source_image = plt.imread(inputs[0])

        ## Train with all-real batch
        discriminator.zero_grad()
        label = torch.full((1,), real_label)

        source_image_landmark = source_image_landmarks[i]
        current = pd.read_csv(source_image_landmark)
        source_X = current['X']
        source_Y = current['Y']
        #         # X = X.transpose()
        source_X = torch.FloatTensor(source_X[:70])
        source_X = source_X.unsqueeze(0)

        source_Y = torch.FloatTensor(source_Y[:70])
        source_Y = source_Y.unsqueeze(0)

        target_image_landmark = target_image_landmarks[i]
        current = pd.read_csv(target_image_landmark)
        target_X = current['X']
        target_Y = current['Y']

        target_X = torch.FloatTensor(target_X[:70])
        target_X = target_X.unsqueeze(0)

        target_Y = torch.FloatTensor(target_Y[:70])
        target_Y = target_Y.unsqueeze(0)

        output = discriminator(source_X, source_Y, target_X, target_Y).view(-1)
        # print(output)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward(retain_graph=True)
        D_x = output.mean().item()

        ## Train with all-fake batch

        source_image = Image.open(inputs[0]).convert('RGB')
        source_image1 = transform_1(source_image)
        image_tensor = source_image1.unsqueeze(0)
        image_tensor = Variable(image_tensor)
        target_image = Image.open(target_image_array[i]).convert('RGB')
        target_image1 = transform_1(target_image)
        target_image_tensor = target_image1.unsqueeze(0)
        target_image_tensor = Variable(target_image_tensor)
        x_1, x_2, y_1, y_2 = generator(image_tensor, target_image_tensor)
        label.fill_(fake_label)

        output = discriminator(x_1, x_2, y_1, y_2).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward(retain_graph=True)
        D_G_z1 = output.mean().item()

        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        generator.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = discriminator(x_1, x_2, y_1, y_2).view(-1)
        errG = criterion(output, label)
        errG.backward(retain_graph=True)
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        print(errG)
