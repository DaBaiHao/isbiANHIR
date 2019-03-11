import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError

transform = transforms.Compose(
    [transforms.Resize([227,227]),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
data = torchvision.datasets.ImageFolder(root="images/breast_1/",
                                        transform=transform )

loader = torch.utils.data.DataLoader(data, batch_size=3,shuffle=True, num_workers=2)
for epoch in range(2):
    for i, data in enumerate(loader, 0):
         inputs, labels = data

