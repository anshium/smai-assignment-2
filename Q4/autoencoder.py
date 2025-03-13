import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_fscore_support, roc_curve
from torch.utils.data import DataLoader, Subset

print("Imported Everything!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

# class AutoEncoder(nn.Module):
#     def __init__(self, bottleneck_dim = 32):
#         super().__init__()

#         self.encoder = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(28 * 28, 128),
#             nn.ReLU(),
#             nn.Linear(128, bottleneck_dim),
#             nn.ReLU()
#         )

#         self.decoder = nn.Sequential(
#             nn.Linear(bottleneck_dim, 128),
#             nn.ReLU(128, 28 * 28),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)

#         return x.view(-1, 1, 28, 28)
class AutoEncoder(nn.Module):
    def __init__(self, bottleneck_dim = 32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, bottleneck_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded

# Loading the MNIST dataset

transform = transforms.ToTensor()

mnist_data = datasets.MNIST(root = "./data", train = True, download= True, transform=transform)

data_loader = DataLoader(dataset=mnist_data, batch_size = 64, shuffle= True)


# model stuff

model = AutoEncoder()
criterion = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr = 1e-3, weight_decay=1e-5)

num_epochs = 10
outputs = []


for epoch in range(num_epochs):
    for (img, _) in data_loader:
        img = img.reshape(-1, 28 * 28)

        recon = model(img)
        loss = criterion(recon, img)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"Epoch:{epoch + 1}, Loss:{loss.item():.4f}")
    outputs.append((epoch, img, recon))

for k in range(0, num_epochs, 4):
    plt.figure(figsize=(9, 2))

    plt.gray()

    imgs = outputs[k][1].detach().numpy()
    recon = outputs[k][2].detach().numpy()

    for i, item in enumerate(imgs):
        if i >= 9:
            break

        plt.subplot(2, 9, i + 1)

        item = item.reshape(-1, 28, 28)

        plt.imshow(item[0])

    for i, item in enumerate(recon):
        if i >= 9:
            break

        plt.subplot(2, 9, 9 + i + 1)
        item = item.reshape(-1, 28, 28)

        plt.imshow(item[0])