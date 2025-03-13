import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image

from torch.utils.data import DataLoader

BATCH_SIZE = 100

# MNIST Dataset
train_dataset = datasets.MNIST(root="./mnist_data/", train = True, transforms=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root="./mnist_data/", train = False, transforms=transforms.ToTensor(), download=False)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle = True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle = False)

class VariationalAutoEncoder(nn.Module):
    def __init__(self, x_dim, h_dim_1, h_dim_2, z_dim):
        super(VariationalAutoEncoder, self).__init__()

        self.x_dim = x_dim
        self.h_dim_1 = h_dim_1
        self.h_dim_2 = h_dim_2
        self.z_dim = z_dim

        self.fc1 = nn.Linear(x_dim, h_dim_1)
        self.fc2 = nn.Linear(h_dim_1, h_dim_2)
        self.fc31 = nn.Linear(h_dim_2, z_dim)
        self.fc32 = nn.Linear(h_dim_2, z_dim)


        self.fc4 = nn.Linear(z_dim, h_dim_2)
        self.fc5 = nn.Linear(h_dim_2, h_dim_1)
        self.fc6 = nn.Linear(h_dim_1, x_dim)

    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))

        return self.fc31(h), self.fc32(h)

    '''
    1. What is log_var.
    2. What is mu.

    Are they part of that parameterisation trick they were mentioning about?

    I think I get this now, somewhat.
    '''
    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return eps.mul(std).add(mu)

    def decoder(self, z):
        h = F.relu(self.fc4(z)) # Why not 2x here
        h = F.relu(self.fc5(h))

        return F.sigmoid(self.fc6(h))


    '''
    When we would do a backward pass, why wouldn't z be learnt - like mu and log_var
    '''
    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 784))

        z = self.sampling(mu, log_var)

        return self.decoder(z), mu, log_var

vae = VariationalAutoEncoder(x_dim = 28 * 28, h_dim_1 = 512, h_dim_2 = 256, z_dim = 2)

# Is there any other way of setting the device?
if torch.cuda.is_available():
    vae.cuda()

optimiser = optim.Adam(vae.parameters())

def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return BCE + KLD

def train(epoch):
    vae.train()

    train_loss = 0

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.cuda()

        optimiser.zero_grad()

        recon_batch, mu, log_var = vae(data)

        loss = loss_function(recon_batch, data, mu, log_var)

        loss.backward()

        train_loss += loss.item() # .item() returns the value of the tensor as a standard python number

        optimiser.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

def test():
    vae.eval()
    test_loss = 0

    with torch.no_grad():
        for data, _ in test_loader:
            data = data.cuda()

            recon, mu, log_var = vae(data)

            # summing up the batch loss
            test_loss += loss_function(recon, data, mu, log_var).item()

    test_loss /= len(test_loader.dataset) # Average
    print('====> Test set loss: {:.4f}'.format(test_loss))

for epoch in range(1, 51):
    train(epoch)
    test()

with torch.no_grad():
    z = torch.randn(64, 2).cuda()
    sample = vae.decoder(z).cuda()
    
    save_image(sample.view(64, 1, 28, 28), './samples/sample_' + '.png')