# Code to implement VAE-gumple_softmax in pytorch
# author: Devinder Kumar (devinder.kumar@uwaterloo.ca), modified by Yongfei Yan
# The code has been modified from pytorch example vae code and inspired by the origianl \
# tensorflow implementation of gumble-softmax by Eric Jang.

import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--temp', type=float, default=1.0, metavar='S',
                    help='tau(temperature) (default: 1.0)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--hard', action='store_true', default=False,
                    help='hard Gumbel softmax')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data/MNIST', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data/MNIST', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    if args.cuda:
        U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    
    if not hard:
        return y.view(-1, latent_dim * categorical_dim)

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(-1, latent_dim * categorical_dim)


class AutoencoderBase(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()

        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, hidden_dim)

        self.fc4 = nn.Linear(hidden_dim, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, 784)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, inp):
        h1 = self.relu(self.fc1(inp))
        h2 = self.relu(self.fc2(h1))
        return self.relu(self.fc3(h2))

    def decode(self, latent):
        h4 = self.relu(self.fc4(latent))
        h5 = self.relu(self.fc5(h4))
        return self.sigmoid(self.fc6(h5))

    def latent(self, hidden):
        raise NotImplementedError

    def prior_loss(self, hidden):
        return 0.0

    def forward(self, inp, *latent_args, **latent_kwargs):
        hidden = self.encode(inp.view(-1, 784))
        latent = self.latent(hidden, *latent_args, **latent_kwargs)
        return (self.decode(latent), self.prior_loss(hidden))


class DiscreteAutoencoderBase(AutoencoderBase):

    def __init__(self, latent_dim, categorical_dim):
        super().__init__(latent_dim * categorical_dim)
        self.latent_dim = latent_dim
        self.categorical_dim = categorical_dim

    def _logits(self, hidden):
        return hidden.view(hidden.size(0), self.latent_dim, self.categorical_dim)


class GumbelSoftmaxAutoencoder(DiscreteAutoencoderBase):

    def latent(self, hidden, temp, hard):
        return gumbel_softmax(self._logits(hidden), temp, hard)

    def prior_loss(self, hidden):
        posterior = F.softmax(self._logits(hidden), dim=-1).reshape(*hidden.size())
        log_ratio = torch.log(posterior * self.categorical_dim + 1e-20)
        return torch.sum(posterior * log_ratio, dim=-1).mean()


class DiscreteAutoencoder(DiscreteAutoencoderBase):

    def latent(self, hidden):
        raise NotImplementedError


latent_dim = 30
categorical_dim = 10  # one-of-K vector

temp_min = 0.5
ANNEAL_RATE = 0.00003

model = GumbelSoftmaxAutoencoder(latent_dim, categorical_dim)
if args.cuda:
    model.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, prior_loss):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False) / x.shape[0]
    return BCE + prior_loss


def train(epoch, temp):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, prior_loss = model(data, temp, args.hard)
        loss = loss_function(recon_batch, data, prior_loss)
        loss.backward()
        train_loss += loss.item() * len(data)
        optimizer.step()
        if batch_idx % 100 == 1:
            temp = np.maximum(temp * np.exp(-ANNEAL_RATE * 100), temp_min)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Temperature: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item(), temp))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))

    return temp


def test(epoch):
    model.eval()
    test_loss = 0
    for i, (data, _) in enumerate(test_loader):
        if args.cuda:
            data = data.cuda()
        recon_batch, prior_loss = model(data, temp_min, args.hard)
        test_loss += loss_function(recon_batch, data, prior_loss).item() * len(data)
        if i == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n],
                                    recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
            save_image(comparison.data.cpu(),
                       'data/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


def run():
    temp = args.temp
    for epoch in range(1, args.epochs + 1):
        temp = train(epoch, temp)
        test(epoch)

        M = 64 * latent_dim
        np_y = np.zeros((M, categorical_dim), dtype=np.float32)
        np_y[range(M), np.random.choice(categorical_dim, M)] = 1
        np_y = np.reshape(np_y, [M // latent_dim, latent_dim, categorical_dim])
        sample = torch.from_numpy(np_y).view(M // latent_dim, latent_dim * categorical_dim)
        if args.cuda:
            sample = sample.cuda()
        sample = model.decode(sample).cpu()
        save_image(sample.data.view(M // latent_dim, 1, 28, 28),
                   'data/sample_' + str(epoch) + '.png')


if __name__ == '__main__':
    run()
