# Code to implement VAE-gumple_softmax in pytorch
# author: Devinder Kumar (devinder.kumar@uwaterloo.ca), modified by Yongfei Yan
# The code has been modified from pytorch example vae code and inspired by the origianl \
# tensorflow implementation of gumble-softmax by Eric Jang.

import argparse
import copy
import functools
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

torch.manual_seed(123)

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


def one_hot(x, param_dim):
    one_hot_size = x.size() + (param_dim,)
    one_hot = torch.zeros(one_hot_size).view(-1, param_dim)
    if args.cuda:
        one_hot = one_hot.cuda()
    one_hot.scatter_(-1, x.view(-1, 1), 1)
    return one_hot.view(one_hot_size)


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
        return y

    shape = y.size()
    ind = y.argmax(dim=-1)
    y_hard = one_hot(ind, y.size(-1))
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard


def saturating_sigmoid(logits):
    return torch.clamp(torch.clamp(1.2 * torch.sigmoid(logits) - 0.1, max=1), min=0)


def mix(a, b, prob=0.5):
    mask = (torch.rand_like(a) < prob).float()
    return mask * a + (1 - mask) * b


def improved_semantic_hashing(logits, noise_std):
    noise = torch.normal(mean=torch.zeros_like(logits), std=noise_std)
    noisy_logits = logits + noise
    continuous = saturating_sigmoid(noisy_logits)
    discrete = (noisy_logits > 0).float() + continuous - continuous.detach()
    return mix(continuous, discrete)


class CategoricalAutoencoderBase(nn.Module):

    latent_dim = None
    param_dim = None
    categorical_dim = None
    needs_temperature = None

    def __init__(self):
        super().__init__()
        hidden_dim = self.latent_dim * self.param_dim

        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, hidden_dim)

        self.fc4 = nn.Linear(hidden_dim, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, 784)

    def encode(self, inp):
        inp = inp.view(-1, 784)
        h1 = F.relu(self.fc1(inp))
        h2 = F.relu(self.fc2(h1))
        return self.fc3(h2).view(
            -1, self.latent_dim, self.param_dim
        )

    def decode(self, latent):
        latent = latent.view(-1, self.latent_dim * self.param_dim)
        h4 = F.relu(self.fc4(latent))
        h5 = F.relu(self.fc5(h4))
        return torch.sigmoid(self.fc6(h5)).view(-1, 1, 28, 28)

    def latent(self, hidden):
        raise NotImplementedError

    def hard_latent(self, hidden):
        raise NotImplementedError

    def prior_loss(self, hidden):
        raise NotImplementedError

    def embed(self, categories):
        raise NotImplementedError

    def forward(self, inp, *latent_args, **latent_kwargs):
        hidden = self.encode(inp)
        if self.training:
            latent = self.latent(hidden, *latent_args, **latent_kwargs)
            prior_loss = self.prior_loss(hidden)
        else:
            latent = self.embed(self.hard_latent(hidden))
            prior_loss = None
        return (self.decode(latent), prior_loss)


class GumbelSoftmaxAutoencoder(CategoricalAutoencoderBase):

    needs_temperature = True

    def __init__(self, latent_dim, categorical_dim):
        self.latent_dim = latent_dim
        self.param_dim = categorical_dim
        self.categorical_dim = categorical_dim
        super().__init__()

    def latent(self, hidden, temp, hard=False):
        return gumbel_softmax(hidden, temp, hard)

    def hard_latent(self, hidden):
        return torch.argmax(hidden, dim=-1)

    def prior_loss(self, hidden):
        posterior = F.softmax(hidden, dim=-1).reshape(hidden.size(0), -1)
        log_ratio = torch.log(posterior * self.param_dim + 1e-20)
        return torch.sum(posterior * log_ratio, dim=-1).mean()

    def embed(self, categories):
        return one_hot(categories, self.categorical_dim)


class DiscreteAutoencoder(CategoricalAutoencoderBase):

    param_dim = 1
    categorical_dim = 2
    needs_temperature = False

    def __init__(self, discretization, latent_dim):
        self.latent_dim = latent_dim
        super().__init__()
        self.discretization = discretization

    def latent(self, hidden):
        return self.discretization(hidden)

    def hard_latent(self, hidden):
        return (hidden > 0.0).long().view(*hidden.size()[:-1])

    def prior_loss(self, hidden):
        return 0.0

    def embed(self, categories):
        return categories.float()


class LatentPredictor(nn.Module):

    def __init__(
        self, hidden_dim, categorical_dim, num_digits_per_chunk=8, embedding_dim=64
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.categorical_dim = categorical_dim
        self.num_digits_per_chunk = num_digits_per_chunk
        self.num_codes = categorical_dim ** num_digits_per_chunk
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(self.num_codes, self.embedding_dim)
        self.rnn = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=2,
        )
        self.output = nn.Linear(hidden_dim, self.num_codes)

    def reset(self, batch_size):
        def zeros():
            x = torch.zeros(self.rnn.num_layers, batch_size, self.hidden_dim)
            if args.cuda:
                x = x.cuda()
            return x

        self.hidden_state = tuple(zeros() for _ in range(2))

    def chunk_latent(self, latent):
        (_, latent_dim) = latent.size()
        latent = latent.view(
            -1, latent_dim // self.num_digits_per_chunk, self.num_digits_per_chunk
        )
        return sum(
            2 ** i * latent[:, :, i] for i in range(self.num_digits_per_chunk)
        )

    def unchunk_latent(self, chunked_latent):
        latent = []
        for _ in range(self.num_digits_per_chunk):
            latent.append(chunked_latent % self.categorical_dim)
            chunked_latent //= self.categorical_dim
        return torch.stack(latent, dim=-1).view(chunked_latent.size(0), -1)

    def forward(self, chunked_latent):
        input = self.embedding(chunked_latent)
        (output, self.hidden_state) = self.rnn(input, self.hidden_state)
        return self.output(output)

    def infer(self, batch_size, latent_dim):
        self.reset(batch_size)
        self.eval()
        latent_dim //= self.num_digits_per_chunk
        prediction = torch.zeros(batch_size, latent_dim)
        latent = torch.zeros(batch_size, 1, dtype=torch.long)
        if args.cuda:
            prediction = prediction.cuda()
            latent = latent.cuda()
        for i in range(latent_dim):
            latent_dist = torch.distributions.Categorical(logits=self(latent))
            latent = latent_dist.sample()
            prediction[:, i] = latent.view(-1)
        return self.unchunk_latent(prediction)


class Classifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, input):
        x = F.relu(self.fc1(input.view(input.size(0), -1)))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


latent_dim = 30
param_dim = 10  # one-of-K vector

temp_min = 0.5
ANNEAL_RATE = 0.00003

#autoencoder = GumbelSoftmaxAutoencoder(latent_dim=30, categorical_dim=10)
autoencoder = DiscreteAutoencoder(
    discretization=functools.partial(improved_semantic_hashing, noise_std=1),
    latent_dim=32,
)
autoencoder_kwargs = {}
predictor = LatentPredictor(
    hidden_dim=128, categorical_dim=autoencoder.categorical_dim
)
classifier = Classifier()
if args.cuda:
    autoencoder.cuda()
    predictor.cuda()
    classifier = classifier.cuda()
optimizer = optim.Adam(
    list(autoencoder.parameters()) + list(predictor.parameters()),
    lr=1e-3,
)
classifier_optimizer = optim.Adam(classifier.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, prior_loss):
    loss = F.binary_cross_entropy(
        recon_x, x.view(-1, 784), size_average=False
    ) / x.shape[0]
    if prior_loss is not None:
        loss += prior_loss
    return loss


def train(epoch, temp, **autoencoder_kwargs):
    autoencoder.train()
    predictor.train()
    autoencoder_kwargs = copy.copy(autoencoder_kwargs)
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        if autoencoder.needs_temperature:
            autoencoder_kwargs["temp"] = temp
        recon_batch, prior_loss = autoencoder(data, **autoencoder_kwargs)
        autoencoder_loss = loss_function(recon_batch, data, prior_loss)

        chunked_hard_latent = predictor.chunk_latent(
            autoencoder.hard_latent(autoencoder.encode(data))
        )
        predictor.reset(data.size(0))
        prediction = predictor(chunked_hard_latent)
        predictor_loss = F.cross_entropy(
            prediction[:, :-1, :].contiguous().view(-1, predictor.num_codes),
            chunked_hard_latent[:, 1:].contiguous().view(-1),
        )

        (autoencoder_loss + predictor_loss).backward()
        train_loss += autoencoder_loss.item() * len(data)
        optimizer.step()
        if batch_idx % 100 == 1:
            temp = np.maximum(temp * np.exp(-ANNEAL_RATE * 100), temp_min)

        if batch_idx % args.log_interval == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tAE loss: {:.6f} '
                'Predictor loss: {:.6f} Temperature: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    autoencoder_loss.item(), predictor_loss.item(), temp
                )
            )

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))

    return temp


def test(epoch, **autoencoder_kwargs):
    autoencoder.eval()
    test_loss = 0
    test_predictor_loss = 0
    for i, (data, _) in enumerate(test_loader):
        if args.cuda:
            data = data.cuda()
        recon_batch, prior_loss = autoencoder(data, **autoencoder_kwargs)

        chunked_hard_latent = predictor.chunk_latent(
            autoencoder.hard_latent(autoencoder.encode(data))
        )
        predictor.reset(data.size(0))
        prediction = predictor(chunked_hard_latent)
        predictor_loss = F.cross_entropy(
            prediction[:, :-1, :].contiguous().view(-1, predictor.num_codes),
            chunked_hard_latent[:, 1:].contiguous().view(-1),
        )
        test_predictor_loss += predictor_loss.item()

        test_loss += loss_function(recon_batch, data, prior_loss).item() * len(data)
        if i == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n],
                                    recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
            save_image(comparison.data.cpu(),
                       'data/reconstruction_' + str(epoch) + '.png', nrow=n)

    #test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f} Predictor: {:.4f}'.format(
        test_loss, test_predictor_loss / len(test_loader))
    )


def train_classifier():
    def maybe_cuda(x):
        if args.cuda:
            x = x.cuda()
        return x

    classifier.train()
    for _ in range(3):
        for batch_idx, (data, labels) in enumerate(train_loader):
            classifier_optimizer.zero_grad()
            prediction = classifier(maybe_cuda(data.view(data.size(0), -1)))
            loss = F.cross_entropy(prediction, maybe_cuda(labels))
            loss.backward()
            classifier_optimizer.step()

    classifier.eval()
    print('Classifier test accuracy: {}'.format(sum(
        (
            classifier(
                maybe_cuda(data.view(data.size(0), -1))
            ).argmax(dim=-1) == maybe_cuda(labels)
        ).float().mean()
        for (data, labels) in test_loader
    ) / len(test_loader)))


def inception_score(images):
    predictions = F.softmax(classifier(images))
    marginal = predictions.mean(dim=0)
    print('Marginal class probabilities: {}'.format(marginal.detach().cpu().numpy()))
    return (predictions * (torch.log(predictions) - torch.log(marginal))).mean()


def run():
    train_classifier()

    temp = args.temp
    for epoch in range(1, args.epochs + 1):
        temp = train(epoch, temp, **autoencoder_kwargs)
        test(epoch)

        num_examples = 64
        sample = torch.randint(
            high=autoencoder.categorical_dim,
            size=(num_examples, autoencoder.latent_dim),
        )
        if args.cuda:
            sample = sample.cuda()
        sample = autoencoder.decode(autoencoder.embed(sample)).view(-1, 1, 28, 28)
        save_image(sample.cpu().data,
                   'data/sample_' + str(epoch) + '.png')
        print('Independent sampling inception score:', inception_score(sample).item())

        sample = predictor.infer(
            batch_size=num_examples, latent_dim=autoencoder.latent_dim
        )
        sample = autoencoder.decode(autoencoder.embed(sample)).view(-1, 1, 28, 28)
        save_image(sample.cpu().data,
                   'data/sample_rnn_' + str(epoch) + '.png')
        print('RNN sampling inception score:', inception_score(sample).item())

    print(inception_score(sample).item())


if __name__ == '__main__':
    run()
