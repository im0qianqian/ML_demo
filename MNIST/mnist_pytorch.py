import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets, transforms
import time


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(28 * 28, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 10)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return self.fc3(x)


def train(model, train_data, optimizer, epoch):
    model.train()
    idx = 0
    for data, target in train_data:
        data, target = data.to(device).view(-1, 28 * 28), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if idx % args['log_interval'] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, idx * len(data), len(train_loader.dataset),
                       idx * 100.0 / len(train_data), loss.item()))
        idx = idx + 1


def test(model, test_data):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_data:
            data, target = data.to(device).view(-1, 28 * 28), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    args = {
        'batch_size': 1000,
        'test_batch_size': 1000,
        'lr': 0.001,
        'epochs': 10,
        'log_interval': 10,
        'use_cuda': torch.cuda.is_available(),
        'data_loader_num_workers': 4,
    }
    device = torch.device('cuda' if args['use_cuda'] else 'cpu')
    kwargs = {'num_workers': args['data_loader_num_workers'], 'pin_memory': True} if args['use_cuda'] else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args['batch_size'], shuffle=True, **kwargs)
    train_data = list(train_loader)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args['test_batch_size'], shuffle=True, **kwargs)
    test_data = list(test_loader)

    net = Net().to(device)
    optimizer = optim.Adam(net.parameters(), lr=args['lr'])

    start_time = time.time()
    for epoch in range(args['epochs']):
        train(net, train_data, optimizer, epoch)
        test(net, test_data)
    print('Total Time: ', time.time() - start_time)
