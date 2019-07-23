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
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return self.fc3(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, x, y, optimizer, epoch):
    model.train()
    for idx in range(len(x)):
        data, target = x[idx], y[idx]
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, idx * len(data), len(train_data.data),
                   idx * 100.0 / x.size(dim=0), loss.item()))


def test(model, x, y):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for idx in range(len(x)):
            data, target = x[idx], y[idx]
            output = model(data)
            test_loss += F.cross_entropy(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_data.data)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_data.data),
        100. * correct / len(test_data.data)))


if __name__ == '__main__':
    args = {
        'batch_size': 1000,
        'lr': 0.001,
        'epochs': 10,
    }

    train_data = datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))
    test_data = datasets.MNIST('./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))

    net = Net().to(device)
    optimizer = optim.Adam(net.parameters(), lr=args['lr'])

    start_time = time.time()
    for epoch in range(args['epochs']):
        train(net, train_data.data.to(device).float().view(-1, args['batch_size'], 28 * 28),
              train_data.targets.to(device).view(-1, args['batch_size']), optimizer, epoch)
        test(net, test_data.data.to(device).float().view(-1, args['batch_size'], 28 * 28),
             test_data.targets.to(device).view(-1, args['batch_size']))
    print('Total Time: ', time.time() - start_time)
