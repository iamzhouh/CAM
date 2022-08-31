import torch
import torch.nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional
import torch.optim

batch_size = 64
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307, ), (0.3081, ))])   #归一化

train_dataset = datasets.MNIST(root = './dataset/mnist/', train=True, download = True, transform = transform)
train_loader = DataLoader(train_dataset, shuffle = True, batch_size = batch_size)

test_dataset = datasets.MNIST(root = './dataset/mnist/', train=False, download = True, transform = transform)
test_loader = DataLoader(test_dataset, shuffle = True, batch_size = batch_size)

class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size = 3, padding = 1)
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size = 3, padding = 1)

    def forward(self, x):
        y = torch.nn.functional.relu(self.conv1(x))
        y = self.conv2(y)
        return torch.nn.functional.relu(y + x)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(16, 32,kernel_size=5)

        self.rblock1 = ResidualBlock(16)
        self.rblock2 = ResidualBlock(32)

        self.mp = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(32, 10)

        self.GAP = torch.nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        in_size = x.size(0)

        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.mp(x)
        x = self.rblock1(x)
        # x.shape = [64,16,12,12]


        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = self.mp(x)
        # x.shape = [64,32,8,8]

        x = self.rblock2(x)
        # x.shape = [64,32,4,4]

        x = self.GAP(x)
        # x.shape = [64,32,1,1]
        x = x.view(in_size, -1)

        x = self.fc(x)
        return x

model = Net()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum = 0.5)

def train(epoch):
    running_loss = 0.0
    for batch_index, data  in enumerate(train_loader, 0):
        inputs, targets = data
        y_pred = model(inputs)
        loss = criterion(y_pred, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_index % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch+1, batch_index+1, running_loss/300))
            running_loss = 0.0

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _,predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('accuracy on test set: %d %% ' % (100*correct/total))


if __name__ == '__main__':
    test()
    for epoch in range(10):
        train(epoch)
        test()
    PATH = 'state_dict_model.pth'
    torch.save(model.state_dict(), PATH)