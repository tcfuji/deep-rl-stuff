import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

torch.manual_seed(123)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        """
        For the nn.Conv2d below, the opertaion takes a batch of images with
        3 channels and produces a batch with 6 channels through convolution.
        Since the input for kernel_size is 5, we have a kernel with a 5 x 5 dim.
        By default, Pytorch kernels use He initialization from `Delving deep into
        rectifiers: Surpassing human-level performance on ImageNet
        classification` - He, K. et al. (2015).
        Other defaults:
        - stride: (1, 1)
        - padding: 0
        - dilation: 1
        Output: h_out = floor( (h_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
        w_out = floor( ((w_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1 )
        So, h_out = floor( ((32 + 2 * 0 - 1 * (5 - 1) - 1) / 1) + 1 ) = 28 = w_out
        """
        self.conv1 = nn.Conv2d(3, 6, 5)
        """
        Max pooling: First input is 2, which means the kernel size is 2 x 2.
        Second input is 2, which means the stride is 2 x 2.
        h_out = floor( ((28 + 2 * 0 - 1 * (2 - 1) - 1) / 2) + 1 ) = 14 = w_out
        """
        self.pool = nn.MaxPool2d(2, 2)
        """
        6 to 16 channels. Kernel size is same as last conv2d (5, 5).
        h_out = floor( ((14 + 2 * 0 - 1 * (5 - 1) - 1) / 1) + 1 ) = 10 = w_out
        When applied with 'forward' method using maxpool2d, we have
        h_out = floor(((10 + 2 * 0 - 1 * (2 - 1) - 1) / 2) + 1) = 5 = w_out
        """
        self.conv2 = nn.Conv2d(6, 16, 5)
        """
        self.fc1 = nn.Linear(Channels * h_out * w_out, output_dim)
        """
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # the size -1 is inferred from other dimensions to fit with self.fc1
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True,
                                         num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False,
                                        num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'frog', 'horse',
            'ship', 'truck')

model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# training
# start = torch.cuda.Event(enable_timing=True)
# end = torch.cuda.Event(enable_timing=True)

# start.record()
start = time.time()
for epoch in range(5):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients since they accumulate
        optimizer.zero_grad()

        # forward and back and optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print stats
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print("[%d, %5d] loss: %.3f" %
                    (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
end.record()
# wait for everything to finish running
# torch.cuda.synchronize()
# print("Time elapsed during training: {}s".format(start.elapsed_time(end)))
print("Time elapsed during training: {}s".format(time.time() - start))

print("Finished Training!")

# print overall accuracy on 10000 test images
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        # torch.max outputs (values, indices)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the model on the 10000 test images: %d %%' %
        (100 * correct / total))

# print accuracy by class
class_correct = [0. for _ in range(10)]
class_total = [0. for _ in range(10)]
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels)
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]
    ))

"""
Output:
Accuracy of the model on the 10000 test images: 58 %
Accuracy of plane : 65 %
Accuracy of   car : 73 %
Accuracy of  bird : 46 %
Accuracy of   cat : 25 %
Accuracy of  deer : 55 %
Accuracy of   dog : 55 %
Accuracy of  frog : 71 %
Accuracy of  frog : 58 %
Accuracy of horse : 81 %
Accuracy of  ship : 55 %
"""
