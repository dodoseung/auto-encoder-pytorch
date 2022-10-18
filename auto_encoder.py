import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np

# Hyper parameters
epochs = 50
batch_size = 256
learning_rate = 0.0002
shuffle = True
drop_last = True

# Training setting
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)
if device == 'cuda':
  torch.cuda.manual_seed_all(0)

# Set the transform
transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((28, 28))])

# Set the training data
data_train = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
loader_train = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

# Set the test data
data_test = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
loader_test = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

# Auto Encoder
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(5, 5), stride=(1, 1), padding='same')
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=(5, 5), stride=(1, 1), padding='same')
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        return x

# Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(8, 8, kernel_size=(5, 5), stride=(1, 1), padding='same')
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(5, 5), stride=(1, 1), padding='same')
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv3 = nn.Conv2d(16, 1, kernel_size=(5, 5), stride=(1, 1), padding='same')

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.upsample1(x)
        x = F.relu(self.conv2(x))
        x = self.upsample2(x)
        x = F.sigmoid(self.conv3(x))
        return x

# Gaussian noise
def add_gaussian_noise(img, scale=0.1):
    gaussian_noise = torch.randn_like(img)
    gaussian_noise_img = img + scale * gaussian_noise
    gaussian_noise_img = gaussian_noise_img.clip(min=0.0, max=1.0)
    return gaussian_noise_img

# Set the model, optimizer, and loss function
model = AutoEncoder().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

# Train model
model.train()
plt.figure(figsize=(3, 4))

for epoch in range(epochs):
    for step, (img, label) in enumerate(loader_train):
        input = add_gaussian_noise(img, 0.1).to(device)
        output = model(input)
        correct_img = img.to(device)

        loss = loss_fn(output, correct_img)  
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
          
    print("Train Epoch: {}   \tLoss: {:.6f}".format(epoch, loss.item()))

    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(img[0].squeeze(0))
    plt.axis('off')

    # Gaussian noise image
    plt.subplot(1, 3, 2)
    plt.imshow(input[0].squeeze(0).to('cpu').detach().numpy())
    plt.axis('off')

    # Model output image
    plt.subplot(1, 3, 3)
    plt.imshow(output[0].squeeze(0).to('cpu').detach().numpy())
    plt.axis('off')

    plt.show()
    
    # Save the model
    torch.save(model.state_dict(), 'auto_encoder.pt')

