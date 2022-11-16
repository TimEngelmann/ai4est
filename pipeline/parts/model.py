import torch
import torch.nn as nn
import logging
import torch.nn.functional as F
from torchvision.models import resnet18
import torchvision
# from IPython import embed
import torch.optim as optim
# from labml_nn.optimizers.amsgrad import AMSGrad
import torchvision.models as models
from torch.optim import Adam


class SimpleCNN(nn.Module):
    def __init__(self, img_dimension, n_channels):
        super().__init__()
        self.img_dimension=img_dimension
        self.n_channels=n_channels

        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=self.n_channels, out_channels=16, kernel_size=5), #16x24x24 OR 16x(img_dimension-4)*2
            nn.ReLU(),
            nn.MaxPool2d(2, 2) #16x12x12 OR 16x(img_dimension/2-2)*2
        )

        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5), #32x8x8 OR 32x(img_dimension/2-6)*2
            nn.ReLU(),
            nn.MaxPool2d(2, 2) #32x4x4 OR 32x(img_dimension/4-3)*2
        )


        self.fc_layer = nn.Sequential(
            nn.Linear(in_features= int(32*(self.img_dimension/4-3)**2), out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.conv_layer1(x)
        out=self.conv_layer2(out)
        out = out.view(-1, int(32*(self.img_dimension/4-3)**2))
        out = self.fc_layer(out)
        return out


class Resnet18Benchmark(nn.Module):
    def __init__(self):
        super().__init__()
        self.model= models.resnet18()
        self.model.fc= nn.Linear(512,1) #Change the final linear layer
        self.activation= nn.ReLU()


    def forward(self, x):
        out=self.model(x)
        out=self.activation(out)
        return out

def train(model, training_hyperparameters, train_loader):

    loss_fn=training_hyperparameters["loss_fn"]
    n_epochs=training_hyperparameters["n_epochs"]
    device=training_hyperparameters["device"]
    log_interval=training_hyperparameters["log_interval"]
    learning_rate=training_hyperparameters["learning_rate"]

    logging.info(f"Starting to train the model with {n_epochs} epochs total")
    model.to(device)
    model.train()

    if training_hyperparameters["optimizer"]=="amsgrad":
        # optimizer= AMSGrad(model.parameters(), lr=learning_rate)
        optimizer= Adam(model.parameters(), lr=learning_rate, amsgrad=True)
    elif training_hyperparameters["optimizer"]=="adam":
        optimizer= Adam(model.parameters(), lr=learning_rate, amsgrad=False)


    for epoch in range(n_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for batch_id, (img, carbon, _, _) in enumerate(train_loader, 0):
            # zero the parameter gradients
            img, carbon= img.to(torch.float32), carbon.to(torch.float32)
            img, carbon= img.to(device), carbon.to(device)
            optimizer.zero_grad()
            outputs = model(img)
            outputs=outputs.squeeze()
            loss = loss_fn(outputs, carbon)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if batch_id % log_interval == log_interval-1:
                print(f'[Epoch {epoch + 1}, Batch {batch_id + 1:5d}] loss: {running_loss / log_interval:.3f}')
                running_loss = 0.0
    logging.info(f'Finished Training')


# def val(model, val_dataloader, device):
#     '''
#     A function that is deployed in order to training a pytorch model.
#
#     :param model: a torch model that we are interested in training
#     :param train_dataloader: a torch Dataloader that contains the data meant for trading
#     :param device: the device that is used for the training process (CPU or GPU ('cuda'))
#     :return:
#     '''
#
#     # Set the model on training mode
#     model.eval()
#     test_loss = 0
#
#     with torch.no_grad():
#         for data, target in val_dataloader:
#             # Send the batch to the GPU
#             data, target = data.float(), target.float()
#             data, target = data.to(device), target.to(device)
#
#             # Get the model predictions for the current batch
#             output = model.forward(data)
#
#             # Calculate the MSE Loss between the predictions and the ground truth values
#             test_loss += F.mse_loss(output, target, reduction='sum').item()  # sum up batch loss
#
#     print('\nTest set: Average loss: {:.4f})\n'.format(test_loss/len(val_dataloader.dataset)))
