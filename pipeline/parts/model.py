import pandas as pd
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
from torch.utils.tensorboard import SummaryWriter
import datetime


writer = SummaryWriter()
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


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
        out = self.conv_layer2(out)
        out = out.view(-1, int(32*(self.img_dimension/4-3)**2))
        out = self.fc_layer(out)
        return out


class Resnet18Benchmark(nn.Module):
    def __init__(self):
        super().__init__()
        self.model= models.resnet18() #weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc= nn.Linear(512,1) #Change the final linear layer
        self.activation= nn.ReLU()


    def forward(self, x):
        out=self.model(x)
        out=self.activation(out)
        return out

def train(model, training_hyperparameters, train_loader, val_loader, test_loader, site_name):

    loss_fn=training_hyperparameters["loss_fn"]
    n_epochs=training_hyperparameters["n_epochs"]
    device=training_hyperparameters["device"]
    log_interval=training_hyperparameters["log_interval"]
    learning_rate=training_hyperparameters["learning_rate"]

    # val_results = pd.DataFrame()

    logging.info(f"Starting to train the model with {n_epochs} epochs total")
    model.to(device)
    model.train()

    if training_hyperparameters["optimizer"]=="amsgrad":
        # optimizer= AMSGrad(model.parameters(), lr=learning_rate)
        optimizer= Adam(model.parameters(), lr=learning_rate, amsgrad=True)
    elif training_hyperparameters["optimizer"]=="adam":
        optimizer= Adam(model.parameters(), lr=learning_rate, amsgrad=False)

    epoch_loss = 0
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
            epoch_loss += loss.item()
            if batch_id % log_interval == log_interval-1:
                print(f'[Epoch {epoch + 1}, Batch {batch_id + 1:5d}] loss: {running_loss / log_interval:.3f}')
                running_loss = 0.0
        # val_results = pd.concat([val_results, results], axis=1)
        writer.add_scalar("Train/Epoch", epoch_loss, epoch)
        epoch_loss = 0
        writer.flush()
        val(model, epoch, val_loader, loss_fn, device)
    logging.info(f'Finished Training')
    logging.info(f'Testing Model')
    test_results = test(model, test_loader, loss_fn, device)
    writer.close()
    test_results.to_csv('{}.csv'.format(site_name))


def val(model, epoch, val_dataloader, loss_fn, device):
    '''
    A function that is deployed in order to validate a pytorch model.
    :param model: a torch model that we are interested in training
    :param epoch: the epoch that is currently being validated
    :param val_dataloader: a torch Dataloader that contains the data meant for validating
    :param device: the device that is used for the training process (CPU or GPU ('cuda'))
    :return:
    '''
    # Set the model on training mode
    model.eval()
    val_loss = 0
    val_epoch_loss = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for batch_id, (data, target, _, _) in enumerate(val_dataloader, 0):
            # Send the batch to the GPU
            data, target = data.float(), target.float()
            data, target = data.to(device), target.to(device)

            # Get the model predictions for the current batch
            output = model.forward(data)
            output = output.squeeze()
            targets.append(target.to("cpu"))
            predictions.append(output.to("cpu"))

            # Calculate the MSE Loss between the predictions and the ground truth values
            val_loss += loss_fn(output, target)
            val_epoch_loss += val_loss

            # print('\nValidation set: Batch Loss: {:.4f})\n'.format(val_loss))
            val_loss = 0
    writer.add_scalar("Val/Epoch", val_epoch_loss, epoch)
    writer.flush()
    val_epoch_loss = 0

def test(model, test_dataloader, loss_fn, device):
    '''
    A function that is deployed in order to test a pytorch model.
    :param model: a torch model that we are interested in training
    :param test_dataloader: a torch Dataloader that contains the data meant for testing
    :param device: the device that is used for the training process (CPU or GPU ('cuda'))
    :return:
    '''

    test_results = pd.DataFrame()
    # Set the model on training mode
    model.eval()
    test_loss = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for batch_id, (data, target, _, _) in enumerate(test_dataloader, 0):
            # Send the batch to the GPU
            data, target = data.float(), target.float()
            data, target = data.to(device), target.to(device)

            # Get the model predictions for the current batch
            output = model.forward(data)
            output = output.squeeze()
            targets.append(target.to("cpu"))
            predictions.append(output.to("cpu"))

            # Calculate the MSE Loss between the predictions and the ground truth values
            test_loss += loss_fn(output, target) # sum up batch loss

            print('\nTest set: Batch Loss: {:.4f})\n'.format(test_loss))
            writer.add_scalar("Test/Batch_loss", test_loss, batch_id)
            test_loss = 0

    for i in range((len(targets))):
        targets[i] = targets[i].tolist()
        predictions[i] = predictions[i].tolist()

    targets = [item for sublist in targets for item in sublist]
    if type(predictions[-1]) is not list:
        predictions[-1] = [predictions[-1]]
    predictions = [item for sublist in predictions for item in sublist]

    test_results['true_value'] = targets
    test_results['preds'] = predictions
    return test_results