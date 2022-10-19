from data_split import create_test_train_dataloader
import torch
import torch.nn.functional as F
from torchvision.models import resnet18


def train(model, train_dataloader, optimizer, epoch, log_interval, device):
    '''

    A function that is deployed in order to training a pytorch model.

    :param model: a torch model that we are interested in training
    :param train_dataloader: a torch Dataloader that contains the data meant for trading
    :param optimizer: the optimizer that is used during training
    :param epoch: the current training epoch
    :param log_interval: the interval used to log the results of the training process
    :param device: the device that is used for the training process (CPU or GPU ('cuda'))
    :return:
    '''

    # Set the model on training mode
    model.train()

    # for each batch
    for batch_idx, (data, target) in enumerate(train_dataloader):
        # Send the batch to the GPU
        data, target = data.float(), target.float()
        data, target = data.to(device), target.to(device)

        # Start the training process
        optimizer.zero_grad()

        # Get the model predictions for the current batch
        output = model.forward(data)

        # Calculate the MSE Loss between the predictions and the ground truth values
        loss = F.mse_loss(output, target)

        # Back-propagate the loss through the model
        loss.backward()

        # Take an optimizer step
        optimizer.step()

        # If it is time to log the training process
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                           len(train_dataloader.dataset),
                                                                           100. * batch_idx / len(train_dataloader),
                                                                           loss.item()))

        # If loss is under a small value ε, stop the training process
        epsilon = 1e-5
        if loss < epsilon:
            return true


if __name__ == '__main__':
    # TODO : set path to reforestree folder
    path_to_reforestree = ""
    # TODO :  set path where patches will be saved
    path_to_dataset = ""

    # Run it on GPU
    device = 'cuda'
    # Create the dataloaders thanks to Victoria!
    trdl, tsdl = create_test_train_dataloader(path_to_dataset)

    # Add a final layer on the ResNet to make it into a regressions task
    model = torch.nn.Sequential(resnet18(), torch.nn.Linear(in_features=1000, out_features=1), torch.nn.ReLU())

    # Send the model to the GPU
    model = model.to(device)

    # The optimizer and the scheduler used during training
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # Training the model for epochs amount of epochs.
    epochs = 50
    for epoch in range(1, epochs + 1):
        flag = train(model, train_dl, optimizer, epoch, log_interval=1)
        if flag:
            break
        # test(model, tsdl)
        scheduler.step()
