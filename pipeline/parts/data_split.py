from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from torchvision.transforms import ToTensor
import torch

class PatchesDataSet(Dataset):
    def __init__(self, path, transform=None):
        """
        Args:
            path (string): Path to the directory with all the npz files
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.path = path
        self.files = list(Path(self.path).glob('**/*.npz'))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        image= np.load(str(self.files[item]))["img"][0:3,:,:]
        image=np.moveaxis(image, 0, -1)  # numpy.ndarray (H x W x C) in the range [0, 255]
        label = np.load(str(self.files[item]))["label"]
        label = int(label)
        if self.transform is not None:
            image = self.transform(image)
        return (image, label)

def create_test_train_set(path_to_dataset, train_perc=0.8):
    patches_dataset = PatchesDataSet(path=path_to_dataset, transform=ToTensor())
    train_size=int(train_perc*len(patches_dataset))
    test_size=len(patches_dataset)-train_size
    train_dataset, test_dataset= torch.utils.data.random_split(patches_dataset, [train_size, test_size])
    return train_dataset, test_dataset

def create_test_train_dataloader(path_to_dataset, train_perc=0.8):
    train_dataset, test_dataset = create_test_train_set(path_to_dataset=path_to_dataset, train_perc=train_perc)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
    print('Training set has {} instances'.format(len(train_dataset)))
    print('Testing set has {} instances'.format(len(test_dataset)))
    return train_loader, test_loader



