"""Module with network training"""


import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader


def train_model(model: torch.nn.Module, num_epochs: int,
                optimizer, loss_func,
                train_dataloader: DataLoader, valid_dataloader: DataLoader,
                device: str = 'cuda:0'):

    device = torch.device(device)

    model.to(device=device)

    for epoch in range(num_epochs):

        model.train()

        train_one_epoch(model=model, dataloader=train_dataloader,
                        epoch_idx=epoch, loss_func=loss_func,
                        optimizer=optimizer, device=device)

        model.eval()

        # valid_one_epoch(model=model, dataloader=valid_dataloader,
        #                 epoch_idx=epoch, loss_func=loss_func,
        #                 optimizer=optimizer, device=device)


def train_one_epoch(model: torch.nn.Module, dataloader: DataLoader,
                    optimizer, loss_func,
                    epoch_idx: int, device: torch.device):
    tqdm_dataloader = tqdm(dataloader)

    for dataset_element in tqdm_dataloader:
        optimizer.zero_grad()

        image = dataset_element['image'].to(device)
        label = dataset_element['label'].to(device)

        result = model(image)

        loss_value = loss_func(result, label)

        loss_value.backward()
        optimizer.step()

        tqdm_dataloader.set_postfix(text=f'Loss number: {loss_value.detach().cpu().item()}')


def valid_one_epoch(model: torch.nn.Module, dataloader: DataLoader,
                    optimizer, loss_func,
                    epoch_idx: int, device: torch.device):
    tqdm_dataloader = tqdm(dataloader)

    for dataset_element in tqdm_dataloader:
        pass



