"""Module with training of networks"""

import json

import torch
import pandas as pd

from modules.model.network import CassavaNet
from modules.data.dataloader import create_dataloader, split_dataset
from modules.data.augs import train_augmentations, valid_augmentations
from modules.model.training import train_model
from modules.utils import load_config


if __name__ == '__main__':
    config_path = '/home/vadbeg/Projects/Kaggle/cassava-leaf-disease/config.ini'
    config = load_config(config_path=config_path)

    train_images_path = config.get('Data', 'train_images_path')
    train_dataframe_path = config.get('Data', 'train_dataframe_path')

    image_size = tuple(json.loads(config.get('Model', 'image_size')))
    valid_size = config.getfloat('Model', 'valid_size')
    batch_size = config.getint('Model', 'batch_size')
    device = config.get('Model', 'device')
    learning_rate = config.getfloat('Model', 'learning_rate')

    train_dataframe = pd.read_csv(train_dataframe_path)

    train_dataset, valid_dataset = split_dataset(
        dataframe=train_dataframe,
        images_path=train_images_path,
        train_augmentations=train_augmentations,
        valid_augmentations=valid_augmentations,
        image_size=image_size,
        valid_size=valid_size,
    )

    train_dataloader = create_dataloader(dataset=train_dataset, batch_size=batch_size)
    valid_dataloader = create_dataloader(dataset=valid_dataset, batch_size=batch_size)

    model = CassavaNet(model_type='resnet18', pretrained=True)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    loss_func = torch.nn.CrossEntropyLoss()

    train_model(model=model,
                num_epochs=15,
                optimizer=optimizer,
                loss_func=loss_func,
                train_dataloader=train_dataloader,
                valid_dataloader=valid_dataloader)

