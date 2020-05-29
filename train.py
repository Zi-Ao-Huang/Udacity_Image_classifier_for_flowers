import argparse
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as F
from collections import OrderedDict
from PIL import Image
import json
from utils import load_data, train_model, testing, save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Train.py")
    parser.add_argument('data_dir', nargs='*', action="store", default="./flowers/")
    parser.add_argument('--arch', dest='arch', default='vgg16', choices=['vgg16', 'densenet121'])
    parser.add_argument('--learning_rate', dest='learning_rate', default='0.001')
    parser.add_argument('--hidden_units', dest='hidden_units', default='1024')
    parser.add_argument('--epochs', dest='epochs', default='3')
    parser.add_argument('--devices', action='store', default='gpu')
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="checkpoint.pth")
    return parser.parse_args()


def main():
    args = parse_args()
    arch = args.arch
    where = args.data_dir
    learning_rate = args.learning_rate
    epochs = args.epochs
    save_dir = args.save_dir
    hidden_units = int(args.hidden_units)
    devices = args.devices
    if devices == 'gpu':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    train_data, valid_data, test_data, trainloader, validloader, testloader = load_data()
    model, optimizer, classifier = train_model(args, device)
    print ("Training complete.")
    testing(model, testloader, device)
    print ("Testing complete.")
    model.class_to_idx = train_data.class_to_idx
    path = args.save_dir
    save_checkpoint(path, model, optimizer, args, classifier)
    print ("checkpoint created.")


if __name__ == '__main__':
    main()
