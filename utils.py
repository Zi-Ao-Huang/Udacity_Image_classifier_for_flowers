import argparse
import numpy as np
import matplotlib.pyplot as plt
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


def load_data(where  = "./flowers"):
    data_dir = where
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    training_transforms = transforms.Compose([transforms.RandomRotation(30),
                                              transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])
    validation_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])
    testing_transforms = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                 [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(data_dir + '/train', transform=training_transforms)
    valid_data = datasets.ImageFolder(data_dir + '/valid', transform=validation_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=testing_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)

    return trainloader, validloader, testloader


def train_model(args, device):
    model = getattr(models, args.arch)(pretrained=True)
    hidden_units = int(args.hidden_units)
    trainloader, validloader, testloader = load_data()
    for param in model.parameters():
        param.requires_grad = False

    if args.arch == "vgg16":
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(25088, hidden_units)),
                                  ('drop', nn.Dropout(p=0.5)),
                                  ('relu', nn.ReLU()),
                                  ('fc2', nn.Linear(hidden_units, 102)),
                                  ('output', nn.LogSoftmax(dim=1))]))
    elif args.arch == "densenet121":
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(1024, hidden_units)),
                                  ('drop', nn.Dropout(p=0.5)),
                                  ('relu', nn.ReLU()),
                                  ('fc2', nn.Linear(hidden_units, 102)),
                                  ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=float(args.learning_rate))
    model.to(device)

    epochs = int(args.epochs)
    steps = 0
    running_loss = 0
    print_every = 20
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        group_loss = criterion(logps, labels)
                        test_loss += group_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {test_loss/len(validloader):.3f}.. "
                      f"Accuracy: {accuracy/len(validloader):.3f}")

                running_loss = 0
                model.train()
    print("Training done!")

    return model, optimizer, classifier


def testing(model, testloader, device):
    correct = 0
    total = 0
    model.to(device)
    trainloader, validloader, testloader = load_data()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy on the test images: %d %%' % (100 * correct / total))


def save_checkpoint(path, model, optimizer, args, classifier):
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'arch': args.arch,
                  'model': model,
                  'classifier' : classifier,
                  'learning_rate': args.learning_rate,
                  'hidden_units': args.hidden_units,
                  'epochs': args.epochs,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx}
    torch.save(checkpoint, path)


def load_checkpoint(path):
    checkpoint = torch.load(path)
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    learning_rate = checkpoint['learning_rate']
    model.epochs = checkpoint['epochs']
    model.load_state_dict(checkpoint['state_dict'])
    model.optimizer = checkpoint['optimizer']
    model.class_to_idx = checkpoint['class_to_idx']

    return model


def load_names(filename):
    with open(filename) as f:
        category_names = json.load(f)
    return category_names


def process_image(image):
    image = Image.open(image)
    image = image.resize((256,256))
    left = (256-224)/2
    top = (256-224)/2
    right = (256+224)/2
    bottom = (256+224)/2
    image = image.crop((left, top, right, bottom))
    image = np.array(image)/255
    mean = np.array([0.485,0.456,0.406])
    std = np.array([0.229,0.224,0.225])
    image = (image - mean) / std
    image = image.transpose((2,0,1))
    return image


def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    image = image.transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    ax.imshow(image)
    return ax
