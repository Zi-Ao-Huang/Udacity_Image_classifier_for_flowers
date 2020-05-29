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
from utils import load_data, load_checkpoint, process_image, load_names

def parse_args():
    parser = argparse.ArgumentParser(description="predict.py")
    parser.add_argument('--checkpoint', action="store", default="checkpoint.pth")
    parser.add_argument('--img_input', dest='img_input', default='flowers/test/66/image_05549.jpg')
    parser.add_argument('--top_k', dest='top_k', default='5')
    parser.add_argument('--category_names', dest='category_names', default='cat_to_name.json')
    parser.add_argument('--devices', action='store', default='gpu')
    return parser.parse_args()

args = parse_args()

def predict(img_path, model, top_k):
    devices = args.devices
    if devices == 'gpu':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    model.to(device)
    img_torch = process_image(img_path)
    img_torch = torch.from_numpy(img_torch)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()

    if devices == 'gpu':
        with torch.no_grad():
            output = model.forward(img_torch.cuda())
    else:
        with torch.no_grad():
            output = model.forward(img_torch)

    probs = F.softmax(output.data,dim=1)
    top_prob = np.array(probs.topk(top_k)[0][0])
    class_to_idx = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [class_to_idx[x] for x in np.array(probs.topk(top_k)[1][0])]

    return top_prob, top_classes

def main():
    model = load_checkpoint(args.checkpoint)
    category_names = load_names(args.category_names)
    top_k = int(args.top_k)
    devices = args.devices
    if devices == 'gpu':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    img_path = args.img_input
    top_prob, top_classes = predict(img_path, model, top_k)
    max_index = top_classes[0]
    names = [category_names[str(index)] for index in top_classes]
    flower_name = category_names[str(max_index)]

    i=0
    while i < len(names):
        print("The probability for the category {} is {}".format(names[i], top_prob[i]))
        i += 1
    print("The results indicate that this flower is is most likely to be {}".format(flower_name))
    print("Predition done!")

if __name__ == "__main__":
    main()
