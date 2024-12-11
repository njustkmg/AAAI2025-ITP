import argparse
import torch
import torch.nn as nn
import torchvision as tv
from torchvision import transforms, utils
import torchvision.models as models
import time
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

def make_id_ood_ImageNet():
    val_tx = tv.transforms.Compose([
        tv.transforms.Resize(256),
        tv.transforms.CenterCrop(224),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    train_set = tv.datasets.ImageFolder('./imagenet_datasets/ImageNet/train', val_tx)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=256, shuffle=False,
        num_workers=8, pin_memory=True, drop_last=False)
    
    return train_loader

def make_CIFAR10_data():
    train_tx = tv.transforms.Compose([
        tv.transforms.Resize(32),
        tv.transforms.CenterCrop(32),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])       
        
    train_set = tv.datasets.CIFAR10("./cifar_datasets/", 
                                   train=True, 
                                   transform=train_tx, 
                                   download=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=False, num_workers=0)

    return train_loader

def make_CIFAR100_data():
    train_tx = tv.transforms.Compose([
        tv.transforms.Resize(32),
        tv.transforms.CenterCrop(32),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])      
        
    train_set = tv.datasets.CIFAR100("./cifar_datasets/", 
                                   train=True, 
                                   transform=train_tx, 
                                   download=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=False, num_workers=0)
    
    return train_loader


def main(args):    
    torch.backends.cudnn.benchmark = True

    # ================== dataloader =================
    if args.dataset == 'CIFAR10':    
        train_loader = make_CIFAR10_data()
        num_classes = 10
    elif args.dataset == 'CIFAR100':    
        train_loader = make_CIFAR100_data()
        num_classes = 100
    
    if args.dataset == 'ImageNet':
        train_loader = make_id_ood_ImageNet()
        num_classes = 1000
        if args.model == 'RN50':   
            from models.ResNet_pretrain import resnet50_modified
            model = resnet50_modified()
            model.fc = model.model.fc
    else:
        if args.model == 'DN101':
            from models.DenseNet import DenseNet3
            checkpoint = torch.load(f"checkpoints/densenet100_cifar{num_classes}.pth")
            model = DenseNet3(depth = 100, num_classes=num_classes)
            model.load_state_dict(checkpoint)
    # ==================== train ====================
    path = f'statistic/{args.dataset}/{args.model}'

    if not os.path.exists(path):
        os.makedirs(path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    score_list = [[]for _ in range(num_classes)]
    grad_list = [[]for _ in range(num_classes)]
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            features = model.forward_features(images)
            logits = model.forward_head(features)
            
            w = model.fc.weight.data
            w = w.unsqueeze(0)

            f = features.unsqueeze(1)
            score = w * f
            
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            for idx, (_predicted, _labels) in enumerate(zip(predicted, labels)):
                score_list[_labels].append(score[idx, _labels, :].cpu())
                            
        print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
    
    mean_list = []
    var_list = []
    for i in range(num_classes):
        score_list[i] = torch.stack(score_list[i])
        mean_list.append(torch.mean(score_list[i], 0))
        var_list.append(torch.sqrt(torch.var(score_list[i], 0)))
    mean = torch.stack(mean_list)
    std = torch.stack(var_list)
    torch.save(mean, f'{path}/mean.pt')
    torch.save(std, f'{path}/std.pt')
    print(mean)
    print(std)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model", choices=['RN50', 'DN101'], default='DN101')
    parser.add_argument('--dataset', choices=['CIFAR10', 'CIFAR100', 'ImageNet'], default='CIFAR10')

    main(parser.parse_args())

    