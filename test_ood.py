import argparse
import torch
import torchvision as tv
import time
import numpy as np
import os
import torch.nn as nn
import math
from torchvision import transforms, utils
from PIL import Image
from tqdm import tqdm
from tabulate import tabulate
from utils import *

ood_datasets_cifar = ['SVHN', 'Textures', 'iSUN', 'LSUN_resize', 'LSUN', 'Places365'] 
ood_datasets_imagenet = ['iNaturalist', 'SUN', 'Places','Textures'] 

def make_id_ood_ImageNet():
    val_tx = tv.transforms.Compose([
        tv.transforms.Resize(256),
        tv.transforms.CenterCrop(224),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    in_set = tv.datasets.ImageFolder('./imagenet_datasets/ImageNet/val', val_tx)
   
    in_loader = torch.utils.data.DataLoader(
        in_set, batch_size=200, shuffle=False,
        num_workers=8, drop_last=False)

    out_loader_list = []
    for out_datadir in ood_datasets_imagenet:
        out_set = tv.datasets.ImageFolder('./imagenet_datasets' + out_datadir, val_tx)
        out_loader = torch.utils.data.DataLoader(
            out_set, batch_size=200, shuffle=False,
            num_workers=8, drop_last=False)
        out_loader_list.append(out_loader)
    
    return in_loader, out_loader_list

def make_id_ood_CIFAR10(args):
    val_tx = tv.transforms.Compose([
        tv.transforms.Resize(32),
        tv.transforms.CenterCrop(32),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])        
        
    in_set = tv.datasets.CIFAR10("./cifar_datasets/", 
                                   train=False, 
                                   transform=val_tx, 
                                   download=True)
    in_loader = torch.utils.data.DataLoader(in_set, batch_size=args.batch, shuffle=False, num_workers=8)
    
    out_loader_list = []
    for out_datadir in ood_datasets_cifar:
        if "SVHN" in out_datadir:
            out_set = tv.datasets.SVHN(
                root="./cifar_datasets/", 
                split="test",
                download=True, 
                transform=val_tx)
        elif "Places365" in out_datadir:
            out_set = tv.datasets.ImageFolder(f"./cifar_datasets/{out_datadir}", val_tx)
        else:
            out_set = tv.datasets.ImageFolder(f"./cifar_datasets/{out_datadir}", val_tx)

        out_loader = torch.utils.data.DataLoader(
            out_set, batch_size=args.batch, shuffle=False,
            num_workers=8, pin_memory=True, drop_last=False)
        out_loader_list.append(out_loader)
    
    return in_loader, out_loader_list

def make_id_ood_CIFAR100(args):
    val_tx = tv.transforms.Compose([
        tv.transforms.Resize(32),
        tv.transforms.CenterCrop(32),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])        
        
    in_set = tv.datasets.CIFAR100("./cifar_datasets/", 
                                   train=False, 
                                   transform=val_tx, 
                                   download=True)
    in_loader = torch.utils.data.DataLoader(in_set, batch_size=args.batch, shuffle=False, num_workers=8)
    
    out_loader_list = []
    for out_datadir in ood_datasets_cifar:
        if "SVHN" in out_datadir:
            out_set = tv.datasets.SVHN(
                root="./cifar_datasets/", 
                split="test",
                download=True, 
                transform=val_tx)
        elif "Places365" in out_datadir:
            out_set = tv.datasets.ImageFolder(f"./cifar_datasets/{out_datadir}", val_tx)
        else:
            out_set = tv.datasets.ImageFolder(f"./cifar_datasets/{out_datadir}", val_tx)

        out_loader = torch.utils.data.DataLoader(
            out_set, batch_size=args.batch, shuffle=False,
            num_workers=8, pin_memory=True, drop_last=False)
        out_loader_list.append(out_loader)
    
    return in_loader, out_loader_list

def iterate_data_energy(data_loader, model, temper):
    confs = []
    correct = 0
    total = 0
    for b, (x, y) in enumerate(tqdm(data_loader)):
        with torch.no_grad():
            x = x.cuda() 
            y = y.cuda()             
            features = model.forward_features(x)
            logits = model.forward_head(features)

            conf = temper * (torch.logsumexp(logits / temper, dim=1))
            _, predicted = torch.max(logits.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            confs.extend(conf.data.cpu().numpy())
    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
    
    return np.array(confs)

def iterate_data_ours(data_loader, model, temper, p, lam, std, mean):
    confs = []
    correct = 0
    total = 0

    mean_cpu = mean.data.cpu().numpy()
    thresh = np.percentile(mean_cpu, p)
    mask = torch.Tensor((mean_cpu > thresh)).cuda()

    for b, (x, y) in enumerate(tqdm(data_loader)):
        with torch.no_grad():
            x = x.cuda()            
            features = model.forward_features(x)
            logits = model.forward_head(features)
            
            w = model.fc.weight.data
            w = w.unsqueeze(0)
            f = features.unsqueeze(1)
            my_features = w * f
            
            mask_test = torch.where((my_features < (std * lam + mean)), 1.0, 0.0)
             
            my_features = my_features * mask * mask_test
            
            logits_m = torch.sum(my_features.transpose(2, 1), 1) + model.fc.bias.data
            
            y = y.cuda()
            _, predicted = torch.max(logits.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

            # Energy score
            conf = temper * (torch.logsumexp(logits_m / temper, dim=1))
            confs.extend(conf.data.cpu().numpy())
            
    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
    
    return np.array(confs)


def run_eval(model, in_loader, out_loader_list, args, num_classes):
    # switch to evaluate mode
    model.eval()

    print("Running test...")
    print("Dataset:", args.dataset,"Method:", args.method)
    if args.method in ['ITP']:
        if args.dataset == 'CIFAR10':    
            p, lam = 10, 2.2
        elif args.dataset == 'CIFAR100': 
            p, lam = 20, 1.6
        elif args.dataset == 'ImageNet':
            p, lam = 30, 1.5
        feature_std = torch.load(f"statistic/{args.dataset}/{args.model}/std.pt").cuda()
        feature_mean = torch.load(f"statistic/{args.dataset}/{args.model}/mean.pt").cuda()

    out_scores_list = []
    if args.method == 'Energy':
        print("Processing in-distribution data...")
        in_scores = iterate_data_energy(in_loader, model, args.temperature_energy)
        print("Processing out-of-distribution data...")
        for out_loader in out_loader_list:
            out_scores = iterate_data_energy(out_loader, model, args.temperature_energy) 
            out_scores_list.append(out_scores)
    elif args.method == 'ITP':
        print("Processing in-distribution data...")
        in_scores = iterate_data_ours(in_loader, model, args.temperature_energy, p, lam, feature_std, feature_mean)
        print("Processing out-of-distribution data...")
        for out_loader in out_loader_list:
            out_scores = iterate_data_ours(out_loader, model, args.temperature_energy, p, lam, feature_std, feature_mean) 
            out_scores_list.append(out_scores)  
    
    in_examples = in_scores.reshape((-1, 1))

    table_data = []
    auroc_sum = 0
    fpr95_sum = 0
    table_data.append([
        'Dataset',
        'FPR95',
        'AUROC'
    ])
    for idx, out_scores in enumerate(out_scores_list):
        out_examples = out_scores.reshape((-1, 1))
        auroc, aupr_in, aupr_out, fpr95 = get_measures(in_examples, out_examples)

        auroc_sum += auroc
        fpr95_sum += fpr95

        table_data.append([
            '{}'.format(ood_datasets_imagenet[idx] if args.dataset == 'ImageNet' else ood_datasets_cifar[idx]),
            '{:.2f}'.format(fpr95 * 100.),
            '{:.2f}'.format(auroc * 100.)
        ])

    num_datasets = len(out_scores_list)
    average_auroc = auroc_sum / num_datasets
    average_fpr95 = fpr95_sum / num_datasets

    table_data.append([
        'Average',
        '{:.2f}'.format(average_fpr95 * 100.),
        '{:.2f}'.format(average_auroc * 100.)
    ])

    transposed_table_data = list(map(list, zip(*table_data)))
    print(tabulate(transposed_table_data, tablefmt = 'grid', colalign = ('center', 'center', 'center')))

def main(args):
    from models.ResNet_pretrain import resnet50_modified
    model = resnet50_modified()
    model.fc = model.model.fc
    torch.backends.cudnn.benchmark = True

    if args.dataset == 'CIFAR10':    
        in_loader, out_loader_list = make_id_ood_CIFAR10(args)
        num_classes = 10
    elif args.dataset == 'CIFAR100':   
        in_loader, out_loader_list = make_id_ood_CIFAR100(args)
        num_classes = 100
    
    if args.dataset == 'ImageNet':
        in_loader, out_loader_list = make_id_ood_ImageNet()
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

    model = model.cuda()
    run_eval(model, in_loader, out_loader_list, args, num_classes=num_classes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", choices=['RN50', 'DN101'], default='DN101')
    parser.add_argument("--batch", type=int, default=1,help="Batch size.")
    parser.add_argument('--method', choices=['Energy', 'ITP'], default='Energy')
    parser.add_argument('--dataset', choices=['CIFAR10', 'CIFAR100', 'ImageNet'], default='ImageNet')

    # arguments for Energy
    parser.add_argument('--temperature_energy', default=1.0, type=int,
                        help='temperature scaling for energy')

    main(parser.parse_args())

    