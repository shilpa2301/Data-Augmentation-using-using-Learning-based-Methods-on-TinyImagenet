import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset

import argparse
import matplotlib.pyplot as plt
import numpy as np
import copy
import math
import pickle
from torch.optim import lr_scheduler
import random
import time


device='cuda' if torch.cuda.is_available() else 'cpu'
# device



class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Accuracy calculation function
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(100.0 - correct_k.mul_(100.0 / batch_size))
    return res

# Validate function
def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # Switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (inputs, target) in enumerate(val_loader):
        inputs, target = inputs.to(device), target.to(device)

        # Forward pass
        output = model(inputs)
        loss = criterion(output, target)

        # Measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(err1.item(), inputs.size(0))
        top5.update(err5.item(), inputs.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0 and args.verbose:
        if i % 1 == 0:
            print('Test (on val set): [{0}/{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                   epoch, 300, i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}  Top 5-err {top5.avg:.3f}\t Test Loss {loss.avg:.3f}'.format(
        epoch, 300, top1=top1, top5=top5, loss=losses))
    return top1.avg, top5.avg, losses.avg

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

#data augmentation
def random_erasing(img, area_ratio=0.4, aspect_ratio=1.0, mean=[0.4914, 0.4822, 0.4465], p=0.5):

        if random.uniform(0, 1) > p:
            return img

        for attempt in range(100):
            area = img.shape[1] * img.shape[2]
       
            target_area = area_ratio * area

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.shape[2] and h < img.shape[1]:
                x1 = random.randint(0, img.shape[1] - h)
                y1 = random.randint(0, img.shape[2] - w)
                if img.shape[0] == 3:
                    #img[0, x1:x1+h, y1:y1+w] = random.uniform(0, 1)
                    #img[1, x1:x1+h, y1:y1+w] = random.uniform(0, 1)
                    #img[2, x1:x1+h, y1:y1+w] = random.uniform(0, 1)
                    img[0, x1:x1+h, y1:y1+w] = mean[0]
                    img[1, x1:x1+h, y1:y1+w] = mean[1]
                    img[2, x1:x1+h, y1:y1+w] = mean[2]
                    #img[:, x1:x1+h, y1:y1+w] = torch.from_numpy(np.random.rand(3, h, w))
                else:
                    img[0, x1:x1+h, y1:y1+w] = mean[1]
                    # img[0, x1:x1+h, y1:y1+w] = torch.from_numpy(np.random.rand(1, h, w))
                return img

        return img

def train_model_random_erase(model, criterion, optimizer, train_loader, test_loader, epochs, device, patience=10, min_delta=0.0, p=0.5):
    
    epoch_loss=[]
    epoch_acc=[]
    test_acc=[]
    test_loss=[]

    best_err1 = float('inf')
    best_err5 = float('inf')
    best_model_state = None

    for epoch in range(epochs):
      print(f"\nEpoch {epoch + 1}/{epochs}")

      running_loss=0.0
      running_corrects=0

      model.train()
      total_labels=0

      
      for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = torch.stack([random_erasing(img) for img in inputs])
        # print(inputs.shape[0])
        optimizer.zero_grad()
        outputs=model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
        _,predicted=outputs.max(1)
        
        running_corrects +=  predicted.eq(labels).sum().item()
        total_labels += len(labels)
      epoch_loss.append(running_loss/len(train_loader.dataset))      
      epoch_acc.append(running_corrects/total_labels)

      # Evaluate on validation set
      err1, err5, val_loss = validate(test_loader, model, criterion, epoch)
      test_acc.append(100.0 - err1)
      test_loss.append(val_loss)  
      # Save best model based on top-1 error
      is_best = err1 <= best_err1
      best_err1 = min(err1, best_err1)
      if is_best:
          best_err5 = err5
          best_model_state = model.state_dict()
          print(f"Best model state stored with top-1 error: {best_err1:.4f}, top-5 error: {best_err5:.4f}")  
      # Save accuracies to a file
      with open("accuracy_log.txt", "a+") as f:
          f.write(f"Epoch {epoch + 1}: Top-1 Error: {err1:.4f}, Top-5 Error: {err5:.4f}, Loss: {val_loss:.4f}\n")

    # Load the best model state
    model.load_state_dict(best_model_state)
    return model, epoch_loss, epoch_acc, test_acc

def main():

    train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262]), 
            ])

    test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262]), 
        ])


        # Load the original datasets
    full_train_set = torchvision.datasets.ImageFolder(root='~/acv/SaliencyMix/tiny-imagenet-200/train', transform=train_transform)
    test_set = torchvision.datasets.ImageFolder(root='~/acv/SaliencyMix/tiny-imagenet-200/val', transform=test_transform)
    # test_dataset = torchvision.datasets.ImageFolder(root='/home/csgrad/dl_228/tiny-imagenet-200/test', transform=test_transform)

    class_counts = {i:0 for i in range(200)}
    max_samples_per_class = 250

    selected_indices=[]
    for idx, label in enumerate(full_train_set.targets):
      if class_counts[label]<max_samples_per_class:
        selected_indices.append(idx)
        class_counts[label]+=1
      if all(count == max_samples_per_class for count in class_counts.values()):
        break

    train_set = Subset(full_train_set, selected_indices)

    B = 128 #batch size
    lr = 0.001 #learning rate
    epochs = 300 #number of epochs

    train_loader = DataLoader(train_set, batch_size=B, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=B, shuffle=False, num_workers=2)

    print(f"Total train_samples: {len(train_set)}")
    print(f"Total test_samples: {len(test_set)}")

    # model = torchvision.models.resnet18(weights=None)
    #   model = torchvision.models.resnet34(weights=None)
    model = torchvision.models.resnet50(weights=None)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)    
    model.apply(init_weights)
    model.to(device)

    scheduler=None
    # optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1,
    #             momentum=0.9, weight_decay=5e-4)
    # milestones = '60,120,160'
    # scheduler = lr_scheduler.MultiStepLR(optimizer,
    #             milestones=[int(e) for e in milestones.split(',')], gamma=0.2)

    print('*'*100)
    model, train_loss_random_erase, train_acc_random_erase, test_acc_random_erase = train_model_random_erase(
        model, criterion, optimizer, train_loader, test_loader, epochs, device, patience=10, min_delta=0.0, p=0.5
    )

    # Save results for comparison
    data_to_save = {
        "model": model,
        "train_loss_history": train_loss_random_erase,
        "train_acc_history": train_acc_random_erase,
        "val_acc_history": test_acc_random_erase
    }

    with open("data_pkl/resnet50_random_erase_data.pkl", "wb") as f:
        pickle.dump(data_to_save, f)


if __name__ == '__main__':
    main()