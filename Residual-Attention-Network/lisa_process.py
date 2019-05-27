from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import torchvision
from torchvision import transforms, datasets, models
import os
import cv2
import time
# from model.residual_attention_network_pre import ResidualAttentionModel
# based https://github.com/liudaizong/Residual-Attention-Network
from model.residual_attention_network import ResidualAttentionModel_92_32input_update as ResidualAttentionModel
from lisa_dataset import LisaDataset as lisa_dataset

model_file = 'lisa_model_92_mixup300_normal20.pkl'
split_ratio_train = 0.8
is_train = True
is_pretrain = False
train_batch_size = 64
test_batch_size = 20

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def test(model, test_loader, model_file=None):
    if model_file is not None:
        model.load_state_dict(torch.load(model_file))
    model.eval()

    correct = 0
    total = 0
    
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))

    for images, labels in test_loader:
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()
        
        c = (predicted == labels.data).squeeze()
        for i in range(len(labels.data)):
            label = labels.data[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

    print('Accuracy of the model on the test images: %.2f %% (%d/%d)' % (100 * float(correct) / total, correct, total))
    # print('Accuracy of the model on the test images:', float(correct)/total)
    for i in range(len(classes)):
        print('Accuracy of %5s : %.2f %% (%d/%d)' % (
            classes[i], 100 * float(class_correct[i]) / class_total[i], class_correct[i], class_total[i]))
    return float(correct) / total # compatible with Python 2.7

if __name__ == '__main__':
    # Image Preprocessing
    data_transform = transforms.ToTensor()
    
    # dataset
    lisa_dataset = lisa_dataset(root = './Residual-Attention-Network/annotations/', # use root = './annotations/' for serve
                                transform = data_transform)
    total_num = len(lisa_dataset)
    train_num = int(round(split_ratio_train * total_num)) # compatible with Python 2.7
    test_num = total_num - train_num
    lisa_datasets = random_split(lisa_dataset, [ train_num, test_num ])
    train_dataset = lisa_datasets[0]
    test_dataset = lisa_datasets[1]

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=train_batch_size, 
                                               shuffle=True, 
                                               num_workers=8)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=test_batch_size, 
                                              shuffle=False)

    classes = list(lisa_dataset.categories.keys())
    model = ResidualAttentionModel().cuda()
    print(model)

    is_mixup = True
    lr = 0.1  # 0.1
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=0.0001)
    acc_best = 0
    total_epoch = 320
    if is_train is True:
        if is_pretrain is True:
            model.load_state_dict((torch.load(model_file)))
        # Training
        for epoch in range(total_epoch):
            print("Epoch [%d/%d]:" % (epoch+1, total_epoch))
            if epoch > 300:
                is_mixup = False
            else:
                is_mixup = True
            model.train()
            tims = time.time()
            for i, (images, labels) in enumerate(train_loader):
                if is_mixup:
                    inputs, targets_a, targets_b, lam = mixup_data(images.cuda(), labels.cuda(), alpha=1.0)
                    inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
                    outputs = model(inputs)
                    optimizer.zero_grad()
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                    loss.backward()
                    optimizer.step()
                else:
                    images = Variable(images.cuda())
                    # print(images.data)
                    labels = Variable(labels.cuda())
                    # Forward + Backward + Optimize
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                if (i+1) % 10 == 0 or (i+1) == len(train_loader) or i == 0:
                    print("Epoch [%d/%d], Iter [%d/%d] Loss: %.4f" %(epoch+1, total_epoch, i+1, len(train_loader), loss.data))
            print('the epoch takes time:',time.time() - tims)
            print('evaluate test set:')
            acc = test(model, test_loader)
            if acc > acc_best:
                acc_best = acc
                print('current best acc,', acc_best)
                torch.save(model.state_dict(), model_file)
            # Decaying Learning Rate
            if (epoch+1) / float(total_epoch) == 0.3 or (epoch+1) / float(total_epoch) == 0.6 or (epoch+1) / float(total_epoch) == 0.9:
                lr /= 10
                print('reset learning rate to:', lr)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                    print(param_group['lr'])
                #optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                #optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=0.0001)

        # Save the Model (epoch==320)
        #torch.save(model.state_dict(), model_file)

        # After training test
        print('Training Completed.')
        print('The best accuracy is %.2f %%' % (100 * float(acc_best)))

        # model.load_state_dict(torch.load(model_file))
        print('The best accuracy of the model on the train set:')
        test(model, train_loader, model_file)
        print('The best accuracy of the model on the test set:')
        test(model, test_loader, model_file)

    else: # is_train is False
        test(model, test_loader, model_file)
