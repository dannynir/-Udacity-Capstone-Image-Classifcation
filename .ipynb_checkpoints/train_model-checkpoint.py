#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import os
import sys
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import time
import argparse


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


#TODO: Import dependencies for Debugging andd Profiling
#from sagemaker_debugger import Hook
#hook = smd.Hook.create_from_json_file()
from smdebug.pytorch import get_hook,Hook
from smdebug import modes
from smdebug.profiler.utils import str2bool



def test(model, test_loader, criterion, device, hook=None):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    if hook:
        hook.set_mode(modes.EVAL)

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            total += len(inputs)

    avg_loss = running_loss / total
    avg_acc = running_corrects / total
    logger.info(f"test_loss: {avg_loss:.4f}, Test Accuracy: {avg_acc:.4f}")

def train(model, train_loader, criterion, optimizer, device, epochs, hook=None):
    model.to(device)
    model.train()

    if hook:
        #hook.register_hook(model)
        hook.set_mode(modes.TRAIN)

    for epoch in range(epochs):
        logger.info(f"Epoch: {epoch+1}/{epochs}")
        running_loss = 0.0
        running_corrects = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            total += len(inputs)

        epoch_loss = running_loss / total
        epoch_acc = running_corrects / total
        logger.info(f"Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.4f}")

    return model


    
def net(num_classes):
    model = models.resnet50(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
        
    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features,num_classes))
    return model    

def create_data_loaders(data_dir, batch_size):
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = torchvision.datasets.ImageFolder(test_dir, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, len(train_dataset.classes)
   

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Hyperparameters - Batch size: {args.batch_size}, Learning rate: {args.learning_rate}, Epochs: {args.epochs}")
    
    logger.info(f'Reading data from: {args.data_dir}')
    train_loader, test_loader, num_classes = create_data_loaders(args.data_dir, args.batch_size)

    logger.info(f'Number of classes: {num_classes}')
    model = net(num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)

    hook = Hook.create_from_json_file()
    hook.register_hook(model)
    hook.register_loss(criterion)

    logger.info("Kick starting model training...")
    model = train(model, train_loader, criterion, optimizer, device, args.epochs, hook)

    logger.info("Kick starting model testing...")
    test(model, test_loader, criterion, device, hook)

    logger.info(f"Saving model to {args.model_dir}/model.pth")
    path = os.path.join(args.model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)


if __name__=='__main__':
    parser=argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32)
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="number of epochs to train (default: 5)",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.01
    )

    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    
    args=parser.parse_args()
    
    main(args)
