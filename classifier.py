import os

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms


class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        if args.detector_model == "resnet50":
            from torchvision.models import resnet50, ResNet50_Weights
            self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Sequential(nn.Linear(num_ftrs, args.class_count))
        elif args.detector_model == "resnet34":
            from torchvision.models import resnet34, ResNet34_Weights
            self.model = resnet34(weights=ResNet34_Weights.DEFAULT)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Sequential(nn.Linear(num_ftrs, args.class_count))
        elif args.detector_model == "vgg19bn":
            from torchvision.models import vgg19_bn, VGG19_BN_Weights
            self.model = vgg19_bn(weights=VGG19_BN_Weights.DEFAULT)
            num_ftrs = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Sequential(nn.Linear(num_ftrs, args.class_count))
        elif args.detector_model == "vgg16":
            from torchvision.models import vgg16, VGG16_Weights
            self.model = vgg16(weights=VGG16_Weights.DEFAULT)
            num_ftrs = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Sequential(nn.Linear(num_ftrs, args.class_count))

        if args.detector_path != "":
            checkpoint = torch.load(args.detector_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])

    def forward(self, x):
        x = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).forward(x)
        return F.softmax(self.model(x), dim=1)


class ClassifierToTrain(nn.Module):
    def __init__(self, args):
        super(ClassifierToTrain, self).__init__()
        if args.detector_model == "resnet50":
            from torchvision.models import resnet50, ResNet50_Weights
            self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Sequential(nn.Linear(num_ftrs, args.class_count))
        elif args.detector_model == "resnet34":
            from torchvision.models import resnet34, ResNet34_Weights
            self.model = resnet34(weights=ResNet34_Weights.DEFAULT)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Sequential(nn.Linear(num_ftrs, args.class_count))
        elif args.detector_model == "vgg19bn":
            from torchvision.models import vgg19_bn, VGG19_BN_Weights
            self.model = vgg19_bn(weights=VGG19_BN_Weights.DEFAULT)
            num_ftrs = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Sequential(nn.Linear(num_ftrs, args.class_count))
        elif args.detector_model == "vgg16":
            from torchvision.models import vgg16, VGG16_Weights
            self.model = vgg16(weights=VGG16_Weights.DEFAULT)
            num_ftrs = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Sequential(nn.Linear(num_ftrs, args.class_count))


        if args.detector_path != "":
            checkpoint = torch.load(args.detector_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])

        self.path = args.detector_save_path

    def forward(self, x):
        x = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).forward(x)
        return self.model(x)

    def save(self, optimizer, epoch, i):
        checkpoint = {"model_state_dict": self.model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(),
                      "epoch": epoch,
                      "i": i}
        torch.save(checkpoint, os.path.join(self.path, "_{}_{}_.pkl".format(epoch, i)))