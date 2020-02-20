import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.detection import fasterrcnn_resnet50_fpn, maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


class SmallClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 512, 3, padding=1)
        self.conv4 = nn.Conv2d(512, 128, 3, padding=1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
        x = self.fc(x)
        return x


class SmallInstanceSegmentor(nn.Module):
    def __init__(self,
                 num_classes,
                 hidden_layer=256,
                 pretrained=True):
        super().__init__()
        # load an instance segmentation model pre-trained pre-trained on COCO
        model = maskrcnn_resnet50_fpn(pretrained=pretrained)

        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                           hidden_layer,
                                                           num_classes)
        self.model = model

    def forward(self, *input, **kwargs):
        return self.model(*input, **kwargs)


if __name__ == '__main__':
    def run_segmentor():
        model = SmallInstanceSegmentor(num_classes=2, pretrained=True)
        model.eval()
        x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        predictions = model(x)


    run_segmentor()
