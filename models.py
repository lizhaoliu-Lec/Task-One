import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer

from torchvision.models.detection import maskrcnn_resnet50_fpn
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


class SmallDetectorAndSegmentor(nn.Module):
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


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class SmallTransformer(nn.Module):

    def __init__(self,
                 vocab_size,
                 num_classes,
                 d_model=128,
                 num_head=2,
                 dim_feedforward=256,
                 num_layer=2,
                 drop_rate=0.5,
                 init_range=0.1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, drop_rate)
        encoder_layers = TransformerEncoderLayer(d_model, num_head, dim_feedforward, drop_rate)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layer)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.decoder = nn.Linear(d_model, num_classes)
        self.init_range = init_range

        self.init_weights()

    def init_weights(self):
        init_range = self.init_range
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        encoding = self.transformer_encoder(x)
        encoding = torch.mean(encoding, dim=1, keepdim=False)
        out = self.decoder(encoding)
        return out


if __name__ == '__main__':
    def run_segmentor():
        model = SmallDetectorAndSegmentor(num_classes=2, pretrained=True)
        model.eval()
        x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        predictions = model(x)


    def run_transformer():
        model = SmallTransformer(vocab_size=20, num_classes=5)
        print(model)
        x = torch.randint(high=19, size=(5, 4))
        print(x)
        out = model(x)
        print(out)


    run_transformer()
