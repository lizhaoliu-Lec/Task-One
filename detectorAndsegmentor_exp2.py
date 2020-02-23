import torch
import torch.nn as nn
import torchvision

from trainers import InstanceSegmentorTrainer
from utils import join_path
from datasets import PennFudanDataset
from models import SmallDetectorAndSegmentor
from references.detection.utils import collate_fn
import references.detection.transforms as transforms


def main():
    #########################
    # (0) hard code configs #
    #########################
    DATA_BASE_DIR = join_path('datasets', 'PennFudan')
    BATCH_SIZE = 2
    NUM_WORKERS = 4
    TRAIN_STEPS = 200
    _NUM_EPOCHS = 10
    VAL_EVERY = 100
    LOG_EVERY = 10
    NAME = 'SmallInstanceSegmentor'
    CUDA = 0
    RUN_ID = 'example'
    NUM_CLASSES = 2
    DEBUG = True

    #######################
    # (1) Define datasets #
    #######################
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.RandomHorizontalFlip(0.5)])

    train_set = PennFudanDataset(root=DATA_BASE_DIR, transforms=transform, train=True,
                                 download=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE,
                                               shuffle=True, num_workers=NUM_WORKERS,
                                               collate_fn=collate_fn)

    test_set = PennFudanDataset(root=DATA_BASE_DIR, transforms=transform, train=False,
                                download=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE,
                                              shuffle=False, num_workers=NUM_WORKERS,
                                              collate_fn=collate_fn)

    ####################
    # (2) Define model #
    ####################
    model = SmallDetectorAndSegmentor(num_classes=NUM_CLASSES)

    ############################
    # (3) Define loss function #
    ############################

    ########################
    # (4) Define optimizer #
    ########################
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    ####################
    # (5) Init trainer #
    ####################
    trainer = InstanceSegmentorTrainer(model=model,
                                       name=NAME,
                                       optimizer=optimizer,
                                       lr_scheduler=lr_scheduler,
                                       train_loader=train_loader,
                                       train_steps=TRAIN_STEPS,
                                       val_every=VAL_EVERY,
                                       val_loader=test_loader,  # use test set for validation
                                       log_every=LOG_EVERY,
                                       run_id=RUN_ID,
                                       cuda=CUDA,
                                       debug=DEBUG)

    ######################
    # (6) Begin Training #
    ######################
    trainer.train()


if __name__ == '__main__':
    main()
