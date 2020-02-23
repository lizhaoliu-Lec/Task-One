import torch
import torchvision
import torchvision.transforms as transforms

from trainers import ImageClassifierTrainer
from utils import join_path
from stand_alone_self_attention import ResNet26


def main():
    #########################
    # (0) hard code configs #
    #########################
    DATA_BASE_DIR = join_path('datasets', 'cifar10')
    DATA_SET_NAME = 'CIFAR10'
    BATCH_SIZE = 8
    NUM_WORKERS = 8
    TRAIN_STEPS = 1000
    VAL_EVERY = 100
    LOG_EVERY = 50
    NAME = 'SmallClassifier'
    CUDA = 3
    RUN_ID = 'example'
    PLOT_EVERY = 500
    NUM_CLASSES = 10
    DEBUG = True

    #######################
    # (1) Define datasets #
    #######################
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = torchvision.datasets.CIFAR10(root=DATA_BASE_DIR, train=True,
                                             download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE,
                                               shuffle=True, num_workers=NUM_WORKERS)

    test_set = torchvision.datasets.CIFAR10(root=DATA_BASE_DIR, train=False,
                                            download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE,
                                              shuffle=False, num_workers=NUM_WORKERS)

    ####################
    # (2) Define model #
    ####################
    model = ResNet26(num_classes=NUM_CLASSES)

    ###############################################################
    # (3) Define loss function inside the Trainer's step function #
    ###############################################################

    ########################
    # (4) Define optimizer #
    ########################
    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=0.1, momentum=0.9, weight_decay=1e-4)

    ####################
    # (5) Init trainer #
    ####################
    trainer = ImageClassifierTrainer(model=model,
                                     name=NAME,
                                     optimizer=optimizer,
                                     dataset_name=DATA_SET_NAME,
                                     train_loader=train_loader,
                                     train_steps=TRAIN_STEPS,
                                     val_every=VAL_EVERY,
                                     val_loader=test_loader,  # use test set for validation
                                     log_every=LOG_EVERY,
                                     run_id=RUN_ID,
                                     cuda=CUDA,
                                     plot_every=PLOT_EVERY,
                                     debug=DEBUG)

    ######################
    # (6) Begin Training #
    ######################
    trainer.train()


if __name__ == '__main__':
    main()
