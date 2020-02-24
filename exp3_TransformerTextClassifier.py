import torch
from torch.utils.data import DataLoader

from trainers import TextClassifierTrainer
from utils import join_path
from datasets import AGNews
from models import SmallTransformer


def main():
    #########################
    # (0) hard code configs #
    #########################
    DATA_BASE_DIR = join_path('datasets', 'AGNews')
    BATCH_SIZE = 8
    TRAIN_STEPS = 10000
    VAL_EVERY = 100
    LOG_EVERY = 100
    NAME = 'SmallTransformer'
    CUDA = 3
    RUN_ID = 'example'
    DEBUG = False
    NUM_WRITE = 4
    WRITE_EVERY = 100
    DATASET_NAME = 'AGNews'

    #######################
    # (1) Define datasets #
    #######################
    def generate_batch(batch):
        label = torch.tensor([entry[0] for entry in batch])
        text = [entry[1] for entry in batch]
        PAD_IDX = 1
        max_len = max([len(t) for t in text])
        text = [torch.cat([t, torch.tensor([PAD_IDX] * (max_len - len(t)))]) if len(t) < max_len else t for t in text]
        text = torch.stack(text, dim=0)
        return text, label

    train_dataset, test_dataset = AGNews(root=DATA_BASE_DIR)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=generate_batch)

    ####################
    # (2) Define model #
    ####################
    VOCAB_SIZE = len(train_dataset.get_vocab())
    NUM_CLASSES = len(train_dataset.get_labels())
    model = SmallTransformer(vocab_size=VOCAB_SIZE, num_classes=NUM_CLASSES)

    ###############################################
    # (3) Define loss function inside the trainer #
    ###############################################

    ########################
    # (4) Define optimizer #
    ########################
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    ####################
    # (5) Init trainer #
    ####################
    trainer = TextClassifierTrainer(model=model,
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
                                    debug=DEBUG,
                                    num_write=NUM_WRITE,
                                    write_every=WRITE_EVERY,
                                    vocab=train_dataset.get_vocab(),
                                    dataset_name=DATASET_NAME)

    ######################
    # (6) Begin Training #
    ######################
    trainer.train()


if __name__ == '__main__':
    main()
