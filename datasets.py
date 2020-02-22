import os
import zipfile
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.datasets import cifar
from torchvision.datasets.utils import download_url
from torchtext.datasets.text_classification import download_from_url, URLS, extract_archive, build_vocab_from_iterator
from torchtext.datasets.text_classification import _csv_iterator, Vocab, _create_data_from_iterator
from torchtext.datasets.text_classification import TextClassificationDataset

from PIL import Image

__all__ = [
    'cifar', 'PennFudanDataset',
    'AGNews',
]


def download_extract(url, root, filename, md5):
    download_url(url, root, filename, md5)
    filepath = os.path.join(root, filename)
    if zipfile.is_zipfile(filepath):
        fz = zipfile.ZipFile(filepath, 'r')
        for file in fz.namelist():
            fz.extract(file, root)
    else:
        print('Can not extract %s' % filepath)


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


class PennFudanDataset(object):
    """
    Penn-Fudan Database for Pedestrian Detection and Segmentation.
    It contains 170 images with 345 instances of pedestrians, and
    we will use it to illustrate how to use the new features in torchvision in order to
    train an instance segmentation model on a custom dataset.
    """

    def __init__(self,
                 root,
                 transforms,
                 train=True,
                 download=False):
        dataset_url = 'https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip',
        filename = 'PennFudanPed.zip'
        self.root = os.path.join(root, "PennFudanPed")
        self.transforms = transforms
        if download:
            download_extract(dataset_url, root, filename, md5=None)

        is_dir = os.path.isdir
        if not (is_dir(os.path.join(self.root, "PNGImages"))
                and is_dir(os.path.join(self.root, "PedMasks"))):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.train = train  # training set or test set

        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(self.root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(self.root, "PedMasks"))))

        if self.train:
            self.imgs = self.imgs[:-50]
            self.masks = self.masks[:-50]
        else:
            self.imgs = self.imgs[-50:]
            self.masks = self.masks[-50:]

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def _setup_datasets(dataset_name=None,
                    root='.data',
                    ngrams=1,
                    vocab=None,
                    include_unk=False,
                    downloaded_name=None):
    if downloaded_name is None:
        dataset_tar = download_from_url(URLS[dataset_name], root=root)
    else:
        dataset_tar = os.path.join(root, downloaded_name)
    extracted_files = extract_archive(dataset_tar)

    for fname in extracted_files:
        if fname.endswith('train.csv'):
            train_csv_path = fname
        if fname.endswith('test.csv'):
            test_csv_path = fname

    if vocab is None:
        print('Building Vocab based on {}'.format(train_csv_path))
        vocab = build_vocab_from_iterator(_csv_iterator(train_csv_path, ngrams))
    else:
        if not isinstance(vocab, Vocab):
            raise TypeError("Passed vocabulary is not of type Vocab")
    print('Vocab has {} entries'.format(len(vocab)))
    print('Creating training data')
    train_data, train_labels = _create_data_from_iterator(
        vocab, _csv_iterator(train_csv_path, ngrams, yield_cls=True), include_unk)
    print('Creating testing data')
    test_data, test_labels = _create_data_from_iterator(
        vocab, _csv_iterator(test_csv_path, ngrams, yield_cls=True), include_unk)
    if len(train_labels ^ test_labels) > 0:
        raise ValueError("Training and test labels don't match")
    return (TextClassificationDataset(vocab, train_data, train_labels),
            TextClassificationDataset(vocab, test_data, test_labels))


def AGNews(root, *args, **kwargs):
    if not os.path.exists(root):
        os.makedirs(root)
    return _setup_datasets(root=root, downloaded_name='ag_news_csv.tar.gz', *args, **kwargs)


if __name__ == '__main__':
    def run_download_PennFudanDataset():
        d = PennFudanDataset(root=os.path.join('datasets', 'PennFudan'),
                             transforms=get_transform(train=True),
                             download=True)
        print('length of dataset: %d' % len(d))


    def run_download_AGNews():
        train_dataset, test_dataset = AGNews(root=os.path.join('datasets', 'AGNews'))
        VOCAB_SIZE = len(train_dataset.get_vocab())
        NUM_CLASS = len(train_dataset.get_labels())
        print('VOCAB', VOCAB_SIZE)
        print('NUM class', NUM_CLASS)
        vocab = train_dataset.get_vocab()
        print('vocab stoi', vocab.stoi)
        print('vocab stoi pad', vocab.stoi['<pad>'])


    run_download_AGNews()
