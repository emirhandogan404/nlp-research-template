from typing import TYPE_CHECKING, Tuple
import os
from print_on_steroids import logger
from PIL import Image

import torch
import random

# import time

# import model

# from torch import nn
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision.transforms.functional import to_tensor, resize, pad
from torchvision.transforms import RandAugment
import torchvision.transforms as transforms

from torchvision.datasets import MNIST

# from torchvision import transforms
import lightning as L

# from dlib.frameworks.pytorch import get_rank

if TYPE_CHECKING:
    from train import TrainingArgs


def get_train_transforms():
    # TODO: add RandAugment # use magnitude 5 and 3 operations
    return RandAugment(num_ops=3, magnitude=5)


class MNISTDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.data_list = []
        for dir in os.listdir(img_dir):
            mylist = []
            for file in os.listdir(os.path.join(img_dir, dir)):
                mylist.append(os.path.join(dir, file))
            self.data_list.append(mylist)
        self.data_list.sort()

    def __len__(self):
        return sum(len(x) for x in self.data_list)

    def __getitem__(self, index):
        anchor = self.data_list[index[0][0]][index[0][1]]
        positive = self.data_list[index[1][0]][index[1][1]]
        negative = self.data_list[index[2][0]][index[2][1]]

        anchor = Image.open(os.path.join(self.img_dir,anchor))
        positive = Image.open(os.path.join(self.img_dir,positive))
        negative = Image.open(os.path.join(self.img_dir,negative))
        
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative

class MNISTSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __iter__(self):
        self.final = []
        self.idxlist = random.sample(range(len(self.dataset.data_list)), len(self.dataset.data_list))
        random.shuffle(self.idxlist)
        for i in self.idxlist:
            for j in range(len(self.dataset.data_list[i])):
                try:
                    p = random.choice(list(set([x for x in range(len(self.dataset.data_list[i]))]) - set([j])))
                except IndexError:
                    p = 0
                n = random.choice(list(set([x for x in range(len(self.dataset.data_list))]) - set([i])))
                n2 = random.randint(0, len(self.dataset.data_list[n])-1)
                self.final.append(((i, j), (i, p), (n, n2)))
        random.shuffle(self.final)
        return iter(self.final)

    def __len__(self):
        return self.dataset.__len__()


# Annahme: Daten liegen alle in einem Ordner (data_dir)
# Annahme: Bilder sind bereits auf Gesichter zugeschnitten
# Annahme: Name der Dateien entspricht Schema: <name_individuum>-<nr>-img-<nr>.jpg
class TripletDataset(Dataset):
    def __init__(self, img_dir, target_size, transform=None):
        self.img_dir = img_dir
        self.target_size = target_size
        self.transform = transform
        self.data_list = []
        for dir in os.listdir(img_dir):
            mylist = []
            for file in os.listdir(os.path.join(img_dir, dir)):
                mylist.append(os.path.join(dir, file))
            self.data_list.append(mylist)
        self.data_list.sort()

    def __len__(self):
        return sum(len(x) for x in self.data_list)

    def __getitem__(self, index):
        anchor = self.data_list[index[0][0]][index[0][1]]
        positive = self.data_list[index[1][0]][index[1][1]]
        negative = self.data_list[index[2][0]][index[2][1]]

        anchor = Image.open(os.path.join(self.img_dir,anchor))
        positive = Image.open(os.path.join(self.img_dir,positive))
        negative = Image.open(os.path.join(self.img_dir,negative))
        
        pad_width = max(self.target_size[0] - anchor.width, 0)
        pad_height = max(self.target_size[1] - anchor.height, 0)
        
        anchor = pad(anchor, padding=(0, 0, pad_width, pad_height), fill=0)

        pad_width = max(self.target_size[0] - positive.width, 0)
        pad_height = max(self.target_size[1] - positive.height, 0)
        
        positive = pad(positive, padding=(0, 0, pad_width, pad_height), fill=0)
        
        pad_width = max(self.target_size[0] - negative.width, 0)
        pad_height = max(self.target_size[1] - negative.height, 0)
        
        negative = pad(negative, padding=(0, 0, pad_width, pad_height), fill=0)
        
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative

class TripletSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __iter__(self):
        self.final = []
        self.idxlist = random.sample(range(len(self.dataset.data_list)), len(self.dataset.data_list))
        random.shuffle(self.idxlist)
        for i in self.idxlist:
            for j in range(len(self.dataset.data_list[i])):
                try:
                    p = random.choice(list(set([x for x in range(len(self.dataset.data_list[i]))]) - set([j])))
                except IndexError:
                    p = 0
                n = random.choice(list(set([x for x in range(len(self.dataset.data_list))]) - set([i])))
                n2 = random.randint(0, len(self.dataset.data_list[n])-1)
                self.final.append(((i, j), (i, p), (n, n2)))
        random.shuffle(self.final)
        return iter(self.final)

    def __len__(self):
        return self.dataset.__len__()


# read all directories in data_dir (a directory corresponds to one individual)
# returns a list of all image files in data_dir prefixed with the individual's name (and -)


def read_image_files(data_dir):
    individuals = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    image_files = []
    for individual in individuals:
        image_files.extend([individual + "-" + f for f in os.listdir(os.path.join(data_dir, individual))])

    return image_files


class GorillaDM(L.LightningDataModule):
    def __init__(
        self,
        training_args: "TrainingArgs",
        transform=get_train_transforms(),
    ):
        super().__init__()
        self.args = training_args
        self.train_dir = training_args.train_dir
        self.val_dir = training_args.val_dir
        self.test_dir = training_args.test_dir
        self.transform = transform

        logger.debug(f"Train data dir: {self.train_dir}")

        # self.local_rank = get_rank()

    # single gpu -> used for downloading and preprocessing which cannot be parallelized
    def prepare_data(self) -> None:
        pass  # nothing to do here

    def setup(self, stage):
        # TODO: implement train/val split
        # self.train_dataset = BristolDataset(data_dir=self.train_dir, transform=self.transform)
        # self.val_dataset = BristolDataset(data_dir=self.val_dir, transform=self.transform)
        # self.train_sampler = BristolSampler(data_dir=self.train_dir)
        # self.val_sampler = BristolSampler(data_dir=self.val_dir)

        self.train_dataset = TripletDataset("./data/gorilla_experiment_splits/k-fold-splits/cxl-bristol_face-openset=False_0/train", (224, 224), transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]))
        self.val_dataset = val = TripletDataset("./data/gorilla_experiment_splits/k-fold-splits/cxl-bristol_face-openset=False_0/database_set", (224, 224), transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]))
        self.train_sampler = TripletSampler(self.train_dataset)
        self.val_sampler = TripletSampler(self.val_dataset)

    def train_dataloader(self):
        common_args = dict(
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
            persistent_workers=(
                True if self.args.workers > 0 else False
            ),  # https://discuss.pytorch.org/t/what-are-the-dis-advantages-of-persistent-workers/102110/10
            pin_memory=True,
        )
        return DataLoader(self.train_dataset, sampler=self.train_sampler, **common_args)

    def val_dataloader(self):
        common_args = dict(
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
            persistent_workers=(
                True if self.args.workers > 0 else False
            ),  # https://discuss.pytorch.org/t/what-are-the-dis-advantages-of-persistent-workers/102110/10
            pin_memory=True,  # TODO: understand this # short answer: aims to optimize data transfer between the CPU and GPU
        )
        return DataLoader(self.val_dataset, sampler=self.val_sampler, **common_args)


if __name__ == "__main__":
    import model

    # set seed
    torch.manual_seed(42)
    random.seed(45)
    # test sampler and dataset with a single batch
    # data_dir = "/workspaces/gorillavision/datasets/face_detection/all_images_no_cropped_backup"
    # data_dir = "/workspaces/gorillavision/datasets/cxl/face_images_grouped"
    # sampler = BristolSampler(data_dir)
    # dataset = BristolDataset(data_dir=data_dir)
    # dataloader = DataLoader(dataset, sampler=sampler, batch_size=1, num_workers=4)
    # print(len(dataset))
    # batch = next(iter(dataloader))
    # # save an image
    # img = batch[0][0].permute(1, 2, 0)
    # Image.fromarray((img.numpy() * 255).astype("uint8")).save("test.jpg")

    model1 = model.EfficientNetV2Wrapper(
        model_name_or_path="",
        from_scratch=True,
        learning_rate=0.01,
        weight_decay=0,
        lr_schedule="lambda",
        warmup_epochs=1,
        lr_decay=0.99,
        lr_decay_interval=2,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        save_hyperparameters=True,
    )
    print(model1.model)

    # test dataloader
    # batch = next(iter(dataloader))
    # loss = model1._calculate_loss(batch)
    # # print(loss)

    # start_time = time.time()
    # for batch in dataloader:
    #     continue
    # end_time = time.time()
    # print(f"Time: {end_time - start_time} seconds")

    # test MNIST stuff
    sampler = MNISTSampler()
    dataset = MNISTDataset()
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=1, num_workers=0)

    # test model output shape
    batch = next(iter(dataloader))
    out = model1(batch[0])
    out2 = model1.model(batch[0])
    print(out.shape)
    print(out2.shape)

    # print(len(dataset))
    # random.seed(45)
    # batch = next(iter(dataloader))
    # # save an image
    # anchor_img, positive_img, negative_img = batch
    # # anchor_img = anchor_img[0]
    # print(anchor_img.shape)
    # img = anchor_img[0]  # imgs are grayscale
    # img = img.permute(1, 2, 0)
    # img = img.squeeze()
    # Image.fromarray((img.numpy() * 255).astype("uint8")).save("test1.jpg")

    # img = positive_img[0]  # imgs are grayscale
    # img = img.permute(1, 2, 0)
    # img = img.squeeze()
    # Image.fromarray((img.numpy() * 255).astype("uint8")).save("test2.jpg")

    # img = negative_img[0]  # imgs are grayscale
    # img = img.permute(1, 2, 0)
    # img = img.squeeze()
    # Image.fromarray((img.numpy() * 255).astype("uint8")).save("test3.jpg")

    # # iterate over the dataloader to see if it works
    # start_time = time.time()
    # for batch in dataloader:
    #     continue
    # end_time = time.time()
    # print(f"Time: {end_time - start_time} seconds")