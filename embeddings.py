import wandb
from src.model import EfficientNetV2Wrapper
from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd
import torchvision.transforms as transforms
import torch
from PIL import Image
from torchvision.transforms.functional import pad

class GorillaDataset(Dataset):
    def __init__(self, img_dir, target_size, transform=None):
        self.img_dir = img_dir
        self.target_size = target_size
        self.transform = transform
        self.data_list = []
        for dir in os.listdir(img_dir):
            for file in os.listdir(os.path.join(img_dir, dir)):
                self.data_list.append((os.path.join(dir, file), str(dir)))
        self.data_list.sort()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        item = self.data_list[index][0]
        label = self.data_list[index][1]

        item = Image.open(os.path.join(self.img_dir,item))
        
        pad_width = max(self.target_size[0] - item.width, 0)
        pad_height = max(self.target_size[1] - item.height, 0)
        
        item = pad(item, padding=(0, 0, pad_width, pad_height), fill=0)
        
        if self.transform:
            item = self.transform(item)

        return item, label
    
run = wandb.init(project="embedding_test_gorillas", name="efficientnet_gorillas")

model = EfficientNetV2Wrapper.load_from_checkpoint("logs/EfficientNetEmirhan/2xjrvg1b/checkpoints/last_model_ckpt.ckpt")
model = model.to("cpu")

dataset = GorillaDataset("./data/gorilla_experiment_splits/k-fold-splits/cxl-bristol_face-openset=False_0/database_set", (224, 224), transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]))
dataloader = DataLoader(dataset, batch_size=1)

columns = ["value_{}".format(index) for index in range(0, 256)]
columns = ["target"] + columns
df = pd.DataFrame(columns=columns)
df["target"] = df["target"].astype("category")

with torch.no_grad():
    for i, batch in enumerate(dataloader):
        if i >= 1000:
            break
        a, lbl = batch
        embedding = model(a)
        embedding = embedding[0].numpy().tolist()
        
        new_row = [lbl, *embedding]

        df = pd.concat([pd.DataFrame([new_row], columns=columns), df], ignore_index=True)

table = wandb.Table(columns=df.columns.to_list(), data=df.values, )
wandb.log({"mnist_embeddings": table})
wandb.finish()