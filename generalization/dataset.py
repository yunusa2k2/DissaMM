from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from PIL import Image
import os
import torch

class COCOLoader(Dataset):
    def __init__(self, img_dir, ann_file, transform=None):
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.transform = transform
        self.ids = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]

        # Load image info
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # Use first caption
        caption = anns[0]["caption"]

        # Load category (supercategory)
        cat_id = anns[0]["category_id"]
        cat = self.coco.loadCats(cat_id)[0]["supercategory"]

        # Load image file
        path = self.coco.loadImgs(img_id)[0]["file_name"]
        img_path = os.path.join(self.img_dir, path)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, caption, cat