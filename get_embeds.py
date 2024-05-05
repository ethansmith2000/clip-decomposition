import torch
from PIL import Image
import os
import pandas as pd
import transformers
from tqdm import tqdm
import time

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, clip_path):
        self.data_dir = data_dir
        self.transform = transformers.CLIPFeatureExtractor.from_pretrained(clip_path, subfolder="image_processor")
        self.image_paths = [os.path.join(data_dir,f) for f in os.listdir(data_dir)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert("RGB")
            image = self.transform(image, return_tensors="pt")["pixel_values"].to(torch.float16)
            return image
        except Exception as e:
            return None


def collate_fn(batch):
    batch = [b[0] for b in batch if b is not None]
    return torch.stack(batch)

def time_fn(fn, fn_name, *args, **kwargs):
    start = time.time()
    ret = fn(*args, **kwargs)
    end = time.time()
    print(f"{fn_name} took {end - start} seconds")
    return ret


if __name__ == '__main__':
    data_dir = "images"
    clip_path = "kandinsky-community/kandinsky-2-2-prior"
    batch_size = 1024
    num_workers = 8
    dataset = ImageDataset(data_dir, clip_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True, pin_memory_device="cuda")
    clip_model = transformers.CLIPVisionModelWithProjection.from_pretrained("kandinsky-community/kandinsky-2-2-prior", subfolder="image_encoder").to("cuda").to(torch.float16)

    embeds = []
    start_load = None
    for batch in tqdm(dataloader):
        end_load = time.time()
        if start_load is not None:
            print("load time", end_load - start_load)
        with torch.no_grad():
            batch = time_fn(lambda x: x.cuda(non_blocking=True), "to cuda", batch)
            emb = time_fn(lambda x: clip_model(x).image_embeds, "forward", batch)
            embeds.append(emb)
        start_load = time.time()

    embeds = torch.cat(embeds).float().cpu()
    torch.save(embeds, "embeds.pt")



