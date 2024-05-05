import aiohttp
import asyncio
import os
import argparse
import pandas as pd
from io import BytesIO
from PIL import Image
import math
from tqdm import tqdm

async def download_image(session, url, filename, idx, pbar):
    try:
        async with session.get(url) as resp:
            if resp.status != 200:
                print(f"Failed to download {url}. Status: {resp.status}","file:",idx)
                pbar.update(1)
                return

            image_data = await resp.read()
            with open(filename, 'wb') as f:
                f.write(image_data)

            pbar.update(1)

    except Exception as e:
        print(f"Error downloading {url}: {e}")


async def main(args):
    os.makedirs(args.root, exist_ok=True)

    df = pd.read_parquet(args.df_path)
    urls = list(df["url"])
    digits = len(str(len(urls)))
    tasks = []

    filename_column_name = None
    extension = ".png"

    file_num = 0
    if filename_column_name is None:
        file_exists=True
        while file_exists:
            filename = os.path.join(args.root, str(file_num).zfill(digits) + extension)
            if os.path.exists(filename):
                file_num += 1
            else:
                file_exists=False

        df["filename"] = [str(i+file_num).zfill(digits)+extension for i in range(len(urls))]
        df.to_csv(args.df_path, index=False)

    # Create a single ClientSession
    async with aiohttp.ClientSession() as session:
        pbar = tqdm(total=len(urls), desc="Downloading", ncols=100)

        for i, url in enumerate(urls):
            if filename_column_name is None:
                filename = os.path.join(args.root, str(file_num).zfill(digits) + extension)
            else:
                filename = os.path.join(args.root, df.iloc[i][filename_column_name])
            tasks.append(download_image(session, url, filename, i, pbar))
            file_num += 1

        # Here, we're limiting the number of simultaneous tasks. Adjust as needed.
        sem = asyncio.Semaphore(50)

        async def bound_download(task):
            async with sem:
                await task

        await asyncio.gather(*(bound_download(task) for task in tasks))

parser = argparse.ArgumentParser()
parser.add_argument("--df_path", type=str, default=None, required=True)
parser.add_argument("--root", type=str, default=None, required=True)
args = parser.parse_args()
asyncio.run(main(args))




