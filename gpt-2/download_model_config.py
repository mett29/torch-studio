import os
import requests
from tqdm import tqdm

subdir = os.path.join('gpt-2', 'config')
if not os.path.exists(subdir):
    os.makedirs(subdir)
subdir = subdir.replace('\\', '/') # needed for Windows

for filename in ['encoder.json', 'vocab.bpe', 'hparams.json']:

    r = requests.get("https://openaipublic.blob.core.windows.net/gpt-2/models/124M/" + filename, stream=True)

    with open(os.path.join(subdir, filename), 'wb') as f:
        file_size = int(r.headers["content-length"])
        chunk_size = 1000
        with tqdm(ncols=100, desc="Fetching " + filename, total=file_size, unit_scale=True) as pbar:
            # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                pbar.update(chunk_size)
