try:
    input_dir = snakemake.input.input_dir
    output_file = snakemake.output[0]
    directory = snakemake.wildcards.directory
    batchsize = int(snakemake.params.batchsize)
except NameError:
    input_dir = "/cache/nikolal/parlaspeech-rs/audio/oN2c-uF4srI/"
    output_file = "brisi.jsonl"
    directory = "oN2c-uF4srI"
    batchsize = 1
from transformers import AutoFeatureExtractor, Wav2Vec2BertForAudioFrameClassification
from datasets import load_dataset, Dataset, Audio
import torch
import numpy as np
import soundfile as sf
import tqdm
import os
import numpy as np
import os
import pandas as pd
import datasets
from datasets import load_dataset, load_metric, Audio
from itertools import zip_longest, pairwise
from pathlib import Path
import gc


mp3s = Path(input_dir).glob("*.mp3")

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda")

from utils import frames_to_intervals


checkpoint = Path(
    "/cache/peterr/mezzanine_resources/filled_pauses/model_filledPause_3e-5_20_4/checkpoint-900"
)
feature_extractor = AutoFeatureExtractor.from_pretrained(str(checkpoint))
model = Wav2Vec2BertForAudioFrameClassification.from_pretrained(str(checkpoint)).to(
    device
)


def evaluator(chunks):
    if isinstance(chunks, datasets.formatting.formatting.LazyRow):
        try:
            # We are dealing with a single chunk:
            sampling_rate = chunks["audio"]["sampling_rate"]
            with torch.no_grad():
                inputs = feature_extractor(
                    chunks["audio"]["array"],
                    return_tensors="pt",
                    sampling_rate=sampling_rate,
                ).to(device)
                logits = model(**inputs).logits
                torch.cuda.empty_cache()
            y_pred = np.array(logits.cpu()).argmax(axis=-1)
            torch.cuda.empty_cache()
            gc.collect()
            return {"y_pred": frames_to_intervals(y_pred[0].tolist())}
        except KeyboardInterrupt as e:
            raise e
        except:
            return {"y_pred": None}
    else:
        sampling_rate = chunks["audio"][0]["sampling_rate"]
        with torch.no_grad():
            inputs = feature_extractor(
                [i["array"] for i in chunks["audio"]],
                return_tensors="pt",
                sampling_rate=sampling_rate,
            ).to(device)
            logits = model(**inputs).logits
            torch.cuda.empty_cache()
        y_pred = np.array(logits.cpu()).argmax(axis=-1)
        torch.cuda.empty_cache()
        gc.collect()
        return {"y_pred": [frames_to_intervals(i) for i in y_pred]}


def get_start_end(s: str):
    s = str(Path(s).name)[12:].replace(".mp3", "")
    s, e = s.split("-")
    return float(s), float(e)


df = pd.DataFrame(data={"audio": [i for i in mp3s]})
df["name"] = df.audio.apply(lambda p: p.name)
df["path"] = df.audio.apply(lambda p: str(p))
df["start"] = [get_start_end(i)[0] for i in df.name.values]
df["end"] = [get_start_end(i)[1] for i in df.name.values]
df["duration"] = df.end - df.start
df = df[(df.duration <= 30) & (df.duration > 0)]
df = df.drop_duplicates(subset="name").reset_index()
df["audio"] = df.audio.apply(str)
df = df.drop(columns="name")
ds = datasets.Dataset.from_pandas(df)
ds = ds.cast_column("audio", Audio(sampling_rate=16_000, mono=True))

for i in range(7):
    try:
        ds = ds.map(
            evaluator,
            batch_size=batchsize,
            batched=batchsize != 1,
            desc="Running inference",
            remove_columns=["audio"],
        )
        break
    except torch.OutOfMemoryError as e:

        newbatchsize = max(1, batchsize // 2)
        if newbatchsize == batchsize:
            raise e
        else:
            batchsize = newbatchsize
        print(f"Reducing batchsize to {batchsize} and retrying.")
        with torch.no_grad():
            torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        gc.collect()
        continue
df["y_pred"] = [i for i in ds["y_pred"]]
df.drop(columns="audio start end duration".split()).to_json(
    output_file, orient="records", lines=True, index=False, force_ascii=False
)
