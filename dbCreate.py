import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from vectordb import Memory

df = pd.read_csv("data/train.csv", header=0, names=["text", "label"])
df["text"] = df["text"].fillna("").astype(str)
texts = df["text"].tolist()
labels = df["label"].tolist()
metadatas = [{"label": lbl} for lbl in labels]

memory = Memory(
    memory_file="paraArt",
    #chunking_strategy={ "mode":"sliding_window", "window_size":10**6, "overlap":0 },
    chunking_strategy={ "mode": "paragraph" },
    embeddings="all-MiniLM-L6-v2"   
)
memory.save(texts, metadatas)

