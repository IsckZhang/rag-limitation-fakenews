import os
import pandas as pd
import json
from torch.cuda import device_count
from vectordb import Memory
from tqdm import tqdm
import pickle

COT = False
USE_RAG = True
MAX_SEQ_LEN = 131000

def main():
    test_data_path = "data/test.csv"  
    # Load the test dataset
    df = pd.read_csv(test_data_path)
    df = df.dropna(subset=['text'])        #  remove any NaN texts
    df['text'] = df['text'].astype(str)
    CONTEXT_LEVEL = 'paraArt'
    memory = Memory(memory_file=CONTEXT_LEVEL)
    #df = pd.read_csv(test_data_path)
    contexts = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Pre-Processing data"):
        text = row['text']
        context = memory.search(text, top_n=5, unique=True)
        contexts.append(context)

    # Save the contexts to a pickle file
    with open(f'contexts-{CONTEXT_LEVEL}.pkl', 'wb') as f:
        pickle.dump(contexts, f)

    


if __name__ == "__main__":
    main()