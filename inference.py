import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
import pandas as pd
import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from torch.cuda import device_count
from vectordb import Memory
from tqdm import tqdm


def make_chat(tokenizer, prompt)-> str:
    _message = [
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(_message, tokenize=False, add_generation_prompt=True)


def memory_to_str(memory, token_lim:int, tokenizer) -> str:
    """
    Convert memory to string with a token limit.
    """
    memory_str = ""
    for item in memory:
        text = item['chunk']
        label = "real" if item['metadata']['label'] else "fake"
        if len(tokenizer.tokenize(text)) > token_lim:
            continue
        memory_str += f"The following piece is a {label} artical:\n{text}\n"
        if len(tokenizer.tokenize(memory_str)) > token_lim:
            break
    return memory_str.strip()


COT = False
USE_RAG = True
CONTEXT_LEVEL = 'paraArt'
MAX_SEQ_LEN = 131000

def main():
    test_data_path = "data/test.csv" 
    
    model_name = "./non-rag-sft"

    memory = None

    # Load the template
    with open(f'./instruction{"_cot" if COT else ""}.template') as fp:
        template = fp.read()

    # Override template and load memory if using RAG
    if USE_RAG:
        memory = Memory(memory_file=CONTEXT_LEVEL)
        with open(f'./instruction{"_cot" if COT else ""}_rag.template') as fp:
            template = fp.read()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load the test dataset
    queue = []
    ground_truth = []
    df = pd.read_csv(test_data_path)
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Pre-Processing data"):
        text = row['text']
        if USE_RAG:
            context = memory.search(text, top_n=5, unique=True)
            token_lim = MAX_SEQ_LEN - len(tokenizer.tokenize(text) + tokenizer.tokenize(template)) - 50
            context = memory_to_str(context, token_lim, tokenizer)
            if context == "":
                context = "No relevant context available."
        label = 'Real' if row['label'] else 'Fake'
        if USE_RAG:
            prompt = template.format(text=text, context=context)
        else:
            prompt = template.format(text=text)
        chat = make_chat(tokenizer, prompt)
        queue.append(chat)
        ground_truth.append(label)

    # Initialize the LLM
    llm = LLM(model=model_name,
              max_model_len=MAX_SEQ_LEN,
              quantization='bitsandbytes',
              load_format='bitsandbytes',
              tensor_parallel_size=1)
    max_tokens = 4096 if COT else 10
    sampling_params = SamplingParams(temperature=0.1, max_tokens=max_tokens, top_p=0.95)
    responses = llm.generate(queue, sampling_params, use_tqdm=True)

    out_list = []
    correct_count = 0
    for truth, response in zip(ground_truth, responses):
        resp_str = response.outputs[0].text.strip()
        out_list.append({'response': resp_str,  'ground_truth': truth})
        if not COT and resp_str.lower().startswith(truth.lower()):
            correct_count += 1
        if COT and resp_str.lower().endswith(truth.lower()):
            correct_count += 1

    with open(f"{os.path.basename(model_name).lower()}{'-cot' if COT else ''}{'-rag' if USE_RAG else ''}-result.jsonl", "w") as f:
        for item in out_list:
            f.write(json.dumps(item) + "\n")

    print(f"Results saved to {os.path.basename(model_name).lower()}{'-cot' if COT else ''}{'-rag' if USE_RAG else ''}-result.jsonl")
    print(f"Accuracy: {correct_count / len(responses):.2%}")


if __name__ == "__main__":
    main()