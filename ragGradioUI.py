from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import gradio as gr
import torch
from vectordb import Memory  # Added for RAG integration

# checkpoint = "meta-llama/Llama-3.1-8B-Instruct"
checkpoint = "non-rag-sft"
device = "cuda"  # "cuda" or "cpu"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(checkpoint, quantization_config=nf4_config)

# Load vectorDB (paraArt embedding)
memory = Memory(memory_file='paraArt')  # Added line

def predict(message, history):
    # Retrieve context from vectorDB for RAG
    rag_context = memory.search(message, top_n=3, unique=True)  # RAG retrieval step added
    context_str = "\n".join(rag_context)

    _history = history.copy()
    rag_augmented_message = f"Context: {context_str}\n\nQuestion: {message}"  # Augmented prompt
    _history.append({"role": "user", "content": rag_augmented_message})

    input_text = tokenizer.apply_chat_template(_history, tokenize=False)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_new_tokens=4096, temperature=0.2, top_p=0.9, do_sample=True)
    decoded = tokenizer.decode(outputs[0])
    response = decoded.split("<|start_header_id|>assistant<|end_header_id|>\n")[-1].split("<|eot_id|>")[0]
    return response

demo = gr.ChatInterface(predict, type="messages")
demo.launch()
