from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import gradio as gr
import torch

#checkpoint = "meta-llama/Llama-3.1-8B-Instruct"
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

def predict(message, history):
    _history = history.copy()
    _history.append({"role": "user", "content": message})
    input_text = tokenizer.apply_chat_template(_history, tokenize=False)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_new_tokens=4096, temperature=0.2, top_p=0.9, do_sample=True)
    decoded = tokenizer.decode(outputs[0])
    response = decoded.split("<|start_header_id|>assistant<|end_header_id|>\n")[-1].split("<|eot_id|>")[0]
    return response

demo = gr.ChatInterface(predict, type="messages")

demo.launch()