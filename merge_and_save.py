from unsloth import FastLanguageModel

if __name__ == '__main__':
    adaptor_path = f'./checkpoint'
    model, tokenizer= FastLanguageModel.from_pretrained(adaptor_path, load_in_4bit=True)
    model.save_pretrained_merged("./non-rag-sft", tokenizer, save_method = "merged_16bit")
