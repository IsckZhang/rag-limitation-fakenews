import os
import pandas as pd
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

with open('./instruction.template') as fp:
    template = fp.read()

def create_datasets(tokenizer, data_path, test_size=0.01, seed=42):
    assert tokenizer.chat_template is not None, "Chat tokenizer is not initialized."
    df = pd.read_csv(data_path)
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    def row_to_messages(example):
        batch = []
        for text, label in zip(*example.values()):
            _raw =  [
                    {"role": "user", "content": template.format(text=text)},
                    {"role": "assistant", "content": 'Real.' if label else 'Fake.'}
                ]
            batch.append(tokenizer.apply_chat_template(_raw, tokenize=False))
        return {"content": batch}
    
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(row_to_messages, batched=True, remove_columns=dataset.column_names)
    dataset = dataset.train_test_split(test_size=test_size, seed=seed)
   

    train_data = dataset["train"]
    valid_data = dataset["test"]

    print(f"Total train samples: {len(train_data)}")
    print(f"Total valid samples: {len(valid_data)}")

    return train_data, valid_data


def main():
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"  
    data_path = os.path.join("data", "train.csv")  

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=43008,
        dtype=None,           
        load_in_4bit=True,    
    )

    train_data, valid_data = create_datasets(tokenizer, data_path=data_path)

    collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        instruction_template='<|start_header_id|>user<|end_header_id|>',
        response_template='<|start_header_id|>assistant<|end_header_id|>'
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    training_args = TrainingArguments(
        output_dir="",
        per_device_eval_batch_size=8,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        lr_scheduler_type="linear",
        warmup_ratio=0.10,
        num_train_epochs=1,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        max_grad_norm=0.3,
        save_strategy="steps",
        save_total_limit=5,
        save_steps=0.1,
        eval_steps=0.1,
        eval_strategy="steps",
        seed=42,
        greater_is_better=False,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True
    )


    output_name = (
        f"llama-3-fakenews-{training_args.num_train_epochs}ep-"
        f"{training_args.per_device_train_batch_size}bsz-{training_args.gradient_accumulation_steps}accum"
    )
    training_args.output_dir = output_name

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=training_args,
        packing=False,  
        data_collator=collator,
        dataset_text_field="content",
        dataset_num_proc=8
    )
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model(output_dir=output_name)
    print(f"Model saved to {output_name}")


if __name__ == '__main__':
    main()
