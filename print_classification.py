from sklearn.metrics import classification_report
import json
import re

if __name__ == "__main__":
    #file_in = 'output/non-rag-sft-result.jsonl'
    #file_in = 'output/meta-llama-3.1-8b-instruct-bnb-4bit-result.jsonl'
    #file_in = 'output/non-rag-sft-rag-paraart-result.jsonl'
    file_in = 'output/llama-3.1-8b-instruct-rag-paraart-result.jsonl'
    y_pred = []
    y_true = []
    # Load the classification report from a JSON file
    with open(file_in, 'r') as f:
        for line in f:
            report = json.loads(line)
            y_pred.append(report["response"].lower().replace('.', ''))
            y_true.append(report["ground_truth"].lower())

    # Print the classification report
    print(classification_report(y_pred=y_pred, y_true=y_true, labels=['real', 'fake']))