import argparse
import json
from pathlib import Path
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


CYRILLIC = set(
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--model", default="ekwek/Soprano-80M")
    return p.parse_args()


def read_metadata(dataset_path):
    csv = dataset_path / "metadata.csv"
    txt = dataset_path / "metadata.txt"
    
    path = csv if csv.exists() else txt if txt.exists() else None
    if not path:
        raise FileNotFoundError(f"No metadata in {dataset_path}")
    
    with open(path, encoding='utf-8') as f:
        lines = f.read().strip().split('\n')
    
    skip_header = lines[0].startswith('audio_path')
    texts = []
    
    for line in lines[skip_header:]:
        if '|' in line:
            texts.append(line.split('|', 1)[1])
    
    return texts


def get_missing_chars(texts, tokenizer):
    used_chars = set(''.join(texts))
    all_chars = used_chars | CYRILLIC
    
    missing = set()
    for char in all_chars:
        ids = tokenizer.encode(char, add_special_tokens=False)
        if not ids or 0 in ids:
            missing.add(char)
    
    return missing


def expand_vocab(tokenizer, model, new_chars, output_path):
    chars = sorted(new_chars)
    added = tokenizer.add_tokens(chars)
    
    old_size = model.config.vocab_size
    new_size = len(tokenizer)
    
    model.resize_token_embeddings(new_size)
    
    # Init new embeddings from existing mean
    embed_layer = model.get_input_embeddings()
    with torch.no_grad():
        base = embed_layer.weight[:old_size].mean(dim=0)
        for i in range(old_size, new_size):
            embed_layer.weight[i] = base + torch.randn_like(base) * 0.02
    
    model.config.vocab_size = new_size
    
    tokenizer.save_pretrained(output_path)
    model.save_pretrained(output_path)
    
    return added, new_size


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    
    texts = read_metadata(args.dataset)
    missing = get_missing_chars(texts, tokenizer)
    
    print(f"Dataset: {len(texts)} samples")
    print(f"Missing chars: {len(missing)}")
    print(f"Chars: {sorted(missing)}")
    
    if missing:
        added, vocab_size = expand_vocab(tokenizer, model, missing, args.output)
        print(f"Added {added} tokens → vocab size {vocab_size}")
        
        mapping = {c: tokenizer.convert_tokens_to_ids(c) for c in sorted(missing)}
        with open(args.output / "russian_tokens.json", 'w', encoding='utf-8') as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
    else:
        print("Nothing to add")
        tokenizer.save_pretrained(args.output)
        model.save_pretrained(args.output)


if __name__ == '__main__':
    main()