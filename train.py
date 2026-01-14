import argparse
import pathlib
import random
import time
import json
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset import AudioDataset


def get_args():
    parser = argparse.ArgumentParser(description="Train Soprano TTS with Russian support")
    parser.add_argument("--input-dir", required=True, type=pathlib.Path,
        help="Path to dataset directory containing train.json and val.json")
    parser.add_argument("--save-dir", required=True, type=pathlib.Path,
        help="Path to save checkpoints and final model")
    parser.add_argument("--model-path", default="ekwek/Soprano-80M", type=str,
        help="Path to base model or expanded Russian model")
    parser.add_argument("--epochs", default=10, type=int,
        help="Number of training epochs")
    parser.add_argument("--batch-size", default=4, type=int,
        help="Batch size for training")
    parser.add_argument("--learning-rate", default=5e-4, type=float,
        help="Maximum learning rate")
    parser.add_argument("--device", default="cuda:0", type=str,
        help="Device to train on (cuda:0, cpu, etc.)")
    parser.add_argument("--eval-steps", default=500, type=int,
        help="Evaluate every N steps")
    parser.add_argument("--save-steps", default=500, type=int,
        help="Save checkpoint every N steps")
    parser.add_argument("--top-k", default=3, type=int,
        help="Keep top K best checkpoints")
    return parser.parse_args()


class CheckpointManager:
    """Manages saving and keeping top K best checkpoints."""
    
    def __init__(self, save_dir, top_k=3):
        self.save_dir = pathlib.Path(save_dir)
        self.top_k = top_k
        self.checkpoints = []
        self.checkpoint_file = self.save_dir / "checkpoints.json"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._load_checkpoint_info()
    
    def _load_checkpoint_info(self):
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)
                self.checkpoints = [(c['val_loss'], c['step'], c['path']) for c in data]
    
    def _save_checkpoint_info(self):
        data = [{'val_loss': c[0], 'step': c[1], 'path': c[2]} for c in self.checkpoints]
        with open(self.checkpoint_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def save(self, model, tokenizer, optimizer, step, epoch, val_loss, train_loss):
        checkpoint_name = f"checkpoint-step{step}-epoch{epoch}-loss{val_loss:.4f}"
        checkpoint_path = self.save_dir / checkpoint_name
        
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        
        training_state = {
            'step': step, 'epoch': epoch,
            'val_loss': val_loss, 'train_loss': train_loss,
            'optimizer_state_dict': optimizer.state_dict()
        }
        torch.save(training_state, checkpoint_path / "training_state.pt")
        
        self.checkpoints.append((val_loss, step, str(checkpoint_path)))
        self.checkpoints.sort(key=lambda x: x[0])
        
        while len(self.checkpoints) > self.top_k:
            worst = self.checkpoints.pop()
            worst_path = pathlib.Path(worst[2])
            if worst_path.exists():
                shutil.rmtree(worst_path)
                print(f"  Removed: {worst_path.name}")
        
        self._save_checkpoint_info()
        self._print_status()
        return checkpoint_path
    
    def _print_status(self):
        print(f"\n  Top {self.top_k} Checkpoints:")
        for i, (loss, s, _) in enumerate(self.checkpoints):
            marker = "►" if i == 0 else " "
            print(f"    {marker} {i+1}. Step {s}: val_loss={loss:.4f}")
    
    def get_best(self):
        return pathlib.Path(self.checkpoints[0][2]) if self.checkpoints else None


class CollateFunction:
    def __init__(self, tokenizer, batch_size, seq_len):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_len = seq_len
    
    def __call__(self, texts):
        tokens_batch = self.tokenizer(texts, padding=False, truncation=False)
        batch, cur_sample, cur_size = [], [], 0
        
        for i in range(len(texts)):
            tokens = torch.tensor(tokens_batch['input_ids'][i][:-1], dtype=torch.long)
            cur_size += tokens.size(0)
            cur_sample.append(tokens)
            if cur_size >= self.seq_len + 1:
                batch.append(torch.cat(cur_sample)[:self.seq_len + 1])
                cur_sample, cur_size = [], 0
                if len(batch) == self.batch_size:
                    break
        
        if cur_sample and not batch:
            batch.append(torch.cat(cur_sample + [torch.zeros(self.seq_len, dtype=torch.long)])[:self.seq_len + 1])
        
        while len(batch) < self.batch_size:
            batch.append(batch[-1])
        
        batch = torch.stack(batch)
        return batch[:, :-1], batch[:, 1:]


def get_lr(step, total_steps, max_lr, min_lr, warmup=0.1, cooldown=0.1):
    warmup_steps = int(total_steps * warmup)
    cooldown_steps = int(total_steps * cooldown)
    
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step < total_steps - cooldown_steps:
        return max_lr
    return min_lr + (max_lr - min_lr) * ((total_steps - step) / cooldown_steps)


def compute_loss(logits, y, tokenizer, device):
    pred = logits.view(-1, logits.size(-1))
    labels = y.reshape(-1)
    loss = torch.nn.functional.cross_entropy(pred, labels, reduction='none')
    
    audio_start = tokenizer.convert_tokens_to_ids('[0]')
    audio_end = tokenizer.convert_tokens_to_ids('[7999]')
    audio_mask = torch.logical_and(y >= audio_start, y <= audio_end).view(-1)
    
    russian_token_ids = set()
    for char in "абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ":
        token_id = tokenizer.convert_tokens_to_ids(char)
        if token_id != tokenizer.unk_token_id:
            russian_token_ids.add(token_id)
    
    russian_mask = torch.tensor([l.item() in russian_token_ids for l in labels], device=device)
    
    audio_loss = loss[audio_mask].mean() if audio_mask.any() else torch.tensor(0.0).to(device)
    text_loss = loss[~audio_mask].mean() if (~audio_mask).any() else torch.tensor(0.0).to(device)
    russian_loss = loss[russian_mask].mean() if russian_mask.any() else torch.tensor(0.0).to(device)
    
    acc = (logits.argmax(dim=-1) == y).view(-1)[audio_mask].float().mean() if audio_mask.any() else torch.tensor(0.0).to(device)
    
    return audio_loss, text_loss, russian_loss, acc


@torch.no_grad()
def evaluate(model, dataloader, tokenizer, device, device_type):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits = model(x).logits
            audio_loss, text_loss, russian_loss, acc = compute_loss(logits, y, tokenizer, device)
        total_loss += audio_loss.item()
        total_acc += acc.item()
        n += 1
    
    model.train()
    return (total_loss / n, total_acc / n) if n > 0 else (0.0, 0.0)


def main():
    args = get_args()
    
    device = args.device
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    seq_len = 1024
    
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)
    torch.set_float32_matmul_precision('high')
    
    # Load model
    print(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    model.to(torch.bfloat16).to(device).train()
    print(f"Vocab size: {len(tokenizer)}")
    
    # Data
    collate_fn = CollateFunction(tokenizer, args.batch_size, seq_len)
    
    train_data = AudioDataset(f'{args.input_dir}/train.json')
    train_loader = DataLoader(train_data, batch_size=args.batch_size*16, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True, collate_fn=collate_fn)
    
    val_data = AudioDataset(f'{args.input_dir}/val.json')
    val_loader = DataLoader(val_data, batch_size=args.batch_size*16, shuffle=False,
        num_workers=1, pin_memory=True, persistent_workers=True, collate_fn=collate_fn)
    
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    
    print(f"\nConfig: {args.epochs} epochs, {steps_per_epoch} steps/epoch, {total_steps} total steps")
    print(f"Train: {len(train_data)} samples | Val: {len(val_data)} samples\n")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), args.learning_rate,
        betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    
    checkpoint_mgr = CheckpointManager(args.save_dir, args.top_k)
    
    # Training
    global_step = 0
    min_lr = 0.1 * args.learning_rate
    
    for epoch in range(args.epochs):
        print(f"\n{'='*50}\nEpoch {epoch+1}/{args.epochs}\n{'='*50}")
        
        epoch_loss, epoch_acc, epoch_steps = 0.0, 0.0, 0
        epoch_text_loss, epoch_russian_loss = 0.0, 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits = model(x).logits
                audio_loss, text_loss, russian_loss, acc  = compute_loss(logits, y, tokenizer, device)
            
            audio_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            lr = get_lr(global_step, total_steps, args.learning_rate, min_lr)
            for pg in optimizer.param_groups:
                pg['lr'] = lr
            optimizer.step()
            
            epoch_loss += audio_loss.item()
            epoch_text_loss += text_loss.item()
            epoch_russian_loss += russian_loss.item()
            epoch_acc += acc.item()
            epoch_steps += 1
            global_step += 1
            
            pbar.set_postfix(loss=f'{audio_loss.item():.3f}', text=f'{text_loss.item():.3f}', rus=f'{russian_loss.item():.3f}', acc=f'{acc.item():.4f}', lr=f'{lr:.1e}')
            
            # Eval & Save
            if global_step % args.eval_steps == 0:
                val_loss, val_acc = evaluate(model, val_loader, tokenizer, device, device_type)
                print(f"\n  [Step {global_step}] Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            if global_step % args.save_steps == 0:
                val_loss, _ = evaluate(model, val_loader, tokenizer, device, device_type)
                print(f"\n  Saving checkpoint...")
                checkpoint_mgr.save(model, tokenizer, optimizer, global_step, epoch+1,
                    val_loss, epoch_loss/epoch_steps)
        
        # End of epoch
        avg_loss = epoch_loss / epoch_steps
        val_loss, val_acc = evaluate(model, val_loader, tokenizer, device, device_type)
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_loss:.4f} | Text: {epoch_text_loss/epoch_steps:.4f} | Russian: {epoch_russian_loss/epoch_steps:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        checkpoint_mgr.save(model, tokenizer, optimizer, global_step, epoch+1, val_loss, avg_loss)
    
    # Final
    print(f"\n{'='*50}\nTraining Complete!\n{'='*50}")
    
    final_path = args.save_dir / "final"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Final model: {final_path}")
    
    best = checkpoint_mgr.get_best()
    if best:
        print(f"Best checkpoint: {best} (loss: {checkpoint_mgr.checkpoints[0][0]:.4f})")


if __name__ == '__main__':
    main()
