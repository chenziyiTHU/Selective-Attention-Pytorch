import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import functional as F
from model.model import ModelArgs, SelectiveTransformer
from model.tokenizer import Tokenizer
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import numpy as np
import math
from dataclasses import dataclass


@dataclass
class TrainingArgs: 
    model_path: str
    tokenizer_path: str
    max_length: int = 512
    batch_size: int = 256
    total_steps: int = 524288
    learning_rate: float = 0.005
    warmup_steps: int = 1000
    beta_1: float = 0.9
    beta_2: float = 0.999
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_args: ModelArgs = ModelArgs()
    auxilary_loss: bool = True


class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        tokens = []
        for text in data:
            tokens.extend(tokenizer.encode(text, bos=True, eos=True))
        self.tokens = torch.tensor(tokens)
        
    def __len__(self):
        return len(self.tokens) - self.max_length

    def __getitem__(self, idx):
        input_id = self.tokens[idx:idx + self.max_length]
        target_id = self.tokens[idx + 1:idx + self.max_length + 1]
        return input_id, target_id
    

class C4Dataset(Dataset):
    def __init__(self, tokenizer, max_length, split="train"):
        self.tokenizer = tokenizer
        self.max_length = max_length

        print(f"Loading C4 dataset ({split} split)...")
        self.dataset = load_dataset("allenai/c4", "en", split=split, streaming=True)

        print("Tokenizing dataset...")
        self.tokens = []
        count = 0
        for item in self.dataset:
            text = item["text"]
            tokens = tokenizer.encode(text, bos=True, eos=True)
            self.tokens.extend(tokens)
        
        self.tokens = torch.tensor(self.tokens)
        print(f"Dataset loaded. Total tokens: {len(self.tokens)}")
        
    def __len__(self):
        return len(self.tokens) - self.max_length

    def __getitem__(self, idx):
        input_id = self.tokens[idx:idx + self.max_length]
        target_id = self.tokens[idx + 1:idx + self.max_length + 1]
        return input_id, target_id
    

def memory_loss(N, num_layers, F_masks, tau=1):
    """
    Auxiliary loss to encourage masking out more elements.

    Args:
        N (int): Number of non-pad tokens in the input sequence.
        num_layers (int): Number of layers in the model.
        F_masks (List[torch.Tensor]): List of masking score of each layer.
        tau (float): Clamp value for the masking score.
    """
    batch_size = F_masks[0].shape[0]
    loss = 0
    for F_mask in F_masks:
        F_mask_clamp = torch.clamp(F_mask, max=tau)  # (batch, N, N)
        total_masked = torch.sum(F_mask_clamp, dim=2) / tau  # (batch, N)
        mem_requried = torch.arange(N).expand(batch_size, -1) - total_masked
        loss += mem_requried.max(dim=1)[0].mean()
    return loss / (num_layers * N)


def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)


def train(training_args: TrainingArgs):
    torch.manual_seed(42)
    device = torch.device(training_args.device)

    model_args = training_args.model_args
    
    tokenizer = Tokenizer(model_path=training_args.tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    
    print("Initializing model...")
    model = SelectiveTransformer(model_args)
    model.to(device)
    
    print("Preparing datasets...")
    train_dataset = C4Dataset(tokenizer=tokenizer,
                              max_length=training_args.max_length,
                              split="train")
    train_loader = DataLoader(train_dataset, 
                              batch_size=training_args.batch_size, 
                              shuffle=True)
    
    optimizer = AdamW(model.parameters(), 
                      betas=(training_args.beta_1, training_args.beta_2), 
                      lr=training_args.learning_rate)
    
    lr_scheduler = get_lr_scheduler(optimizer,
                                    warmup_steps=training_args.warmup_steps,
                                    total_steps=training_args.total_steps)
    
    print("Start training...")
    global_step = 0
    model.train()
    total_loss = 0

    while global_step < training_args.total_steps:
        for batch_idx, (input_ids, target_ids) in enumerate(train_loader):
            if global_step >= training_args.total_steps:
                break

            input_ids, target_ids = input_ids.to(device), target_ids.to(device)
            outputs, F_masks = model(input_ids)
            
            loss = F.cross_entropy(
                outputs.view(-1, model_args.vocab_size),
                target_ids.view(-1),
                ignore_index=tokenizer.pad_id
            )

            if training_args.auxilary_loss:
                N = input_ids.shape[1]
                loss += memory_loss(N, model_args.n_layers, F_masks)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            lr_scheduler.step()
            current_lr = lr_scheduler.get_last_lr()[0]
            
            total_loss += loss.item()
            global_step += 1
            
            if global_step % 100 == 0:
                avg_loss = total_loss / 100
                print(f"Step: {global_step}/{training_args.total_steps}, "
                      f"Loss: {avg_loss:.4f}, "
                      f"LR: {current_lr:.6f}")
                total_loss = 0
            
            if global_step % 10000 == 0:
                torch.save({
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict(),
                    'loss': avg_loss,
                }, f'checkpoint_step_{global_step}.pt')
        
    torch.save({
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, 'model_final.pt')


if __name__ == "__main__":
    training_args = TrainingArgs(
        model_path="model.pt",
        tokenizer_path="tokenizer.model"
    )
    train(training_args)