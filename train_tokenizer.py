from sentencepiece import SentencePieceTrainer
from datasets import load_dataset
import os

def train_tokenizer(
    vocab_size: int = 8000,
    model_prefix: str = "tokenizer",
    input_file: str = "train.txt",
    model_type: str = "bpe",
    character_coverage: float = 0.9995,
    num_examples: int = 10000
):
    print("Loading C4 dataset...")
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    
    print("Saving texts to file...")
    with open(input_file, "w", encoding="utf-8") as f:
        count = 0
        for item in dataset:
            f.write(item["text"] + "\n")
            count += 1
            if count >= num_examples:
                break
            if count % 10000 == 0:
                print(f"Processed {count} examples")
    
    print("Training tokenizer...")
    SentencePieceTrainer.Train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        pad_id=0,
        eos_id=1,
        bos_id=2,
        unk_id=3,
        pad_piece="<pad>",
        eos_piece="</s>",
        bos_piece="<s>",
        unk_piece="<unk>",
        user_defined_symbols=["<mask>"],
    )
    
    os.remove(input_file)
    print("Tokenizer training completed!")

if __name__ == "__main__":
    train_tokenizer(
        vocab_size=8000,
        model_prefix="SelectiveAttentionTokenizer",
        input_file="train.txt",
        model_type="bpe",
        character_coverage=0.9995,
        num_examples=10000
    )