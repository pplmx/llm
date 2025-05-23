import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import time

# Adjust path to import from src (if script is run from project root or scripts/ folder)
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from src.llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer
    from src.llm.data.loader import TextDataset, create_dataloader
    from src.llm.models.decoder import DecoderModel
except ModuleNotFoundError as e:
    print("Error: Could not import necessary modules. Ensure PYTHONPATH is set correctly or run from project root.")
    print(f"Details: {e}")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

def train(args):
    """
    Main training function.
    """
    # 1. Setup Device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # 2. Initialize Tokenizer
    print("Initializing tokenizer...")
    try:
        with open(args.file_path, "r", encoding="utf-8") as f:
            # Reading lines to build corpus, could also read whole file
            # For SimpleCharacterTokenizer, a list of unique characters is fine.
            # Reading the whole file to extract all characters for the vocab.
            corpus_text = f.read() 
    except FileNotFoundError:
        print(f"Error: Training file not found at {args.file_path}")
        return
    
    # SimpleCharacterTokenizer expects a list of strings.
    # To build vocab from all chars in file, pass the content as a single-element list.
    tokenizer = SimpleCharacterTokenizer(corpus=[corpus_text])
    print(f"Tokenizer initialized. Vocabulary size: {tokenizer.vocab_size}")
    print(f"PAD token ID: {tokenizer.pad_token_id}")

    # 3. Initialize Dataset and DataLoader
    print("Loading data...")
    dataset = TextDataset(
        file_path=args.file_path,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        overlap=args.overlap,
        padding_value=tokenizer.pad_token_id # Explicitly use tokenizer's pad_token_id
    )
    if len(dataset) == 0:
        print("Error: Dataset is empty. Check file_path and its content.")
        return
        
    dataloader = create_dataloader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True # Shuffle for training
    )
    print(f"Dataset and DataLoader created. Number of sequences in dataset: {len(dataset)}")

    # 4. Initialize Model
    print("Initializing model...")
    # Hardcoded dropout values for simplicity as per instructions
    embedding_dropout_p = 0.1
    attn_dropout_p = 0.1
    mlp_dropout_p = 0.1
    
    model = DecoderModel(
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_len=args.max_seq_len, # For positional encoding in EmbeddingLayer
        pos_encoding_learned=False, # Default from DecoderModel, can be arg if needed
        embedding_dropout_p=embedding_dropout_p,
        attn_dropout_p=attn_dropout_p,
        mlp_dropout_p=mlp_dropout_p,
        norm_first=True, # Default from DecoderModel (Pre-LN)
        is_causal=True,  # Default for Decoder, ensures causal masking in MHA
        padding_idx=tokenizer.pad_token_id, # Crucial for EmbeddingLayer to ignore PAD
        device=device, # Pass device for model parameter initialization
        dtype=torch.float32 # Default dtype
    )
    model.to(device)
    print(f"Model initialized and moved to {device}.")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable model parameters: {total_params:,}")


    # 5. Initialize Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 6. Loss Function
    # CrossEntropyLoss will ignore targets with value `tokenizer.pad_token_id` if `ignore_index` is set.
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # 7. Training Loop
    print("\nStarting training...")
    model.train() # Set model to training mode

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        total_epoch_loss = 0
        batch_count = 0

        for i, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device) # Labels are also token IDs

            optimizer.zero_grad()
            
            # Forward pass
            # attn_mask=None: Causal masking is handled by is_causal=True in MHA.
            # Padding is handled by padding_idx in EmbeddingLayer.
            logits = model(input_ids, attn_mask=None)
            
            # Calculate loss
            # Logits: [B, S, V], Labels: [B, S]
            # Criterion expects Logits: [B*S, V] or [N,C], Labels: [B*S] or [N]
            loss = criterion(logits.view(-1, tokenizer.vocab_size), labels.view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Clip gradients
            optimizer.step()

            total_epoch_loss += loss.item()
            batch_count += 1

            if (i + 1) % args.log_interval == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Batch [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
        
        avg_epoch_loss = total_epoch_loss / batch_count if batch_count > 0 else 0
        epoch_duration = time.time() - epoch_start_time
        print(f"--- Epoch {epoch+1} completed in {epoch_duration:.2f}s ---")
        print(f"Average Loss for Epoch {epoch+1}: {avg_epoch_loss:.4f}")
        print("---------------------------------------------------")

    print("\nTraining finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a simple Decoder-only Transformer model.")
    
    # File and Data arguments
    parser.add_argument("--file_path", type=str, required=True, help="Path to the training text file.")
    parser.add_argument("--max_seq_len", type=int, default=32, help="Maximum sequence length for dataset and model.")
    parser.add_argument("--overlap", type=int, default=0, help="Overlap for TextDataset sequences.")

    # Model architecture arguments
    parser.add_argument("--hidden_size", type=int, default=64, help="Model hidden size.")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of transformer layers.")
    parser.add_argument("--num_heads", type=int, default=2, help="Number of attention heads.")
    # vocab_size is derived from tokenizer, so not an arg here.

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to train on ('cpu' or 'cuda').")
    parser.add_argument("--log_interval", type=int, default=10, help="Print loss every N batches.")

    args = parser.parse_args()
    
    print("Training arguments:")
    for arg, value in sorted(vars(args).items()):
        print(f"  {arg}: {value}")
    print("-" * 30)

    train(args)
```
