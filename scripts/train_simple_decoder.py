import argparse
import time
from pathlib import Path

# Original imports that were implicitly removed by the instruction's example, but are necessary and should be kept.
import torch
import torch.nn as nn
import torch.optim as optim

# Import llm package to ensure core components are registered
from llm.data.loader import TextDataset, create_dataloader
from llm.models.decoder import DecoderModel
from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer


def train(args):
    """
    Main training function.
    """
    # 1. Setup Device
    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
            print("Device not specified. Auto-detected CUDA available. Using CUDA.")
        else:
            args.device = "cpu"
            print("Device not specified. CUDA not available. Using CPU.")
    elif args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # 2. Initialize Tokenizer
    print("Initializing tokenizer...")
    try:
        with Path(args.file_path).open(encoding="utf-8") as f:
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
        padding_value=tokenizer.pad_token_id,  # Explicitly use tokenizer's pad_token_id
    )
    if len(dataset) == 0:
        print("Error: Dataset is empty. Check file_path and its content.")
        return

    dataloader = create_dataloader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,  # Shuffle for training
    )
    print(f"Training Dataset and DataLoader created. Number of sequences: {len(dataset)}")

    # Validation DataLoader (if val_file_path is provided)
    val_dataloader = None
    if args.val_file_path:
        print(f"Validation file provided: {args.val_file_path}")
        try:
            val_dataset = TextDataset(
                file_path=args.val_file_path,
                tokenizer=tokenizer,  # Use the same tokenizer
                max_seq_len=args.max_seq_len,
                overlap=args.overlap,  # Using same overlap as training, can be made configurable
                padding_value=tokenizer.pad_token_id,
            )
            if len(val_dataset) > 0:
                val_dataloader = create_dataloader(
                    dataset=val_dataset,
                    batch_size=args.batch_size,  # Can use a different batch size for validation
                    shuffle=False,  # No need to shuffle validation data
                )
                print(f"Validation Dataset and DataLoader created. Number of sequences: {len(val_dataset)}")
            else:
                print("Warning: Validation dataset is empty. Skipping validation.")
        except FileNotFoundError:
            print(f"Error: Validation file not found at {args.val_file_path}. Skipping validation.")
        except Exception as e:
            print(f"Error creating validation dataset/loader: {e}. Skipping validation.")

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
        max_seq_len=args.max_seq_len,  # For positional encoding in EmbeddingLayer
        pos_encoding_learned=False,  # Default from DecoderModel, can be arg if needed
        embedding_dropout_p=embedding_dropout_p,
        attn_dropout_p=attn_dropout_p,
        mlp_dropout_p=mlp_dropout_p,
        norm_first=True,  # Default from DecoderModel (Pre-LN)
        is_causal=True,  # Default for Decoder, ensures causal masking in MHA
        padding_idx=tokenizer.pad_token_id,  # Crucial for EmbeddingLayer to ignore PAD
        device=device,  # Pass device for model parameter initialization
        dtype=torch.float32,  # Default dtype
    )
    model.to(device)
    print(f"Model initialized and moved to {device}.")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable model parameters: {total_params:,}")

    # 5. Initialize Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 6. Initialize Learning Rate Scheduler
    # T_max is often set to the total number of training steps (len(dataloader) * args.epochs)
    # or just args.epochs if step is called per epoch.
    # For CosineAnnealingLR, T_max is the number of iterations until the first restart.
    # If scheduler.step() is called per epoch, T_max=args.epochs is common.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    print(f"Initialized CosineAnnealingLR scheduler with T_max={args.epochs}.")

    # 7. Loss Function
    # CrossEntropyLoss will ignore targets with value `tokenizer.pad_token_id` if `ignore_index` is set.
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # 8. Training Loop
    print("\nStarting training...")
    model.train()  # Set model to training mode

    # Early stopping variables (only relevant if val_dataloader is used)
    best_val_loss = float("inf")
    epochs_no_improve = 0
    early_stop = False

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        total_epoch_loss = 0
        batch_count = 0

        for i, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)  # Labels are also token IDs

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients
            optimizer.step()

            total_epoch_loss += loss.item()
            batch_count += 1

            if (i + 1) % args.log_interval == 0:
                print(f"Epoch [{epoch + 1}/{args.epochs}], Batch [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        avg_epoch_loss = total_epoch_loss / batch_count if batch_count > 0 else 0
        epoch_duration = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"--- Epoch {epoch + 1} completed in {epoch_duration:.2f}s ---")
        print(f"Average Training Loss for Epoch {epoch + 1}: {avg_epoch_loss:.4f}")
        print(f"Current Learning Rate at end of Epoch {epoch + 1}: {current_lr:.6f}")

        # Validation loop
        if val_dataloader:
            model.eval()  # Set model to evaluation mode
            total_val_loss = 0
            val_batch_count = 0
            val_epoch_start_time = time.time()

            with torch.no_grad():  # Disable gradient calculations
                for val_batch in val_dataloader:
                    input_ids_val = val_batch["input_ids"].to(device)
                    labels_val = val_batch["labels"].to(device)

                    logits_val = model(input_ids_val, attn_mask=None)
                    loss_val = criterion(logits_val.view(-1, tokenizer.vocab_size), labels_val.view(-1))

                    total_val_loss += loss_val.item()
                    val_batch_count += 1

            avg_val_loss = total_val_loss / val_batch_count if val_batch_count > 0 else 0
            val_epoch_duration = time.time() - val_epoch_start_time
            print(
                f"Average Validation Loss for Epoch {epoch + 1}: {avg_val_loss:.4f} (calculated in {val_epoch_duration:.2f}s)"
            )

            # Early stopping check
            if avg_val_loss < best_val_loss - args.early_stopping_min_delta:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                # Optional: Save model checkpoint here
                print(f"Validation loss improved to {best_val_loss:.4f}.")
            else:
                epochs_no_improve += 1
                print(f"No significant improvement in validation loss for {epochs_no_improve} epoch(s).")

            if epochs_no_improve >= args.early_stopping_patience:
                early_stop = True
                print(f"Early stopping triggered after {args.early_stopping_patience} epochs without improvement.")

            model.train()  # Set model back to training mode

        # Step the scheduler after the epoch (and after validation)
        scheduler.step()

        if early_stop:
            print("Breaking training loop due to early stopping.")
            break

        print("---------------------------------------------------")

    print("\nTraining finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a simple Decoder-only Transformer model.")

    # File and Data arguments
    parser.add_argument("--file_path", type=str, required=True, help="Path to the training text file.")
    parser.add_argument("--val_file_path", type=str, default=None, help="Optional path to the validation text file.")
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
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Device to train on ('cpu' or 'cuda'). Defaults to auto-detect.",
    )
    parser.add_argument("--log_interval", type=int, default=10, help="Print loss every N batches.")

    # Early stopping arguments
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3,
        help="Number of epochs to wait for improvement before stopping early. Only active if val_file_path is provided.",
    )
    parser.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=0.001,
        help="Minimum change in validation loss to be considered an improvement for early stopping.",
    )

    args = parser.parse_args()

    print("Training arguments:")
    for arg, value in sorted(vars(args).items()):
        print(f"  {arg}: {value}")
    print("-" * 30)

    train(args)
