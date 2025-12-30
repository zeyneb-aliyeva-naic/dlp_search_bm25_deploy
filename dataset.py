"""Dataset classes and data loading utilities."""

import json
import random
import string
from pathlib import Path
from typing import List, Dict, Any, Union, Callable

import torch
from torch.utils.data import Dataset, DataLoader

from src.tokenizer import Tokenizer

def load_json_data(path: Path, shuffle: bool=True) -> List[Dict[str, str]]:
    """Load and optionally shuffle JSON data."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if shuffle:
            random.shuffle(data)
    return data

def load_json_data_multiple(paths: List[Path], shuffle: bool=True) -> List[Dict[str, str]]:
    """Load and optionally shuffle multiple JSON data files."""
    all_data = []
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_data.extend(data)
                print(f"Loaded {len(data)} examples from {path.name}")
        except Exception as e:
            print(f"Error loading file {path}: {e}")
    
    if shuffle:
        random.shuffle(all_data)
    
    print(f"Combined dataset contains {len(all_data)} examples")
    return all_data

class SpellingDataset(Dataset):
    def __init__(
        self, 
        paths: Union[Path, List[Path]], 
        tokenizer: Tokenizer,
        build_vocab: bool=False,
        vocab_save_path: str='vocab.json',
        truncate: Union[float, None] = None
    ):
        """
        Dataset for spelling correction task.
        
        Args:
            paths: Path or list of paths to the JSON data files
            tokenizer: Tokenizer instance
            build_vocab: Whether to build vocabulary from this dataset
            vocab_save_path: Path to save vocabulary
            truncate: If not None, use only this fraction of the data
        """
        # Handle both single path and list of paths
        if isinstance(paths, list):
            self.data = load_json_data_multiple(paths, shuffle=True)
        else:
            self.data = load_json_data(paths, shuffle=True)
            
        self.tokenizer = tokenizer
        
        if build_vocab:
            # Extract clean and noisy texts for vocabulary building
            all_texts = []
            for item in self.data:
                all_texts.append(item["clean"].lower())
                all_texts.append(item["noisy"].lower())
            
            # Build vocabulary with all texts
            self.tokenizer.build_vocab(all_texts)
            self.tokenizer.save_vocab(vocab_save_path)
            
        if truncate:
            self.data = self.data[:int(len(self.data) * truncate)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch, tokenizer):
    """Collate function for DataLoader."""
    clean_texts = [item["clean"].lower() for item in batch]
    noisy_texts = [item["noisy"].lower() for item in batch]

    # Tokenize and pad all texts
    clean_ids = [tokenizer.pad_sequence(tokenizer.encode(text)) for text in clean_texts]
    noisy_ids = [tokenizer.pad_sequence(tokenizer.encode(text)) for text in noisy_texts]

    # Convert to tensors
    clean_ids_tensor = torch.tensor(clean_ids, dtype=torch.long)
    noisy_ids_tensor = torch.tensor(noisy_ids, dtype=torch.long)

    return {
        "clean_ids": clean_ids_tensor,
        "noisy_ids": noisy_ids_tensor,
        "clean_texts": clean_texts,
        "noisy_texts": noisy_texts
    }

def get_dataloaders(
    train_paths: Union[str, Path, List[Union[str, Path]]],
    val_paths: Union[str, Path, List[Union[str, Path]]],
    tokenizer: Tokenizer,
    batch_size: int = 32,
    build_vocab: bool = True,
    vocab_save_path: str='vocab.json',
    truncate_train: Union[float, None] = None,
    truncate_test: Union[float, None] = None
) -> tuple:
    """Create train and validation dataloaders."""
    
    # Convert to list if single path
    if not isinstance(train_paths, list):
        train_paths = [train_paths]
    
    if not isinstance(val_paths, list):
        val_paths = [val_paths]
    
    # Convert string paths to Path objects
    train_paths = [Path(path) if isinstance(path, str) else path for path in train_paths]
    val_paths = [Path(path) if isinstance(path, str) else path for path in val_paths]
    
    print(f"Training on {len(train_paths)} files: {[p.name for p in train_paths]}")
    print(f"Validating on {len(val_paths)} files: {[p.name for p in val_paths]}")
    
    train_dataset = SpellingDataset(
        train_paths,
        tokenizer,
        build_vocab=build_vocab,
        truncate=truncate_train,
        vocab_save_path=vocab_save_path
    )

    val_dataset = SpellingDataset(
        val_paths,
        tokenizer,
        truncate=truncate_test
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )
    
    return train_dataloader, val_dataloader