import os
from pathlib import Path
import copy
import torch
import glob

            
def rotating_save_checkpoint(state, prefix, path="./checkpoints", nb=5):    
    if not os.path.isdir(path):
        os.makedirs(path)
    filenames = []
    first_empty = None
    best_filename = Path(path) / f"{prefix}_best.pth"
    torch.save(state, best_filename)
    for i in range(nb):
        filename = Path(path) / f"{prefix}_{i}.pth"
        if not os.path.isfile(filename) and first_empty is None:
            first_empty = filename
        filenames.append(filename)
    
    if first_empty is not None:
        torch.save(state, first_empty)
    else:
        first = filenames[0]
        os.remove(first)
        for filename in filenames[1:]:
            os.rename(filename, first)
            first = filename
        torch.save(state, filenames[-1])

def build_checkpoint(exp_name, unique_id, tpe, model, optimizer, acc, loss, epoch):
    return {
        "exp_name": exp_name,
        "unique_id": unique_id,
        "type": tpe,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "acc": acc,
        "loss": loss,
        "epoch": epoch,
    }
            
def restore_checkpoint(filename, model=None, optimizer=None):
    """restores checkpoint state from filename and load in model and optimizer if provided"""
    print(f"Extracting state from {filename}")

    state = torch.load(filename)
    if model:
        print(f"Loading model state_dict from state found in {filename}")
        model.load_state_dict(state["model"])
    if optimizer:
        print(f"Loading optimizer state_dict from state found in {filename}")
        optimizer.load_state_dict(state["optimizer"])
    return state

def restore_best_checkpoint(prefix, path="./checkpoints", model=None, optimizer=None):
    filename = Path(path) / f"{prefix}_best"
    return restore_checkpoint(filename, model, optimizer)


def restore_best_checkpoint(exp_name, unique_id, tpe,
                            model=None, optimizer=None, path="./checkpoints", extension="pth"):
    filename = Path(path) / f"{exp_name}_{unique_id}_{tpe}_best.{extension}"
    return restore_checkpoint(filename, model, optimizer)

