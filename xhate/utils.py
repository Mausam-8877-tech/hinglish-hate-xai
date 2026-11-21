"""Shared helpers: config loading, seeding, device choice and JSON files.

Used by scripts/train.py and can be reused by every task:

    from xhate.utils import load_config, set_seed, get_device

    cfg = load_config("configs/baseline_cnn.yaml", overrides=["train.lr=0.0005"])
    set_seed(cfg["seed"])
    device = get_device(cfg["train"]["device"])

Q&A
---
Q: Why YAML configs instead of command-line flags?
A: One file per experiment is easy to read, easy to compare, and is saved next
   to the results, so we always know exactly which settings produced a number.

Q: What does an override like "train.lr=0.0005" do?
A: It changes ONE value of the config without editing the file. The part before
   "=" is the path inside the YAML (dots go one level deeper); the part after it
   is parsed as YAML, so "0.0005" becomes a float, "true" a bool and "[3, 4]" a list.
   This is what we use for the small hyperparameter search.

Q: Why does set_seed set so many seeds?
A: Python, NumPy and PyTorch each have their OWN random generator. Shuffling,
   dropout and weight initialisation use different ones, so all of them must be
   fixed for a run to be repeatable.
"""

# json: read and write .json files.
import json
# random: Python's own random generator (one of the seeds we fix).
import random
# Path: OS-independent file paths.
from pathlib import Path

# numpy: has its own random generator (another seed to fix).
import numpy as np
# yaml: reads the .yaml config files.
import yaml


def load_config(path, overrides=None):
    """Read a YAML config file and apply ``key.subkey=value`` overrides.

    Args:
        path: path of the .yaml file.
        overrides: optional list of strings like ``"train.lr=0.0005"``.

    Returns:
        The config as a nested dict.
    """
    # Open the file as UTF-8 and parse it into a dict.
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}  # an empty file becomes an empty dict
    # Apply every override in order (later ones win).
    for item in overrides or []:
        # Split "train.lr=0.0005" into "train.lr" and "0.0005".
        if "=" not in item:
            raise ValueError(f"Override {item!r} must look like key.subkey=value")  # clear message
        key, value = item.split("=", 1)  # split only at the FIRST "="
        # Walk down the dict along the dotted path, creating levels if needed.
        node = cfg
        parts = key.strip().split(".")  # "train.lr" -> ["train", "lr"]
        # Every part except the last is a section ("train").
        for part in parts[:-1]:
            node = node.setdefault(part, {})  # go one level deeper (create it if missing)
        # Parse the value as YAML so numbers / bools / lists get the right type.
        node[parts[-1]] = yaml.safe_load(value)
    # Return the final config.
    return cfg


def save_config(cfg, path):
    """Write a config dict back to a YAML file (used to save the exact run settings)."""
    # Make sure the folder exists.
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # sort_keys=False keeps the order of the original file.
    with Path(path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)  # allow_unicode: keep Hindi readable


def set_seed(seed):
    """Fix every random generator we use so a run can be repeated exactly (on CPU)."""
    # Python's built-in generator.
    random.seed(seed)
    # NumPy's generator.
    np.random.seed(seed)
    # PyTorch is imported here so that importing xhate.utils does not require torch.
    import torch
    # PyTorch CPU generator (also seeds every GPU generator).
    torch.manual_seed(seed)


def get_device(name="auto"):
    """Return a torch.device. "auto" picks the GPU if there is one, otherwise the CPU."""
    # Imported here so that importing xhate.utils does not require torch.
    import torch
    # "auto": use CUDA when available.
    if name in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")  # best available device
    # Otherwise use exactly what the config asked for ("cpu", "cuda", "cuda:1", ...).
    return torch.device(name)


def save_json(obj, path):
    """Write ``obj`` as pretty-printed UTF-8 JSON (Hindi stays readable)."""
    # Make sure the folder exists.
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # ensure_ascii=False keeps Devanagari as real characters instead of \\uXXXX.
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)  # indent=2: human-readable


def load_json(path):
    """Read a UTF-8 JSON file."""
    # Open and parse.
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)  # JSON -> dict / list
