"""Training / evaluation loop shared by every model.

    from xhate.training.trainer import fit, predict, compute_metrics

    history = fit(model, train_loader, val_loader, loss_fn=loss_fn, device=device,
                  epochs=20, lr=1e-3, patience=3, checkpoint_path="results/run/best.pt")
    out = predict(model, test_loader, device)            # labels, preds, probs
    metrics = compute_metrics(out["labels"], out["preds"])

What ``fit`` expects (so Member 3 can reuse it without changing this file):
    * every batch is a dict with at least "input_ids", "attention_mask", "label";
    * ``model(batch["input_ids"], batch["attention_mask"])`` returns logits (B, C);
    * ``loss_fn(logits, batch)`` returns a scalar loss. The default is plain
      cross-entropy on batch["label"]; a rationale-supervised loss can read
      batch["rationale"] and batch["rationale_mask"] from the same batch.

Q&A
---
Q: What is early stopping?
A: After every epoch we score the validation set. If macro-F1 has not improved
   for ``patience`` epochs in a row, we stop. Training longer would only make the
   model memorise the training posts (overfitting).

Q: Why select on macro-F1 and not accuracy or loss?
A: Macro-F1 is the official HASOC metric. It averages the F1 of NOT and HOF
   equally, so a model that ignores the smaller class gets a low score even when
   its accuracy looks fine.

Q: What is saved in the checkpoint?
A: Only the weights (``model.state_dict()``) of the epoch with the best
   validation macro-F1. At the end of ``fit`` these best weights are loaded back
   into the model, so the test set is scored with the best model, not the last.

Q: Why gradient clipping?
A: LSTMs can produce very large gradients ("exploding gradients") that wreck
   the weights in one step. Clipping the total gradient norm (e.g. to 5) stops that.

Q: Why model.train() / model.eval()?
A: Dropout must be ON while training and OFF while scoring. These two calls
   switch it; ``torch.no_grad()`` also skips gradient bookkeeping when scoring.
"""

# Path: OS-independent file paths.
from pathlib import Path

# torch: tensors, saving checkpoints.
import torch
# sklearn metrics: the standard, well-tested implementations.
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
# nn: CrossEntropyLoss.
from torch import nn


def to_device(batch, device):
    """Move every tensor in a batch dict to ``device`` (other values are left as they are)."""
    # Tensors are moved; anything else (e.g. a list of ids) is kept unchanged.
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def default_loss_fn(weight=None):
    """Return ``loss_fn(logits, batch)``: cross-entropy on batch["label"], optionally class-weighted."""
    # nn.CrossEntropyLoss = softmax + negative log-likelihood, with optional class weights.
    criterion = nn.CrossEntropyLoss(weight=weight)

    def loss_fn(logits, batch):
        """Cross-entropy between the logits and the gold labels."""
        # Make sure the weights are on the same device as the logits.
        criterion.to(logits.device)
        # Average loss over the batch.
        return criterion(logits, batch["label"])

    # Return the function.
    return loss_fn


def compute_metrics(labels, preds):
    """Accuracy, macro-F1 and per-class scores.

    Q: Why does Member 2 compute metrics here and not in xhate/evaluation/?
    A: ``xhate/evaluation/classification.py`` belongs to Member 4's task. We need
       accuracy + macro-F1 for early stopping now, so we use scikit-learn directly;
       Member 4 can replace this with their own function later.

    Returns a dict that can be written to JSON directly.
    """
    # Precision, recall and F1 for class 0 (NOT) and class 1 (HOF).
    # zero_division=0: a class the model never predicts scores 0 instead of a warning.
    p, r, f, support = precision_recall_fscore_support(labels, preds, labels=[0, 1], zero_division=0)
    # Everything as plain Python numbers (JSON cannot store numpy types).
    return {
        "accuracy": float(accuracy_score(labels, preds)),                                 # share of correct predictions
        "macro_f1": float(f1_score(labels, preds, average="macro", labels=[0, 1], zero_division=0)),  # mean F1 of the 2 classes
        "per_class": {
            name: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                   "support": int(support[i])}  # support = number of gold posts of this class
            for i, name in enumerate(["NOT", "HOF"])
        },
        # Rows = gold class, columns = predicted class: [[NOT->NOT, NOT->HOF], [HOF->NOT, HOF->HOF]].
        "confusion_matrix": confusion_matrix(labels, preds, labels=[0, 1]).tolist(),
    }


@torch.no_grad()
def predict(model, loader, device, loss_fn=None):
    """Run the model over a whole DataLoader.

    Returns:
        dict with "labels", "preds" (lists of ints), "probs" (P(HOF) per post, floats)
        and "loss" (mean loss, only if ``loss_fn`` is given, else None).
    """
    # Evaluation mode: dropout off.
    model.eval()
    # Collected outputs.
    labels, preds, probs = [], [], []
    # Running loss (weighted by batch size).
    total_loss, total_n = 0.0, 0
    for batch in loader:
        # Move the batch to the CPU / GPU.
        batch = to_device(batch, device)
        # Forward pass: (B, C) logits.
        logits = model(batch["input_ids"], batch["attention_mask"])
        # Optional loss (e.g. validation loss for the training log).
        if loss_fn is not None:
            total_loss += loss_fn(logits, batch).item() * logits.size(0)  # sum over posts
            total_n += logits.size(0)
        # Probabilities with softmax; column 1 = P(HOF).
        probs.extend(logits.softmax(dim=-1)[:, 1].tolist())
        # Predicted class = the larger logit.
        preds.extend(logits.argmax(dim=-1).tolist())
        # Gold labels.
        labels.extend(batch["label"].tolist())
    # Mean loss if it was computed.
    loss = total_loss / total_n if total_n else None
    # Return everything.
    return {"labels": labels, "preds": preds, "probs": probs, "loss": loss}


def fit(model, train_loader, val_loader, device, epochs=20, lr=1e-3, weight_decay=0.0,
        patience=3, grad_clip=5.0, loss_fn=None, checkpoint_path=None, log=print):
    """Train with early stopping on validation macro-F1 and keep the best weights.

    Args:
        model: any nn.Module following the interface in the module docstring.
        train_loader / val_loader: DataLoaders yielding batch dicts.
        device: torch.device to train on.
        epochs: maximum number of passes over the training data.
        lr: learning rate for Adam.
        weight_decay: L2 penalty (0 = off).
        patience: stop after this many epochs without a better val macro-F1.
        grad_clip: maximum gradient norm (None or 0 = no clipping).
        loss_fn: ``loss_fn(logits, batch)``; default = plain cross-entropy.
        checkpoint_path: where to save the best weights (None = keep in memory only).
        log: function used to print progress (``print`` by default).

    Returns:
        dict with "history" (one entry per epoch), "best_epoch" and "best_val" (metrics).
    """
    # Default loss: unweighted cross-entropy.
    loss_fn = loss_fn or default_loss_fn()
    # Put the model on the device.
    model.to(device)
    # Adam: adapts the step size per weight; the usual default for these models.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Best validation macro-F1 so far (-1 so the first epoch always counts as better).
    best_f1 = -1.0
    # Best epoch, its metrics, and a copy of its weights.
    best_epoch, best_val, best_state = 0, None, None
    # Epochs since the last improvement.
    bad_epochs = 0
    # One dict per epoch (saved to history.json for the report's training curves).
    history = []
    for epoch in range(1, epochs + 1):
        # Training mode: dropout on.
        model.train()
        # Running training loss.
        running, seen = 0.0, 0
        for batch in train_loader:
            # Move the batch to the device.
            batch = to_device(batch, device)
            # Clear the gradients from the previous step.
            optimizer.zero_grad()
            # Forward pass.
            logits = model(batch["input_ids"], batch["attention_mask"])
            # Loss for this batch.
            loss = loss_fn(logits, batch)
            # Backward pass: compute gradients.
            loss.backward()
            # Clip very large gradients (see Q&A).
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            # Update the weights.
            optimizer.step()
            # Add to the running loss (weighted by batch size).
            running += loss.item() * logits.size(0)
            seen += logits.size(0)
        # Score the validation set.
        val_out = predict(model, val_loader, device, loss_fn=loss_fn)
        val_metrics = compute_metrics(val_out["labels"], val_out["preds"])
        # Log line for this epoch.
        entry = {"epoch": epoch, "train_loss": running / max(seen, 1), "val_loss": val_out["loss"],
                 "val_accuracy": val_metrics["accuracy"], "val_macro_f1": val_metrics["macro_f1"]}
        history.append(entry)
        log(f"epoch {epoch:3d} | train_loss {entry['train_loss']:.4f} | val_loss {entry['val_loss']:.4f} "
            f"| val_acc {entry['val_accuracy']:.4f} | val_macro_f1 {entry['val_macro_f1']:.4f}")
        # Better than every earlier epoch?
        if val_metrics["macro_f1"] > best_f1:
            # Remember this epoch as the best one.
            best_f1, best_epoch, best_val = val_metrics["macro_f1"], epoch, val_metrics
            # Copy the weights (clone, so later training steps do not change the copy).
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            # Save them to disk if a path was given.
            if checkpoint_path:
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)  # make the folder
                torch.save(best_state, checkpoint_path)
            # Reset the early-stopping counter.
            bad_epochs = 0
        else:
            # One more epoch without improvement.
            bad_epochs += 1
            # Stop if we have waited long enough.
            if bad_epochs >= patience:
                log(f"early stopping: no improvement for {patience} epochs (best epoch {best_epoch})")
                break
    # Put the best weights back into the model.
    if best_state is not None:
        model.load_state_dict(best_state)
    # Return the training record.
    return {"history": history, "best_epoch": best_epoch, "best_val": best_val}
