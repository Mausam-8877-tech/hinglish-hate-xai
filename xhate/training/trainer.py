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

One training step (inside ``fit``), in order:
    zero_grad -> forward (logits) -> loss -> backward (gradients) -> clip -> optimizer.step

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

Q: Why Adam?
A: Adam adapts the step size for every weight on its own, so it trains well
   with the default learning rate (1e-3) and needs little tuning. It is the
   usual optimiser for CNN / LSTM text classifiers.

Q: What are the optional ``optimizer`` / ``scheduler`` arguments? (added by Member 3)
A: Transformers are fine-tuned with AdamW (decoupled weight decay) and a learning rate
   that warms up and then decays linearly, not with plain Adam at 1e-3. ``fit`` accepts
   a ready-made optimizer and a per-step LR scheduler; when they are not given it
   behaves exactly as before (Adam with ``lr`` and ``weight_decay``).

Q: Why optimizer.zero_grad() at the start of every step?
A: PyTorch ADDS new gradients to the old ones. Without clearing them, every
   step would also apply the gradients of all earlier batches.

Q: Why is the loss multiplied by the batch size before averaging?
A: The last batch is usually smaller. Summing loss * batch_size and dividing by
   the number of posts gives the true mean over posts, not a mean of batch means.
"""

# Path: OS-independent file paths (for the checkpoint file).
from pathlib import Path

# torch: tensors, no_grad and saving checkpoints.
import torch
# sklearn metrics: standard, well-tested implementations of the scores we report.
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
# nn: CrossEntropyLoss and gradient clipping.
from torch import nn


def to_device(batch, device):
    """Move every tensor in a batch dict to ``device`` (other values are left as they are)."""
    # Tensors are moved; anything else (e.g. a list of ids) is kept unchanged.
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def default_loss_fn(weight=None):
    """Return ``loss_fn(logits, batch)``: cross-entropy on batch["label"], optionally class-weighted.

    Args:
        weight: tensor with one weight per class (from ``losses.class_weights``), or None.

    Q: Why return a function instead of the loss module itself?
    A: ``fit`` calls every loss as ``loss_fn(logits, batch)``. Wrapping the
       cross-entropy this way lets Member 3 swap in a loss that also uses the
       rationale fields of the batch, without changing ``fit``.
    """
    # nn.CrossEntropyLoss = softmax + negative log-likelihood, with optional class weights.
    criterion = nn.CrossEntropyLoss(weight=weight)

    def loss_fn(logits, batch):
        """Cross-entropy between the logits and the gold labels."""
        # Make sure the class weights are on the same device (CPU / GPU) as the logits.
        criterion.to(logits.device)
        # Average loss over the batch.
        return criterion(logits, batch["label"])

    # Return the inner function (it "remembers" criterion: a closure).
    return loss_fn


def compute_metrics(labels, preds):
    """Accuracy, macro-F1 and per-class scores.

    Q: Why does Member 2 compute metrics here and not in xhate/evaluation/?
    A: ``xhate/evaluation/classification.py`` belongs to Member 4's task. We need
       accuracy + macro-F1 for early stopping now, so we use scikit-learn directly;
       Member 4 can replace this with their own function later.

    Q: What are precision, recall and F1?
    A: For one class, e.g. HOF: precision = of the posts we CALLED HOF, how many
       really are; recall = of the posts that really ARE HOF, how many we found;
       F1 = their harmonic mean, 2 * P * R / (P + R).

    Returns a dict that can be written to JSON directly.
    """
    # Precision, recall, F1 and support for class 0 (NOT) and class 1 (HOF).
    # zero_division=0: a class the model never predicts scores 0 instead of a warning.
    p, r, f, support = precision_recall_fscore_support(labels, preds, labels=[0, 1], zero_division=0)
    # Everything as plain Python numbers (JSON cannot store numpy types).
    return {
        "accuracy": float(accuracy_score(labels, preds)),                                 # share of correct predictions
        "macro_f1": float(f1_score(labels, preds, average="macro", labels=[0, 1], zero_division=0)),  # mean F1 of the 2 classes
        "per_class": {                                                                    # scores for NOT and for HOF
            name: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),  # scores of class i
                   "support": int(support[i])}  # support = number of gold posts of this class
            for i, name in enumerate(["NOT", "HOF"])  # i = 0 for NOT, 1 for HOF
        },  # end of per_class
        # Rows = gold class, columns = predicted class: [[NOT->NOT, NOT->HOF], [HOF->NOT, HOF->HOF]].
        "confusion_matrix": confusion_matrix(labels, preds, labels=[0, 1]).tolist(),  # numpy array -> list
    }  # end of the metrics dict


@torch.no_grad()  # no gradients needed while scoring: faster and uses less memory
def predict(model, loader, device, loss_fn=None):
    """Run the model over a whole DataLoader.

    Returns:
        dict with "labels", "preds" (lists of ints), "probs" (P(HOF) per post, floats)
        and "loss" (mean loss, only if ``loss_fn`` is given, else None).
    """
    # Evaluation mode: dropout off.
    model.eval()
    # Collected gold labels, predicted labels and P(HOF), one entry per post.
    labels, preds, probs = [], [], []
    # Running loss (sum of loss * batch size) and number of posts seen.
    total_loss, total_n = 0.0, 0
    # Go through the data batch by batch.
    for batch in loader:
        # Move the batch to the CPU / GPU.
        batch = to_device(batch, device)
        # Forward pass: (B, C) logits.
        logits = model(batch["input_ids"], batch["attention_mask"])
        # Optional loss (e.g. validation loss for the training log).
        if loss_fn is not None:
            # Add this batch's loss, weighted by its size (see Q&A).
            total_loss += loss_fn(logits, batch).item() * logits.size(0)  # .item() = tensor -> Python float
            # Count the posts.
            total_n += logits.size(0)
        # Probabilities with softmax; column 1 = P(HOF).
        probs.extend(logits.softmax(dim=-1)[:, 1].tolist())
        # Predicted class = the larger logit.
        preds.extend(logits.argmax(dim=-1).tolist())
        # Gold labels.
        labels.extend(batch["label"].tolist())
    # Mean loss over all posts, or None if no loss was computed.
    loss = total_loss / total_n if total_n else None
    # Return everything.
    return {"labels": labels, "preds": preds, "probs": probs, "loss": loss}


def fit(model, train_loader, val_loader, device, epochs=20, lr=1e-3, weight_decay=0.0,
        patience=3, grad_clip=5.0, loss_fn=None, checkpoint_path=None, log=print,
        optimizer=None, scheduler=None):
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
        optimizer: a torch optimizer to use instead of Adam (``lr``/``weight_decay`` are then ignored).
        scheduler: an LR scheduler stepped after EVERY optimizer step (e.g. linear warmup), or None.

    Returns:
        dict with "history" (one entry per epoch), "best_epoch" and "best_val" (metrics).
    """
    # Default loss: unweighted cross-entropy.
    loss_fn = loss_fn or default_loss_fn()
    # Put the model's weights on the device.
    model.to(device)
    # Adam optimiser over all the model's weights (see Q&A), unless the caller gave one.
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Best validation macro-F1 so far (-1 so the first epoch always counts as better).
    best_f1 = -1.0
    # Best epoch number, its val metrics, and a copy of its weights.
    best_epoch, best_val, best_state = 0, None, None
    # Epochs in a row without improvement (for early stopping).
    bad_epochs = 0
    # One dict per epoch (saved to history.json for the report's training curves).
    history = []
    # One epoch = one full pass over the training data.
    for epoch in range(1, epochs + 1):
        # Training mode: dropout on.
        model.train()
        # Running training loss (sum of loss * batch size) and posts seen.
        running, seen = 0.0, 0
        # Go through the training data batch by batch.
        for batch in train_loader:
            # Move the batch to the device.
            batch = to_device(batch, device)
            # Clear the gradients from the previous step (see Q&A).
            optimizer.zero_grad()
            # Forward pass: (B, C) logits.
            logits = model(batch["input_ids"], batch["attention_mask"])
            # Loss for this batch.
            loss = loss_fn(logits, batch)
            # Backward pass: compute the gradient of the loss for every weight.
            loss.backward()
            # Clip very large gradients (see Q&A); 0 / None means no clipping.
            if grad_clip:
                # Rescale all gradients together so their total norm is at most grad_clip.
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            # Update the weights one step against the gradient.
            optimizer.step()
            # Move the learning rate one step along its schedule (if there is one).
            if scheduler is not None:
                scheduler.step()
            # Add to the running loss (weighted by batch size).
            running += loss.item() * logits.size(0)
            # Count the posts.
            seen += logits.size(0)
        # Score the validation set (also computes the val loss).
        val_out = predict(model, val_loader, device, loss_fn=loss_fn)
        # Accuracy, macro-F1, ... on val.
        val_metrics = compute_metrics(val_out["labels"], val_out["preds"])
        # Everything about this epoch, for the log and history.json.
        entry = {"epoch": epoch,                               # epoch number (1, 2, ...)
                 "train_loss": running / max(seen, 1),         # mean training loss
                 "val_loss": val_out["loss"],                  # mean validation loss
                 "val_accuracy": val_metrics["accuracy"],      # validation accuracy
                 "val_macro_f1": val_metrics["macro_f1"]}      # validation macro-F1 (selection metric)
        # Keep it.
        history.append(entry)
        # Print one line per epoch.
        log(f"epoch {epoch:3d} | train_loss {entry['train_loss']:.4f} | val_loss {entry['val_loss']:.4f} "
            f"| val_acc {entry['val_accuracy']:.4f} | val_macro_f1 {entry['val_macro_f1']:.4f}")  # :.4f = 4 decimals
        # Better than every earlier epoch?
        if val_metrics["macro_f1"] > best_f1:
            # Remember this epoch as the best one.
            best_f1, best_epoch, best_val = val_metrics["macro_f1"], epoch, val_metrics
            # Copy the weights (clone, so later training steps do not change the copy).
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            # Save them to disk if a path was given.
            if checkpoint_path:
                # Create the run folder if needed.
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                # Write the weights to best.pt.
                torch.save(best_state, checkpoint_path)
            # Improvement: reset the early-stopping counter.
            bad_epochs = 0
        # No improvement this epoch.
        else:
            # One more epoch without improvement.
            bad_epochs += 1
            # Stop if we have waited ``patience`` epochs.
            if bad_epochs >= patience:
                # Say why training stopped.
                log(f"early stopping: no improvement for {patience} epochs (best epoch {best_epoch})")
                # Leave the epoch loop.
                break
    # Put the best weights back into the model (so test is scored with the best epoch).
    if best_state is not None:
        # Overwrite the current (last-epoch) weights.
        model.load_state_dict(best_state)
    # Return the training record.
    return {"history": history, "best_epoch": best_epoch, "best_val": best_val}
