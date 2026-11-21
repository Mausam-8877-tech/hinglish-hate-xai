# Configs

One YAML file per experiment. Run with `python scripts/train.py --config configs/<file>.yaml`.

| File | Task | What |
|---|---|---|
| `baseline_cnn.yaml` | Member 2 | Kim CNN on HASOC |
| `baseline_bilstm.yaml` | Member 2 | BiLSTM + attention on HASOC |
| `muril_vanilla.yaml` | Member 3 | MuRIL fine-tuned on HASOC |
| `xlmr_vanilla.yaml` | Member 3 | XLM-R fine-tuned on HASOC |
| `muril_hx_lambda0.yaml` | Member 3 | MuRIL on HASOC + HateXplain, no rationale loss (control) |
| `muril_rationale.yaml` | Member 3 | MuRIL on HASOC + HateXplain + rationale loss (`--set loss.lambda=...`) |
