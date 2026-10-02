# Verification Record

Executed on this server on 2026-10-02 using the existing `amc` environment:
Python 3.10.19, PyTorch 2.10.0+cu128, NVIDIA RTX PRO 6000 Blackwell Max-Q.

## Focused Tests

`test_workflow.py`: **7 tests passed**.

- All 84 state tensor names/shapes match the reference checkpoint; the model
  contains 3,238,312 trainable parameters. Verification leaves initialized
  parameters unchanged and does not load the reference weights.
- Signal decoding, cyclic class order, and float16 BOI bounds match the
  original reader for the checked examples.
- Full-length evaluation preprocessing matches the existing `FeatureExtract`
  helper exactly for the checked examples.
- Forward output is finite and normalized for every requested crop length.
- IQ backbone, CC backbone and LAG all receive finite, nonzero gradients.
- The crop sampler visits exactly the five requested lengths in the test.
- Sample-weighted means and output-path boundary checks pass.

These are implementation checks, not evidence of scientific convergence or
equivalence of the original staged experiment and the new joint experiment.

## Small Real-Data Integration Run

`verification_runs/smoke` ran two epochs on 32 training and 16 held-out signals,
batch size 8, with dense interval 1 and sparse interval 3:

- 8 optimizer steps; 6 step-CSV rows and 2 epoch-CSV rows.
- Recorded intervals: 1, 1, 1, 1, 3, 1 steps. The final partial window is saved.
- Global steps: 1, 2, 3, 4, 7, 8. Total recorded training samples: 64.
- Epoch means computed from step-window means exactly match epoch-CSV means.
- `last.pt`, `best.pt`, PNG/PDF plots and completion status are produced.
- Loading the completed run through `--resume` restores the checkpoint and
  optimizer/scheduler/RNG state, and does not append duplicate step/epoch rows.

Smoke measurements are kept separate and must not be used as thesis results.
A subsequent source change only created the local kernel-cache directory and
improved the epoch status timestamp; all 7 focused tests were rerun and passed
without the previous cache warning. Resume intentionally rejects provenance
changes to this older smoke run.

## Full-Size Memory Probe

One forward/backward/AdamW step was executed with batch size **160**, signal
length **32768**, and the reconstructed model. Only for this probe, preprocessing
was set to evaluation mode to force the longest length, while the backbones
and fusion remained in training mode. This is not the full run's protocol.

- Finite loss and finite gradients; optimizer step completed.
- Peak PyTorch allocated GPU memory: approximately **13.78 GiB**.
- Observed time for that single step: approximately **1.46 seconds**.

This does not guarantee future GPU availability or bound total runtime. The
full experiment uses random lengths, BOI preprocessing and the actual split.
