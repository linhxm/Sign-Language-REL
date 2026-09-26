# Translation demo

Compare cross-entropy (XE) and SCST translations from the same video or pose file.
The interface uses the model code from this repository.

## Google Colab

Open [05_demo_colab.ipynb](../notebooks/05_demo_colab.ipynb) and run the cells in order.
Upload your checkpoints and the tokenizer used during training to Google Drive:

```text
MyDrive/slt_demo/
├── spm.model
└── run1_transformer_subset25/
    ├── best_xe.pt
    └── last_rl.pt
```

Set `DRIVE_DIR` in the notebook to this folder. The demo uses `last_rl.pt` to show
the final SCST model, including runs where SCST did not improve over XE.

## Local use

From the repository root, install the demo dependencies and start the app:

```bash
python -m pip install torch numpy sentencepiece gradio opencv-python-headless "mediapipe==0.10.14"
python demo/app.py --results /path/to/runs --spm /path/to/spm.model
```

Use an environment compatible with MediaPipe 0.10.14. Optional flags include
`--device cpu`, `--device cuda`, `--port`, and `--share` for a public Gradio link.

`--results` is the parent directory containing run folders. Inputs can be a video
or an extracted `.npz` file with a `pose` array of shape `[T, 183]`.
The model was trained on German weather broadcasts; unrelated videos may produce
unreliable translations.
