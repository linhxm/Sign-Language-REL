# Sign-Language-REL

A research project for translating sign language into text. It extracts body and hand landmarks with MediaPipe, trains a translation model, and compares cross-entropy training with reinforcement learning.

The main dataset is **PHOENIX-2014T**, which contains German sign language weather broadcasts. A separate notebook compares I3D and MediaPipe features on How2Sign or PHOENIX.

**Pipeline:** video or image frames → pose features → encoder → Transformer decoder → text.

See the [project report](docs/Sign_Language_REL.pdf) for the experiments and results.

## About the project

Sign language translation turns a sequence of movements into a written sentence. The model needs to understand how body and hand movements change over time and how those movements relate to words in the target language.

This project studies that process using pose features: numerical descriptions of body and hand positions in each frame. The main workflow translates PHOENIX signing sequences into German text. It provides the tools to prepare data, train models, compare experiments, and try the resulting translations in a demo.

The main research question is whether reinforcement learning can improve translation quality after standard training. The experiments also compare different ways of encoding movement and examine how results change with the amount of training data.

## How it works

1. **Extract movement features.** MediaPipe detects body and hand landmarks from video or image frames. Each frame becomes a vector of 183 values, and the vectors form a sequence representing the signing motion.
2. **Encode the sequence.** A pose encoder processes the features across frames. The project includes Transformer, graph, temporal convolution, and Perceiver variants for comparison.
3. **Generate a sentence.** A Transformer decoder predicts text one token at a time. SentencePiece splits text into smaller units that the model can learn and generate.
4. **Train in two stages.** Cross-entropy (XE) first teaches the model to predict the reference text. A second stage applies a selected fine-tuning method. For example, SCST compares a sampled translation with the model's greedy translation and uses the reward difference to guide learning.
5. **Evaluate the output.** The scripts compare generated sentences with reference translations using BLEU-4, a measure of matching word sequences. They also collect training history, repetition and length statistics, and runtime measurements.

The reward can combine translation similarity with penalties for repeated phrases or sentences that are too short. These settings let the experiments examine both translation scores and common generation problems. Improvements are measured against the XE baseline; they are not assumed for every method or dataset size.

## What you can explore

- Compare translation quality before and after fine-tuning.
- Compare six pose encoders under a shared translation pipeline.
- Study the effect of training data size and reward settings.
- Compare precomputed I3D video features with MediaPipe features in a separate experiment.
- Load trained checkpoints in the demo and inspect their translations side by side.

## Project structure

| Path | Purpose |
| --- | --- |
| `notebooks/` | Kaggle workflows and a Colab demo |
| `configs/config.py` | Data paths, model settings, and training options |
| `data/` | Pose extraction, data loaders, tokenizer, and annotation CSVs |
| `models/` | Pose encoders and the translation model |
| `training/` | Cross-entropy and RL training methods |
| `scripts/` | Baselines, evaluation, and result summaries |
| `demo/` | Gradio app and inference code |
| `utils/` | Shared utilities |
| `docs/` | PDF report |

## Start with the notebooks

Run cells from top to bottom. Enable Internet access for setup, and edit the configuration cell to match your mounted datasets.

| Notebook | Purpose | Runtime |
| --- | --- | --- |
| [01 - Extract poses](notebooks/01_extract_poses.ipynb) | Extract and check PHOENIX pose features | Kaggle CPU |
| [02 - Train](notebooks/02_train.ipynb) | Select a dataset fraction, encoder, and training method | Kaggle GPU |
| [03 - Five-percent experiment](notebooks/03_experiment_5pct.ipynb) | Alternative workflow: extract, then train on 5% of the training split | Kaggle CPU, then GPU |
| [04 - Compare features](notebooks/04_compare_features.ipynb) | Compare precomputed I3D and MediaPipe features | Kaggle GPU |
| [05 - Demo](notebooks/05_demo_colab.ipynb) | Compare XE and SCST checkpoints using a Gradio app | Google Colab |

Use **01 → 02** for the main workflow, or **03** for the two-session 5% workflow. Notebook 04 is a separate experiment. Notebook 05 requires trained checkpoints and their matching tokenizer.

Raw images, pose caches, feature datasets, and trained checkpoints are not included. The annotation CSVs are in `data/phoenix-2014t-annotations/`. Validation and test splits remain full when training on a smaller subset. Even a 5% experiment performs real training and can take hours.

## Run from the command line

Run commands from the repository root. Install PyTorch for your machine, then the packages used by the training scripts:

```bash
python -m pip install torch numpy pandas sentencepiece sacrebleu tqdm matplotlib
```

For pose extraction, use an environment compatible with the project's MediaPipe version:

```bash
python -m pip install "mediapipe==0.10.14" opencv-python-headless
python data/extract_poses.py --input_dir /path/to/phoenix/frames --out_dir ./poses_out
```

Set `phoenix_root`, `pose_cache_dir`, and `work_dir` in [configs/config.py](configs/config.py). The default paths target Kaggle. Then run one experiment:

```bash
python main.py --subset 0.05 --encoder transformer --algo scst --phase all
```

To run experiment groups or select a smaller combination:

```bash
python run_all.py --subset 0.05 --groups core,encoders
python train_select.py --mode single --encoder stgcn --algo scst --subset 0.05
```

Encoders: `transformer`, `stgcn`, `gcn`, `graph_transformer`, `tcn`, `perceiver`.
Training methods: cross-entropy followed by `scst`, `ppo`, `mrt`, `raml`, or `dpo`.
`run_all.py` also includes REINFORCE, A2C, curriculum, reward, and latency comparisons.

Experiment groups are `core`, `encoders`, `algos`, `ablations`, `reward`, and `latency`, or `all`. Run `core` before groups that reuse its checkpoint. Completed steps are skipped when their output files and completion markers remain in the same work directory.

## Results and demo

`run_all.py` writes checkpoints, JSON metrics, comparison tables, and report figures under `work_dir`. To rebuild summaries from saved results:

```bash
python scripts/aggregate_results.py --work_dir /path/to/results --out /path/to/results/comparison_table
python scripts/make_report.py --work_dir /path/to/results
```

Keep the JSON metrics for analysis and the checkpoints plus `spm.model` for inference. See [demo instructions](demo/README.md) to run the interface locally.

This is an experimental model trained on a limited domain. Translation quality on unrelated videos may be poor.
