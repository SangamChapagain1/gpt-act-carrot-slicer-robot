# GPT-ACT Carrot Slicer (SO101 Demo)
A complete, runnable example of a modular VLAM system where a realtime VLM (GPT) plans and three ACT skills execute on a physical SO101 robot: pick-and-place, slice with a tool, and transfer sliced pieces.

## Overview
This repo extends LeRobot with a GPT-powered voice assistant that can run three ACT skills in sequence:
pick-and-place, use slicer, and transfer slices. It is meant as a complete carrot-slicing demo on a real SO-101.

![Robot setup](assets/Carrot_Slicer_Robot_Setup.png)
![Top camera view](assets/Top_Camera_View.png)

## Before you start
- **Have SO-101 + LeRobot working first**: follow the official LeRobot docs to assemble, wire, and teleoperate the arm.
- **Be comfortable with Python and the terminal**: creating a virtualenv, running `git clone`, `pip install`, and Python scripts.
- **Verify basic LeRobot scripts**: make sure you can teleop and record with SO-101 from the LeRobot repo before using this assistant.
- **Have or plan to train ACT policies**: either use your own Hugging Face ACT models, or record datasets and train them using the provided Colab notebooks.

This repo contains:
- AI voice assistant (FastAPI backend + minimal frontend)
- Three ACT inference scripts (pick, slice, transfer)
- Optional SmolVLA inference scripts

Hardware:SO101 follower arm (single-arm) with a top RGB camera and optional wrist camera.

## Quick Start

1) Install LeRobot and dependencies (one-time)
```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e .               # base
pip install -e ".[smolvla]"    # optional: for SmolVLA scripts
```

2) Clone this repo and configure ports/cameras
```bash
cd path/to/your/workspace
# This folder you are in now
```
Edit `src/config/ports_and_cameras.py` to match your USB ports and camera indices.

3) Create .env with your OpenAI key
```bash
cd gpt-act-carrot-slicer
cp .env.example .env
# edit .env and set OPENAI_API_KEY=...
```

4) Start the AI assistant (voice + vision)
```bash
source setup.sh
./start_ai_assistant.sh
# Open http://localhost:8080 and click "Connect and Start"
```

5) Record your own datasets (before training)
```bash
source setup.sh
python src/data/record_pick_and_place.py
python src/data/record_use_slicer_to_slice_carrot.py
python src/data/record_transfer_slices_to_pile.py
```
Each script records ~50 episodes and uploads to HuggingFace. Edit the REPO_ID inside each script to match your HF username.

6) Train policies on Google Colab
See `colab_training_examples/` for ready-to-use Jupyter notebooks:
- `train_act_pick_and_place.ipynb` - Pick and place ACT training (~1.5h on A100)
- `train_act_use_slicer.ipynb` - Use slicer ACT training (~1.5h on A100)
- `train_act_transfer_slices.ipynb` - Transfer slices ACT training (~1.5h on A100)

Upload these to [Google Colab](https://colab.research.google.com/), update the dataset/model IDs, and run all cells. Train all three for the full workflow.

7) Run ACT policies directly (without voice)
```bash
source setup.sh
python scripts/run_inference_pick_and_place.py
python scripts/run_inference_use_slicer.py
python scripts/run_inference_transfer_slices.py
```

Optional: SmolVLA (loads but not yet reliable with just 20k steps, will update with more steps)
```bash
python scripts/run_inference_smolvla_pick_and_place.py
python scripts/run_inference_smolvla_use_slicer.py
python scripts/run_inference_smolvla_transfer_slices.py
```

## Hugging Face Links (example)
- Datasets:
  - `sangam-101/so101-pick-and-place-carrot`
  - `sangam-101/so101-slicer-to-slice-carrot`
  - `sangam-101/so101-transfer-slices-to-pile`
- Models (ACT):
  - `sangam-101/act_so101_pick_and_place_carrot_policy`
  - `sangam-101/act_so101_slicer_to_slice_carrot_policy`
  - `sangam-101/act_so101_transfer_slices_to_pile`
- Models (SmolVLA fine-tunes; 20k steps demo):
  - `sangam-101/smolvla_so101_pick_and_place_carrot`
  - `sangam-101/smolvla_so101_slicer_to_slice_carrot`
  - `sangam-101/smolvla_so101_transfer_slices_to_pile`

## Architecture (in short)
- **Planner**: GPT Realtime with vision, language, and function calls.
- **Executors**: Three ACT visuomotor policies (pick-and-place, use slicer, transfer slices) trained with LeRobot.
- **Boundary**: Planner → executor → camera-based verification after each step.

## Roadmap
- Add reset / recovery / home skills for long-horizon autonomy (24/7-style operation).
- Expand the ACT skill library based on task requirements. 

## Requirements
See `requirements.txt`. Install into your environment:
```bash
pip install -r requirements.txt
```
LeRobot must be installed (see step 1).

## Python environment notes
- Use a single virtualenv (e.g., `robotics_env`) for both `lerobot` and this repo.
- A common layout is:
  ```bash
  projects_root/
    lerobot/
    gpt-act-carrot-slicer/
    robotics_env/
  ```
- Activate `robotics_env` before running any commands in this README.
- `setup.sh` assumes the env lives one level up as `../robotics_env`. If your layout is different, either move the env or update that path in `setup.sh`.

## Repo Structure
```
gpt-act-carrot-slicer/
  ai_assistant/
    backend/        # FastAPI server + tools
    frontend/       # Minimal Web UI
  scripts/          # Inference scripts (ACT + SmolVLA)
  src/              # Hardware + camera config for SO101
  start_ai_assistant.sh
  setup.sh
  requirements.txt
  .env.example
  .gitignore
  LICENSE
```

## Notes
- Safety first: supervise every run
- The assistant has “Demo mode” (single confirmation per carrot) and “Safe mode” (confirm per step). See `ai_assistant/backend/main.py`.
- SmolVLA in this repo is for demonstration (20k steps). For real performance, train >20k steps.

## License
Apache-2.0