# Google Colab Training Examples for GPT-ACT Carrot Slicer

This folder contains ready-to-use Google Colab notebooks for training ACT policies for all three tasks in the carrot slicing workflow.

## 📚 Available Notebooks

### ACT Training (All 3 Tasks)
1. **`train_act_pick_and_place.ipynb`**: Pick carrot from plate and place on cutting board
   - Dataset: `so101-pick-and-place-carrot`
   - Training time: ~1.5 hours (100k steps on A100)

2. **`train_act_use_slicer.ipynb`**: Pick slicer, slice carrot, return tool
   - Dataset: `so101-slicer-to-slice-carrot`
   - Training time: ~1.5 hours (100k steps on A100)

3. **`train_act_transfer_slices.ipynb`**: Transfer sliced carrots to pile
   - Dataset: `so101-transfer-slices-to-pile`
   - Training time: ~1.5 hours (100k steps on A100)

## 🚀 How to Use

1. **Upload to Google Colab**
   - Go to [Google Colab](https://colab.research.google.com/)
   - Click `File` → `Upload notebook`
   - Select one of the `.ipynb` files from this folder

2. **Set GPU Runtime**
   - Click `Runtime` → `Change runtime type`
   - Set `Hardware accelerator` to `GPU` (preferably A100)

3. **Update Dataset and Model IDs**
   - Open the training cell (usually cell 10)
   - Update `--dataset.repo_id` to your HuggingFace dataset
   - Update `--policy.repo_id` to where you want the trained model uploaded

4. **Run All Cells**
   - Click `Runtime` → `Run all`
   - Log in to W&B and HuggingFace when prompted

5. **Monitor Training**
   - Check W&B dashboard for training progress
   - Checkpoints are saved to your Google Drive at `/content/drive/MyDrive/lerobot_runs/`

6. **Repeat for All Tasks**
   - Train all three policies using their respective notebooks
   - You can run multiple Colab sessions in parallel if you have Colab Pro

## 📖 Workflow

1. **Record datasets** (on your local machine):
   ```bash
   cd gpt-act-carrot-slicer
   source setup.sh
   python src/data/record_pick_and_place.py
   python src/data/record_use_slicer_to_slice_carrot.py
   python src/data/record_transfer_slices_to_pile.py
   ```

2. **Train policies** (on Google Colab):
   - Upload and run `train_act_pick_and_place.ipynb`
   - Upload and run `train_act_use_slicer.ipynb`
   - Upload and run `train_act_transfer_slices.ipynb`

3. **Run inference** (on your local machine with robot):
   ```bash
   cd gpt-act-carrot-slicer
   source setup.sh
   ./start_ai_assistant.sh
   ```

## 📖 Additional Resources

- [LeRobot Documentation](https://github.com/huggingface/lerobot)
- [HuggingFace Hub](https://huggingface.co/)
- [Weights & Biases](https://wandb.ai/)

## ⚠️ Important Notes

- **Google Drive Mount**: These notebooks save checkpoints to Google Drive to persist across sessions. Make sure you have enough space (~5-10GB per training run × 3 tasks = ~15-30GB total).
- **Session Timeout**: Free Colab sessions may disconnect after ~12 hours. Use Colab Pro for longer sessions.
- **Resume Training**: If disconnected, you can resume from the last checkpoint by updating the training command with `--resume=true` and `--checkpoint_path=/path/to/last/checkpoint`.
- **Train in Order**: It's recommended to train pick-and-place first, then use_slicer, then transfer_slices, as they follow the natural workflow sequence.

