# Local Training Setup

Run everything on your local machine instead of Colab.

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (8GB+ VRAM recommended)
- Git

## Step 1: Install Dependencies

```bash
# Install the environment package
pip install -e .

# Install training dependencies
pip install "unsloth[cu121-torch240]"  # Adjust for your CUDA version
pip install "trl>=0.11.0" transformers accelerate peft datasets
pip install requests uvicorn fastapi
```

## Step 2: Start Local Environment Server

Open a terminal and run:

```bash
python -m uvicorn negotiate_env.server.app:app --host 0.0.0.0 --port 7860
```

Keep this terminal open. The server will run at `http://localhost:7860`

## Step 3: Test the Server (Optional)

In another terminal:

```bash
python test_server.py
```

This will verify the server is working correctly.

## Step 4: Run Training

In another terminal:

```bash
python train_negotiate_unsloth.py \
    --env-url http://localhost:7860 \
    --model-id Qwen/Qwen2.5-1.5B-Instruct \
    --output-dir negotiate-long-horizon-output \
    --num-episodes 1000 \
    --max-turns 50
```

## Step 5: Monitor Progress

Training will show progress bars and log rewards. Expected time:
- T4 GPU: ~4-6 hours
- RTX 3090: ~2-3 hours
- RTX 4090: ~1-2 hours

## Troubleshooting

### Server won't start
```bash
# Kill any existing process on port 7860
lsof -ti:7860 | xargs kill -9

# Try starting again
python -m uvicorn negotiate_env.server.app:app --host 0.0.0.0 --port 7860
```

### Out of memory errors
Reduce batch size in training script:
- Edit `train_negotiate_unsloth.py`
- Change `per_device_train_batch_size=2` to `=1`
- Change `num_generations=4` to `=2`

### Check server logs
The server terminal will show any errors. Look for:
- 500 errors (server crashes)
- 422 errors (invalid action format)
- Connection errors

## Output

After training completes:
- Model saved to: `negotiate-long-horizon-output/`
- Merged model: `negotiate-long-horizon-output/merged/`
- Training logs: `negotiate-long-horizon-output/trainer_state.json`

## Upload to HuggingFace (Optional)

```python
from huggingface_hub import HfApi, login

login()  # Enter your HF token

api = HfApi()
api.upload_folder(
    folder_path="negotiate-long-horizon-output",
    repo_id="YourUsername/negotiate-env-long-horizon",
    repo_type="model",
)
```
