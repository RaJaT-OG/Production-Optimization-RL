
"""
Train the PPO agent on the simple arrival + burst + priority model.

This is the same MaskablePPO algorithm as
agents/train_ppo_jobshop.py, but every job has one operation and may
run on any machine, which is the input format used by the dashboard.

    $env:PYTHONIOENCODING = "utf-8"
    venv\\Scripts\\python.exe agents\\train_ppo_simple.py
"""

import sys

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import agents.train_ppo_jobshop as trainer


trainer.SIMPLE = True
trainer.MODEL_NAME = "simple_ppo"
trainer.LOG_NAME = "simple_ppo"
trainer.TOTAL_TIMESTEPS = 200_000


if __name__ == "__main__":
    trainer.train()
