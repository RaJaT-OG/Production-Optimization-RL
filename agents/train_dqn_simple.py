
"""
Train the DQN agent on the simple arrival + burst + priority model.

This is the same algorithm as agents/train_dqn_jobshop.py, but every
job has one operation and may run on any machine, which is the input
format used by the dashboard.

    $env:PYTHONIOENCODING = "utf-8"
    venv\\Scripts\\python.exe agents\\train_dqn_simple.py
"""

import sys

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import agents.train_dqn_jobshop as trainer


trainer.SIMPLE = True
trainer.MODEL_NAME = "simple_dqn"
trainer.LOG_NAME = "simple_dqn"
trainer.TOTAL_TIMESTEPS = 300_000


if __name__ == "__main__":
    trainer.train()
