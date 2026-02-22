"""Resume training from ep5000 with environment fix."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.trainer import Trainer, TrainerConfig

config = TrainerConfig(
    n_agents=3,
    n_episodes=5500,  # Run 500 more episodes (5000 + 500)
    checkpoint_freq=250,
    log_freq=25,
)
trainer = Trainer(config)
trainer.train(resume_from='checkpoints/checkpoint_ep5000.pt')
