"""Train a PPO policy for task-driven soft-gripper co-design.

Samples egg and cylinder grasping tasks and optimizes morphology choices
using the GripperEnv environment. Checkpoints and logs are written locally."""

# A reinforcement learning agent (PPO) was trained on a task distribution consisting of grasping an egg and a cylinder. Each episode corresponded to a single simulated grasp attempt with randomized object type and weight. The agent observed compact performance features and task properties, learning a policy that maps object-specific conditions to optimal gripper design parameters.
import os
import datetime
from pathlib import Path

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

THIS_DIR = Path(__file__).resolve().parent   # .../Gripper/RL
MODELS_DIR = THIS_DIR / "models"
RUNS_DIR = THIS_DIR / "runs"

import multiprocessing as mp
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

import wandb
from wandb.integration.sb3 import WandbCallback

from gripper_rl_env import GripperEnv


class ForceSaveCallback(BaseCallback):
    """
    Auto-save the PPO model at regular intervals (every save_freq steps).
    Vectorized environment compatible (>= check, unaffected by n_envs step jumps)
    Exception-safe (automatically create directory before saving)
    Clean logging: only print path and time when saving
    """
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "manual", verbose: int = 1):
        super().__init__(verbose)
        self.save_freq = int(save_freq)
        self.save_path = save_path
        self.name_prefix = name_prefix
        self._last_save_at = 0

    def _on_training_start(self) -> None:
        os.makedirs(self.save_path, exist_ok=True)
        # Initial save at step 0
        fname = os.path.join(self.save_path, f"{self.name_prefix}_step_{self.num_timesteps}.zip")
        self.model.save(fname)
        if self.verbose:
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 💾 Initial model save → {fname}")

    def _on_step(self) -> bool:
        # Trigger every save_freq steps
        if (self.num_timesteps - self._last_save_at) >= self.save_freq:
            self._last_save_at = self.num_timesteps
            os.makedirs(self.save_path, exist_ok=True)
            fname = os.path.join(self.save_path, f"{self.name_prefix}_step_{self.num_timesteps}.zip")
            try:
                self.model.save(fname)
                if self.verbose:
                    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 💾 Model save @ step {self.num_timesteps} → {fname}")
            except Exception as e:
                print(f"⚠️ [ForceSaveCallback] Save failed: {e}")
        return True


# 🧠 SEED LOGGING FUNCTION
def log_seed_hierarchy(global_seed, num_envs):
    env_seeds = [global_seed + i for i in range(num_envs)]
    print("\n==================== SEED INFO ====================")
    print(f"[Seed Info] PPO seed               = {global_seed}")
    print(f"[Seed Info] VecEnv base seed       = {global_seed}")
    print(f"[Seed Info] Individual Env seeds   = {env_seeds}")
    print("===================================================\n")
    return env_seeds

# ENV FACTORY
def make_env(seed, rank, mode="compact", object_types=None, verbose=False):
    """
    Factory function for creating task-driven GripperEnv instances.
    Each environment samples an object type and weight number on reset.
    """
    def _init():
        env = GripperEnv(
            mode=mode,
            object_types=object_types,
            seed=seed + rank,
            verbose=verbose
        )
        return env
    return _init


# TRAIN FUNCTION
def train_gripper_env(
    mode="compact",
    object_types=None,
    total_timesteps=1e4,
    num_cpu=4,
    seed=42,
    project_name="softgripper-rl"
):
    """
    Train PPO on GripperEnv with full seed logging and reproducibility.
    """
    env_seeds = log_seed_hierarchy(seed, num_cpu)
    
    """
    Train PPO on GripperEnv under either 'compact' (fast) or 'full' (rich) observation modes.

    Parameters
    ----------
    mode : str
        'compact' or 'full', determines observation and policy type.
    object_types : list[str]
        List of object types defining the task distribution.
    """
    # Default task distribution
    if object_types is None:
        object_types = ["egg", "cylinder"]

    # Config for tracking
    config = {
        # Environment
        "env_mode": mode,
        "object_types": object_types,
        "n_envs": num_cpu,
        "seed": seed,
        
        # PPO policy type
        "policy": "MlpPolicy" if mode == "compact" else "MultiInputPolicy",

        # PPO hyperparameters (explicit)
        "learning_rate": 1e-4,      # Lower because task randomization causes higher variance
        "gamma": 0.9,               # Lower gamma since each episode is one step
        "n_steps": 8, #8,
        "batch_size": 32, #32,
        "clip_range": 0.3,        # Slightly larger for more exploration    
        "ent_coef": 0.03,           # 🔥 increase exploration
        "gae_lambda": 0.9,           # Corresponding to gamma adjustment
        "vf_coef": 0.3,               # Slightly weaker to let actor dominate

        "max_grad_norm": 0.5,         # Prevent gradient explosion
        "target_kl": 0.05,            # Tolerate larger update magnitude
        "clip_range_vf": None,        # Default
        "normalize_advantage": True,  # Smoother gradient signal

        # Training
        "total_timesteps": int(total_timesteps),
        
    }

    run = wandb.init(
        project=project_name,
        config=config,
        sync_tensorboard=True,
        save_code=True,
    )

    # Create vectorized environment
    vec_env = make_vec_env(
        make_env(seed=seed, rank=0, mode=mode, object_types=object_types, verbose=True),
        n_envs=num_cpu,
        seed=seed,
        vec_env_cls=DummyVecEnv,
    )

    # Initialize PPO
    model = PPO(
        config["policy"],
        vec_env,
        verbose=2,
        learning_rate=config["learning_rate"],
        gamma=config["gamma"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
        gae_lambda=config["gae_lambda"],
        max_grad_norm=config["max_grad_norm"],
        vf_coef=config["vf_coef"],
        target_kl=config["target_kl"],
        normalize_advantage=config["normalize_advantage"],
        tensorboard_log=str(RUNS_DIR / run.id),
        seed=seed,
    )


    # Training Loop
    try:
        force_cb = ForceSaveCallback(
            save_freq=960,                            
            save_path=str(MODELS_DIR / run.id),
            name_prefix="manual",
            verbose=1,
        )
        
        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=[force_cb, WandbCallback(
                gradient_save_freq=8,
                model_save_path=str(MODELS_DIR / run.id),
                verbose=2,
            )],
        )


    except KeyboardInterrupt:
        print("⚠️ Training interrupted! Saving model...")
        model.save(str(MODELS_DIR / run.id / "interrupted_model.zip"))
        wandb.save(str(MODELS_DIR / run.id / "interrupted_model.zip"))

    finally:
        run.finish()
        final_path = MODELS_DIR / run.id / "final_model.zip"
        model.save(str(final_path))
        wandb.save(str(final_path))
        print(f"✅ Training completed. Model saved for run: {run.id}")


# ENTRY POINT
if __name__ == "__main__":
    mp.set_start_method("spawn")

    # Example 1: Fast RL with compact observation
    train_gripper_env(
        mode="compact",
        object_types=["egg", "cylinder"],
        total_timesteps=2e4,
        num_cpu=4,
        seed=42,
    )

