import argparse

def get_parser(mode):
    if mode == 'train':
        # add argparse arguments
        parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
        parser.add_argument(
            "--video", action="store_true", default=True, help="Record videos during training."
        )
        parser.add_argument(
            "--video_length",
            type=int,
            default=200,
            help="Length of the recorded video (in steps).",
        )
        parser.add_argument(
            "--video_interval",
            type=int,
            default=2000,
            help="Interval between video recordings (in steps).",
        )
        parser.add_argument(
            "--num_envs", type=int, default=None, help="Number of environments to simulate."
        )
        parser.add_argument("--task", type=str, default=None, help="Name of the task.")
        parser.add_argument(
            "--seed", type=int, default=None, help="Seed used for the environment"
        )
        parser.add_argument(
            "--distributed",
            action="store_true",
            default=False,
            help="Run training with multiple GPUs or nodes.",
        )
        parser.add_argument(
            "--max_iterations", type=int, default=None, help="RL Policy training iterations."
        )
        parser.add_argument(
            "--ml_framework",
            type=str,
            default="torch",
            choices=["torch", "jax", "jax-numpy"],
            help="The ML framework used for training the skrl agent.",
        )
        parser.add_argument(
            "--algorithm",
            type=str,
            default="PPO",
            choices=["PPO", "IPPO", "MAPPO"],
            help="The RL algorithm used for training the skrl agent.",
        )
    elif mode == 'play':
        # add argparse arguments
        parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
        parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
        parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
        parser.add_argument(
            "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
        )
        parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
        parser.add_argument("--task", type=str, default=None, help="Name of the task.")
        parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
        parser.add_argument(
            "--ml_framework",
            type=str,
            default="torch",
            choices=["torch", "jax", "jax-numpy"],
            help="The ML framework used for training the skrl agent.",
        )
        parser.add_argument(
            "--algorithm",
            type=str,
            default="PPO",
            choices=["PPO", "IPPO", "MAPPO"],
            help="The RL algorithm used for training the skrl agent.",
        )
    else:
        raise ValueError
    return parser