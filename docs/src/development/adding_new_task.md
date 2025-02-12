## Adding a New Task

To incorporate a new task into your project, follow the steps outlined below. This guide ensures that your new task is properly set up and integrated within the existing project structure.

### Step 1: Create a New Environment Folder

- Within the `rover_envs/envs` directory, create a new folder named after your task (`TASK_FOLDER`). This folder will house all the necessary configuration files for your new task.

### Step 2: Create the Task Configuration File

- Inside `TASK_FOLDER`, create a configuration file named `TASK_env_cfg.py`, substituting `TASK` with the name of your task. This file will define the task's configuration.

### Step 3: Define the MDPs

- In `TASK_env_cfg.py`, you'll define the configurations for actions, observations, terminations, commands, and, optionally, randomizations that make up your task's Markov Decision Process (MDP).

  You can refer to the [Navigation Task example](https://github.com/abmoRobotics/isaac_rover_orbit/blob/main/rover_envs/envs/navigation/rover_env_cfg.py) for guidance on how to structure this file.

### Step 4: Set Up the Robot Folder

- Within `rover_envs/envs/TASK_FOLDER`, create a new folder named `robots/ROBOT_NAME`, replacing `ROBOT_NAME` with the name of the robot used in the task.

  In this folder, create two files: `__init__.py` and `env_cfg.py`.

### Step 5: Configure `env_cfg.py`

- The `env_cfg.py` file customizes `TASK_env_cfg.py` for a specific robot. At a minimum, it should contain the following Python code:

```python
from rover_envs.assets.robots.YOUR_ROBOT import YOUR_ROBOT_CFG
from rover_envs.envs.YOUR_TASK.TASK_env_cfg.py import TaskEnvCfg

@configclass
class TaskEnvCfg(TaskEnvCfg):

    def __post_init__(self):
        super().__post_init__()

        # Define robot
        self.scene.robot = YOUR_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
```
Make sure to replace YOUR_ROBOT and YOUR_TASK with the appropriate robot and task names

### Step 6: Configure `__init__.py`

This file registers the environment with the OPENAI gym library. Include at least the following code.
```python
import os
import gymnasium as gym
from . import env_cfg

gym.register(
    id="TASK_NAME-v0",
    entry_point='omni.isaac.orbit.envs:RLTaskEnv',
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg.TaskEnvCfg,
        "best_model_path": f"{os.path.dirname(__file__)}/policies/best_agent.pt", # This is optional
    }
)

```

### Step 7: Running the Task

With everything set up, you can now run the task as follows:

```python
# Run training policy
cd examples/02_train
python train.py --task="TASK_NAME-v0" --num_envs=128
```
