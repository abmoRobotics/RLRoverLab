import itertools
import os

from setuptools import find_packages, setup  # noqa: F401

# Check if this is a native install (default: true)
# Set NATIVE_INSTALL=false in Docker to skip Isaac Sim installation
SKIP_ISAAC_SIM = os.getenv("SKIP_ISAAC_SIM_INSTALL", "false").lower() == "true"

INSTALL_REQUIRES = []

# Only install Isaac Sim packages for native installations
if not SKIP_ISAAC_SIM:
    INSTALL_REQUIRES.extend([
        "cosmos-tokenizer@git+https://github.com/NVIDIA/Cosmos-Tokenizer.git",
        "numpy",
        "torch==2.7.0",
        "torchvision==0.22.0",
        "prettytable==3.3.0",
        "pymeshlab",
        "open3d",
        "gdown",
        "termcolor",
        # devices
        "hidapi",
        "wandb",
        "opencv-python",
        "skrl",
        "isaaclab==2.2.0",
        "isaacsim[all,extscache]==5.0.0",
    ])
    
EXTRAS_REQUIRE = {
    "rsl_rl": ["rsl_rl@git+https://github.com/leggedrobotics/rsl_rl.git"],
    #"cosmos": ["cosmos-tokenizer@git+https://github.com/NVIDIA/Cosmos-Tokenizer.git"],
}

# cumulation of all extra-requires
EXTRAS_REQUIRE["all"] = list(itertools.chain.from_iterable(EXTRAS_REQUIRE.values()))
setup(
    name="omni.abmoRobotics.RLRoverLab",
    author="Anton Bjørndahl Mortensen",
    maintainer="Anton Bjørndahl Mortensen",
    maintainer_email="abmoRobotics@gmail.com",
    license="BSD-3-Clause",
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    packages=["rover_envs"],
    classifiers=["Natural Language :: English", "Programming Language :: Python :: 3.10"],
    zip_safe=False,
)
