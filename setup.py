import itertools
import os

from setuptools import find_packages, setup  # noqa: F401

# Check if this is a native install (default: true)
# Set SKIP_ISAAC_SIM_INSTALL=true in Docker to skip Isaac Sim/Lab packages
SKIP_ISAAC_SIM = os.getenv("SKIP_ISAAC_SIM_INSTALL", "false").lower() == "true"

# Packages needed regardless of environment
INSTALL_REQUIRES = [
    "cosmos-tokenizer@git+https://github.com/NVIDIA/Cosmos-Tokenizer.git",
    "prettytable==3.3.0",
    "pymeshlab",
    "open3d",
    "gdown",
    "termcolor",
    "hidapi",
    "wandb",
    "opencv-python",
    "skrl",
]

# Only install Isaac Sim/Lab packages for native installations
# These are already present in the Isaac Sim Docker base image
if not SKIP_ISAAC_SIM:
    INSTALL_REQUIRES.extend([
        "numpy",
        "torch",
        "torchvision",
        "isaaclab==2.3.0",
        "isaacsim[all,extscache]==5.1.0",
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
    python_requires="==3.11.*",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    packages=["rover_envs"],
    classifiers=["Natural Language :: English", "Programming Language :: Python :: 3.11"],
    zip_safe=False,
)
