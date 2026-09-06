from setuptools import find_packages, setup  # noqa: F401

# Packages needed regardless of environment
INSTALL_REQUIRES = [
    "prettytable==3.3.0",
    "pymeshlab",
    "open3d",
    "gdown",
    "termcolor",
    "hidapi",
    "wandb",
    "opencv-python",
    "skrl>=2.1.0",
    "tqdm",
]
setup(
    name="omni.abmoRobotics.RLRoverLab",
    author="Anton Bjørndahl Mortensen",
    maintainer="Anton Bjørndahl Mortensen",
    maintainer_email="abmoRobotics@gmail.com",
    license="BSD-3-Clause",
    include_package_data=True,
    python_requires=">=3.12,<3.13",
    install_requires=INSTALL_REQUIRES,
    packages=find_packages(include=["rover_envs", "rover_envs.*"]),
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.12",
    ],
    zip_safe=False,
)
