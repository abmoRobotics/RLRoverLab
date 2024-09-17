from typing import Optional

# from omni.replicator.isaac.scripts.writers.pytorch_writer import PytorchWriter
import carb
import numpy as np
import torch
import warp as wp
from omni.isaac.core.utils import prims

# ENV
from ..rover_env_cfg import RoverEnvCfg
from .rover_env import RoverEnv

# from omni.replicator.isaac.scripts.writers.pytorch_listener import PytorchListener


class RoverEnvCamera(RoverEnv):
    """ Rover environment for collecting camera data for Learning By Cheating.

    Note:
        This is a placeholder class for the rover environment. That is, this class is not fully implemented yet.
    """

    def __init__(self, cfg: RoverEnvCfg, **kwargs):

        # Set up replicator
        import omni.replicator.core as rep
        self.rep = rep
        self.PytorchWriter = PytorchWriterRover
        self.PytorchListener = PytorchListenerRover

        super().__init__(cfg, **kwargs)
        self._post_process_cfg()

    def _post_process_cfg(self):

        # Create camera. only need to do this once, since cloner will clone it for each env
        # Create render products for each env
        self.render_products = []
        for i in range(self.num_envs):
            prims.create_prim(
                prim_path=f"/World/envs/env_{i}/Robot/Body/Camera",
                prim_type="Camera",
                attributes={
                    # "focusDistance": 1,
                    "focalLength": 2.12,
                    # "fStop": 1.8,
                    "horizontalAperture": 6.055,
                    "verticalAperture": 2.968879962,
                    "clippingRange": (0.01, 1000000),
                    "clippingPlanes": np.array([1.0, 0.0, 1.0, 1.0]),
                },
                translation=(-0.151, 0, 0.73428),
                orientation=(0.64086, 0.29884, -0.29884, -0.64086),
            )
            render_product = self.rep.create.render_product(
                f"/World/envs/env_{i}/Robot/Body/Camera", resolution=(160, 90))
            self.render_products.append(render_product)

        # # Initialize pytorch writer for vectorized collection
        self.pytorch_listener = self.PytorchListener()
        self.pytorch_writer = self.rep.WriterRegistry.get("PytorchWriterRover")
        self.pytorch_writer.initialize(listener=self.pytorch_listener, device="cuda:0")
        self.pytorch_writer.attach(self.render_products)
        self.image_index_test = 0

    def step(self, actions: torch.Tensor):
        obs_dict, reward, terminated, truncated, info = super().step(actions)

        self.extras["rgb"] = self.pytorch_listener.get_rgb_data()
        self.extras["depth"] = self.pytorch_listener.get_depth_data()
        return obs_dict, reward, terminated, truncated, info


try:
    from omni.replicator.core import AnnotatorRegistry, Writer, WriterRegistry

    # Define a listener that will listen to the render products.
    class PytorchListenerRover:

        def __init__(self):
            self.data = {}

        def write_data(self, data: dict) -> None:
            self.data.update(data)

        def get_rgb_data(self) -> Optional[torch.Tensor]:
            if "pytorch_rgb" in self.data:  # noqa R505
                images = self.data["pytorch_rgb"]
                images = images[..., :3]
                images = images.permute(0, 1, 2, 3)
                return images
            else:
                return None

        def get_depth_data(self) -> Optional[torch.Tensor]:
            if "pytorch_depth" in self.data:  # noqa R505
                images = self.data["pytorch_depth"]
                images = images.permute(0, 1, 2)
                return images
            else:
                return None
    # Define a writer that will write the data to the listener.

    class PytorchWriterRover(Writer):
        def __init__(self, listener: PytorchListenerRover, device: str = "cuda"):
            self._frame_id = 0
            self.listener = listener
            self.device = device
            annotators = ["LdrColor", "distance_to_camera"]
            self.annotators = [AnnotatorRegistry.get_annotator(
                annotator, device="cuda", do_array_copy=False) for annotator in annotators]

        def write(self, data: dict) -> None:
            pytorch_rgb = self._convert_to_pytorch(data, "LdrColor").to(self.device)
            pytorch_depth = self._convert_to_pytorch(data, "distance_to_camera").to(self.device)
            self.listener.write_data(
                {"pytorch_rgb": pytorch_rgb, "pytorch_depth": pytorch_depth, "device": self.device})
            self._frame_id += 1

        @carb.profiler.profile
        def _convert_to_pytorch(self, data: dict, key: str) -> torch.Tensor:
            if data is None:
                raise Exception("Data is Null")

            data_tensors = []
            for annotator in data.keys():
                if annotator.startswith(f'{key}'):
                    data_tensors.append(wp.to_torch(data[annotator]).unsqueeze(0))

            # Move all tensors to the same device for concatenation
            device = "cuda:0" if self.device == "cuda" else self.device
            data_tensors = [t.to(device) for t in data_tensors]

            data_tensor = torch.cat(data_tensors, dim=0)
            return data_tensor  # noqa R504

    WriterRegistry.register(PytorchWriterRover)


except Exception as e:
    print(f'Exception: {e}')
    print('PytorchWriterExtended not registered')
    pass
