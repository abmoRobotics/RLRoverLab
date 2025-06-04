from isaaclab.envs.mdp.observations import image_features
from isaaclab.envs.mdp.observations import ObservationTermCfg
from isaaclab.envs.manager_based_env import ManagerBasedEnv
import torch
import os
import numpy as np
from rover_envs.utils.model_downloader import ensure_model_available


class extended_image_features(image_features):
    """
    Extended image features observation term that includes additional parameters.
    Supports Cosmos tokenizer encoder models in addition to the default models.
    """
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        # List of Cosmos tokenizer models
        self.default_cosmos_tokenizer_models = [
            "Cosmos-0.1-Tokenizer-CI8x8",
            "Cosmos-0.1-Tokenizer-CI16x16",
            "Cosmos-0.1-Tokenizer-DI8x8",
            "Cosmos-0.1-Tokenizer-DI16x16",
        ]
        
        # Extract parameters from the configuration
        self.model_zoo_cfg: dict = cfg.params.get("model_zoo_cfg")  # type: ignore
        self.model_name: str = cfg.params.get("model_name", "resnet18")  # type: ignore
        self.model_device: str = cfg.params.get("model_device", env.device)  # type: ignore

        # Check if it's a Cosmos model and handle accordingly
        if self.model_name in self.default_cosmos_tokenizer_models:
            # Initialize base class without calling super().__init__ to avoid duplicate initialization
            from isaaclab.envs.mdp.observations import ManagerTermBase
            ManagerTermBase.__init__(self, cfg, env)
            
            # Prepare Cosmos model configuration
            model_config = self._prepare_cosmos_tokenizer_model(self.model_name, self.model_device)
            
            # Retrieve the model and inference functions
            self._model = model_config["model"]()
            self._reset_fn = model_config.get("reset")
            self._inference_fn = model_config["inference"]
        else:
            # Initialize the parent class for non-Cosmos models
            super().__init__(cfg, env)

    def _prepare_cosmos_tokenizer_model(self, model_name: str, model_device: str) -> dict:
        """Prepare the Cosmos tokenizer encoder model for inference.

        Args:
            model_name: The name of the Cosmos tokenizer model to prepare.
            model_device: The device to store and infer the model on.

        Returns:
            A dictionary containing the model and inference functions.
        """
        def _load_model() -> torch.nn.Module:
            """Load the Cosmos tokenizer encoder model."""
            try:
                # Import cosmos_tokenizer
                from cosmos_tokenizer.image_lib import ImageTokenizer
                
                # Ensure model is available, auto-download if needed
                model_dir = ensure_model_available(model_name, auto_download=True)
                if model_dir is None:
                    raise FileNotFoundError(
                        f"Failed to download or find Cosmos tokenizer model '{model_name}'. "
                        "Please check your internet connection and Hugging Face token."
                    )
                
                # Set up model paths from cache directory
                encoder_ckpt = str(model_dir / "encoder.jit")
                decoder_ckpt = str(model_dir / "decoder.jit")
                
                # Create tokenizer instance - we only need encoder for feature extraction
                tokenizer = ImageTokenizer(
                    checkpoint_enc=encoder_ckpt,
                    checkpoint_dec=decoder_ckpt,  # Still needed for initialization
                    device=model_device,
                    dtype="bfloat16" if "cuda" in model_device else "float32",
                )
                
                # Return the full tokenizer for access to encode method
                return tokenizer.eval()
                
            except ImportError:
                raise ImportError(
                    "cosmos_tokenizer package not found. Please install it to use Cosmos tokenizer models."
                )

        def _inference(model, images: torch.Tensor) -> torch.Tensor:
            """Inference the Cosmos tokenizer encoder model.

            Args:
                model: The Cosmos tokenizer model (ImageTokenizer instance).
                images: The preprocessed image tensor. Shape is (num_envs, height, width, channel).

            Returns:
                The extracted features tensor. Shape is (num_envs, feature_dim).
            """
            # Move the image to the model device
            image_proc = images.to(model_device)
            
            # Convert from (B, H, W, C) to (B, C, H, W) and normalize to [-1, 1]
            # Cosmos tokenizer expects images in range [-1, 1]
            image_proc = image_proc.permute(0, 3, 1, 2).float()
            image_proc = (image_proc / 255.0) * 2.0 - 1.0  # [0,255] -> [0,1] -> [-1,1]
            
            # Forward through encoder to get latent tokens/features
            with torch.no_grad():
                # Use the encode method which returns a tuple
                encoded_output = model.encode(image_proc)
                
                # Handle different return types from encode method
                if isinstance(encoded_output, tuple):
                    # For CI models: returns (latent_embedding,)
                    # For DI models: returns (indices, discrete_code)
                    features = encoded_output[0]  # Take the first element
                else:
                    # If it returns a single tensor
                    features = encoded_output
                
                # Flatten the spatial dimensions to get feature vector per image
                # Shape: (batch, channels, h, w) -> (batch, channels * h * w)
                if features.dim() == 4:
                    features = features.flatten(start_dim=1)
                elif features.dim() == 3:
                    # If already flattened to (batch, seq_len, features)
                    features = features.flatten(start_dim=1)
                    
            return features

        # return the model and inference functions
        return {"model": _load_model, "inference": _inference}