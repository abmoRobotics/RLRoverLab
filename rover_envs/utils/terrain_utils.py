"""Utility functions for terrain configuration in training/evaluation scripts."""

import os
import random
import shutil
import tempfile
from typing import Optional, Tuple


def handle_terrain_config(
    terrain_arg: Optional[str],
    terrain_seed: Optional[int] = None,
    keep_terrain: bool = False,
    terrain_width: int = 200,
    terrain_length: int = 200,
    num_rocks: int = 1800,  # Match mars terrain (~480k rock vertices)
) -> Tuple[Optional[str], Optional[str]]:
    """
    Handle terrain configuration, including random terrain generation.
    
    Args:
        terrain_arg: Terrain argument from CLI ('mars', 'debug', 'random', or None)
        terrain_seed: Seed for random terrain generation (None = random seed)
        keep_terrain: Whether to keep the generated terrain after use
        terrain_width: Width of generated terrain in meters
        terrain_length: Length of generated terrain in meters
        num_rocks: Number of rocks for generated terrain
        
    Returns:
        Tuple of (terrain_name, terrain_path_to_cleanup)
        - terrain_name: Name to pass to set_terrain(), or None if using default
        - terrain_path_to_cleanup: Path to delete after use (if not keeping), or None
    """
    if terrain_arg is None:
        return None, None
    
    if terrain_arg.lower() != "random":
        # Regular terrain name
        return terrain_arg, None
    
    # Generate random terrain
    return generate_random_terrain(
        seed=terrain_seed,
        keep=keep_terrain,
        width=terrain_width,
        length=terrain_length,
        num_rocks=num_rocks,
    )


def generate_random_terrain(
    seed: Optional[int] = None,
    keep: bool = False,
    width: int = 200,
    length: int = 200,
    num_rocks: int = 1800,  # Match mars terrain (~480k rock vertices)
) -> Tuple[str, Optional[str]]:
    """
    Generate a random terrain.
    
    Args:
        seed: Random seed (None = generate random seed)
        keep: Whether to keep the terrain after use
        width: Terrain width in meters
        length: Terrain length in meters  
        num_rocks: Number of rocks to place
        
    Returns:
        Tuple of (terrain_name, cleanup_path)
    """
    # Import here to avoid circular imports and allow lazy loading
    from rover_envs.assets.terrains.generator.terrain_generator import (
        TerrainGeneratorConfig,
        RockConfig,
        get_default_layers,
        generate_and_save_terrain,
    )
    from rover_envs.assets.terrains import register_terrain_from_folder
    
    # Generate seed if not provided
    if seed is None:
        seed = random.randint(1, 999999)
    
    print(f"\n{'='*60}")
    print(f"Generating random terrain (seed: {seed})")
    print(f"{'='*60}")
    
    # Determine output location
    if keep:
        # Save to permanent location in generated folder
        from rover_envs.assets.terrains.generator.terrain_generator import get_generator_assets_path
        assets_path = get_generator_assets_path()
        terrains_base = os.path.dirname(assets_path)
        generated_dir = os.path.join(terrains_base, "generated")
        os.makedirs(generated_dir, exist_ok=True)
        
        terrain_name = f"random_{seed}"
        output_dir = os.path.join(generated_dir, terrain_name)
        cleanup_path = None  # Don't cleanup
        
        # Check if already exists
        if os.path.exists(output_dir):
            print(f"  Terrain '{terrain_name}' already exists, reusing...")
            register_terrain_from_folder(
                name=terrain_name,
                folder=f"generated/{terrain_name}",
                description=f"Random terrain (seed={seed})",
            )
            return terrain_name, None
    else:
        # Save to temporary location
        terrain_name = f"random_{seed}"
        temp_base = tempfile.mkdtemp(prefix="rover_terrain_")
        output_dir = os.path.join(temp_base, terrain_name)
        cleanup_path = temp_base  # Cleanup the temp directory
    
    # Create configuration
    config = TerrainGeneratorConfig(
        name=terrain_name,
        width=width,
        length=length,
        horizontal_scale=0.05,
        vertical_scale=0.05,
        target_vertices=200000, 
        layers=get_default_layers(seed),
        rock_config=RockConfig(
            num_rocks=num_rocks,
            scale_range=(0.05, 0.25),
            embed_percentage=0.25,
        ),
        seed=seed,
    )
    
    # Generate terrain
    generate_and_save_terrain(config, output_dir)
    
    # Register the terrain
    # For temporary terrains, we need to register with absolute path
    if keep:
        register_terrain_from_folder(
            name=terrain_name,
            folder=f"generated/{terrain_name}",
            description=f"Random terrain (seed={seed})",
        )
    else:
        # Register with absolute path for temporary terrain
        from rover_envs.assets.terrains import register_terrain, TerrainFiles
        register_terrain(
            name=terrain_name,
            files=TerrainFiles(
                terrain_only=os.path.join(output_dir, "terrain_only.usd"),
                terrain_merged=os.path.join(output_dir, "terrain_merged.usd"),
                rocks_merged=os.path.join(output_dir, "rocks_merged.usd"),
            ),
            description=f"Temporary random terrain (seed={seed})",
        )
    
    print(f"{'='*60}")
    print(f"Random terrain ready: {terrain_name}")
    if keep:
        print(f"  Saved to: {output_dir}")
    else:
        print(f"  Temporary (will be deleted after use)")
    print(f"{'='*60}\n")
    
    return terrain_name, cleanup_path


def cleanup_terrain(cleanup_path: Optional[str]) -> None:
    """
    Clean up temporary terrain files.
    
    Args:
        cleanup_path: Path to delete, or None to skip
    """
    if cleanup_path and os.path.exists(cleanup_path):
        print(f"\nCleaning up temporary terrain: {cleanup_path}")
        shutil.rmtree(cleanup_path, ignore_errors=True)
