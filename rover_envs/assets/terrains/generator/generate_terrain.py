#!/usr/bin/env python3
"""
Command-line tool for generating procedural terrains.

Usage:
    python generate_terrain.py --name my_terrain --output ./terrains/my_terrain
    python generate_terrain.py --name rocky_terrain --num-rocks 50000 --seed 12345
    python generate_terrain.py --list  # List existing terrains
    
The generated terrain will be automatically registered and can be used with:
    python train.py --terrain my_terrain
"""
import argparse
import os
import sys

# Add the project root to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
sys.path.insert(0, project_root)


def main():
    parser = argparse.ArgumentParser(
        description="Generate procedural terrains for rover environments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Generate a terrain with default settings:
    python generate_terrain.py --name my_terrain

  Generate with custom parameters:
    python generate_terrain.py --name rocky --num-rocks 50000 --width 300 --length 300

  Generate with specific seed for reproducibility:
    python generate_terrain.py --name test_terrain --seed 42

  List available terrains:
    python generate_terrain.py --list
        """
    )
    
    parser.add_argument("--name", type=str, default="generated",
                        help="Name for the terrain (default: generated)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: auto-generated in terrains folder)")
    parser.add_argument("--width", type=int, default=500,
                        help="Terrain width in meters (default: 500)")
    parser.add_argument("--length", type=int, default=500,
                        help="Terrain length in meters (default: 500)")
    parser.add_argument("--num-rocks", type=int, default=25000,
                        help="Number of rocks to place (default: 25000)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--target-vertices", type=int, default=1000000,
                        help="Target vertex count for mesh reduction (default: 1000000)")
    parser.add_argument("--register", action="store_true", default=True,
                        help="Register terrain in terrain_registry.py (default: True)")
    parser.add_argument("--no-register", action="store_true",
                        help="Don't register the terrain")
    parser.add_argument("--list", action="store_true",
                        help="List available terrains and exit")
    
    args = parser.parse_args()
    
    # Handle --list
    if args.list:
        from rover_envs.assets.terrains import list_terrains, get_terrain
        print("\nAvailable terrains:")
        for name in list_terrains():
            terrain = get_terrain(name)
            print(f"  - {name}: {terrain.description}")
        return 0
    
    # Import generator (requires scipy, cv2, pymeshlab, pxr)
    try:
        from rover_envs.assets.terrains.generator import (
            TerrainGeneratorConfig,
            RockConfig,
            get_default_layers,
            generate_and_save_terrain,
        )
    except ImportError as e:
        print(f"Error: Missing required dependencies: {e}")
        print("\nRequired packages: scipy, opencv-python, pymeshlab")
        print("Install with: pip install scipy opencv-python pymeshlab")
        return 1
    
    # Determine output directory
    if args.output:
        output_dir = args.output
    else:
        # Create in the generated folder
        terrains_base = os.path.dirname(script_dir)
        generated_dir = os.path.join(terrains_base, "generated")
        os.makedirs(generated_dir, exist_ok=True)
        output_dir = os.path.join(generated_dir, args.name)
    
    # Check if output exists
    if os.path.exists(output_dir):
        response = input(f"Directory '{output_dir}' already exists. Overwrite? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return 0
        import shutil
        shutil.rmtree(output_dir)
    
    # Create configuration
    config = TerrainGeneratorConfig(
        name=args.name,
        width=args.width,
        length=args.length,
        horizontal_scale=0.05,
        vertical_scale=0.05,
        target_vertices=args.target_vertices,
        layers=get_default_layers(args.seed),
        rock_config=RockConfig(
            num_rocks=args.num_rocks,
            scale_range=(0.05, 0.25),
            embed_percentage=0.25,
        ),
        seed=args.seed,
    )
    
    # Generate terrain
    print(f"\n{'='*60}")
    print(f"Generating terrain: {args.name}")
    print(f"{'='*60}")
    print(f"  Size: {args.width}x{args.length}m")
    print(f"  Rocks: {args.num_rocks}")
    print(f"  Seed: {args.seed or 'random'}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")
    
    try:
        generate_and_save_terrain(config, output_dir)
    except Exception as e:
        print(f"\nError generating terrain: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Register terrain if requested
    if args.register and not args.no_register:
        print("\nTo use this terrain, add the following to terrain_registry.py:")
        print(f"""
register_terrain_from_folder(
    name="{args.name}",
    folder="generated/{args.name}",
    description="Procedurally generated terrain",
)
""")
        print("Or use it directly with:")
        print(f"  env_cfg.scene.set_terrain('{args.name}')")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
