"""
Helper script to create scene mapping between generated images and metadata
This helps match your generated images to the correct ground truth
"""

import os
import json
import glob


def create_scene_mapping_file(
    images_dir: str,
    metadata_file: str,
    output_file: str = "scene_mapping.json"
):
    """
    Create a mapping file between generated image directories and metadata entries
    
    This is interactive - you'll specify which metadata entry each scene corresponds to
    
    Args:
        images_dir: Directory with generated images (e.g., gnn_run/images-30000/images-30000)
        metadata_file: Path to valdata.jsonl or metadata.jsonl
        output_file: Where to save the mapping
    """
    
    # Load metadata
    metadata = {}
    with open(metadata_file, 'r') as f:
        for i, line in enumerate(f):
            if line.strip():
                data = json.loads(line)
                metadata[i] = {
                    'file_name': data['file_name'],
                    'caption': data['caption'],
                    'objects': data['objects'],
                    'scene_graph_raw': data['relations']
                }
    
    # Get all scene directories
    scene_dirs = []
    for root, dirs, files in os.walk(images_dir):
        # Check if this directory contains images
        images = [f for f in files if f.endswith(('.png', '.jpg', '.jpeg'))]
        if images and root != images_dir:
            rel_path = os.path.relpath(root, images_dir)
            scene_dirs.append(rel_path)
    
    scene_dirs = sorted(list(set(scene_dirs)))
    
    print(f"\nFound {len(scene_dirs)} scene directories")
    print(f"Loaded {len(metadata)} metadata entries\n")
    
    print("="*70)
    print("SCENE MAPPING CREATION")
    print("="*70)
    print("\nFor each scene directory, you'll specify which metadata entry it corresponds to.")
    print("You can also use 'auto' mode to try automatic matching by caption similarity.\n")
    
    mapping = {}
    
    for scene_dir in scene_dirs:
        scene_path = os.path.join(images_dir, scene_dir)
        
        # Count images in this scene
        images = glob.glob(os.path.join(scene_path, '*.*'))
        image_files = [f for f in images if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"\nScene: {scene_dir}")
        print(f"  Images: {len(image_files)}")
        print(f"  Sample: {os.path.basename(image_files[0]) if image_files else 'none'}")
        
        # Show metadata options
        print(f"\nAvailable metadata entries:")
        for idx, meta in metadata.items():
            print(f"  [{idx}] {meta['caption']}")
            print(f"      File: {meta['file_name']}")
        
        # Get user input
        while True:
            choice = input(f"\nWhich metadata index for '{scene_dir}'? (0-{len(metadata)-1}, 's' to skip): ").strip()
            
            if choice.lower() == 's':
                print(f"Skipping {scene_dir}")
                break
            
            try:
                idx = int(choice)
                if idx in metadata:
                    mapping[scene_dir] = idx
                    print(f"✓ Mapped '{scene_dir}' to metadata index {idx}")
                    print(f"  Caption: {metadata[idx]['caption']}")
                    break
                else:
                    print(f"Invalid index. Choose from 0-{len(metadata)-1}")
            except ValueError:
                print("Invalid input. Enter a number or 's' to skip")
    
    # Save mapping
    with open(output_file, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"\n✓ Scene mapping saved to: {output_file}")
    print(f"  Mapped scenes: {len(mapping)}/{len(scene_dirs)}")
    
    return mapping


def auto_create_scene_mapping(
    images_dir: str,
    metadata_file: str,
    output_file: str = "scene_mapping.json"
):
    """
    Automatically create scene mapping by assuming scene directories match metadata order
    
    This assumes your scenes are organized in the same order as valdata.jsonl
    """
    
    # Load metadata
    metadata = {}
    with open(metadata_file, 'r') as f:
        for i, line in enumerate(f):
            if line.strip():
                data = json.loads(line)
                metadata[i] = data
    
    # Get scene directories in sorted order
    scene_dirs = []
    for root, dirs, files in os.walk(images_dir):
        images = [f for f in files if f.endswith(('.png', '.jpg', '.jpeg'))]
        if images and root != images_dir:
            rel_path = os.path.relpath(root, images_dir)
            scene_dirs.append(rel_path)
    
    scene_dirs = sorted(list(set(scene_dirs)))
    
    print(f"\nAuto-mapping {len(scene_dirs)} scenes to {len(metadata)} metadata entries")
    
    # Create 1:1 mapping assuming order matches
    mapping = {}
    for i, scene_dir in enumerate(scene_dirs):
        if i < len(metadata):
            mapping[scene_dir] = i
            print(f"✓ {scene_dir} → [{i}] {metadata[i]['caption'][:50]}...")
        else:
            print(f"⚠ {scene_dir} → No metadata available")
    
    # Save mapping
    with open(output_file, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"\n✓ Scene mapping saved to: {output_file}")
    
    return mapping


def display_mapping(mapping_file: str, metadata_file: str):
    """Display the current scene mapping"""
    
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)
    
    metadata = {}
    with open(metadata_file, 'r') as f:
        for i, line in enumerate(f):
            if line.strip():
                data = json.loads(line)
                metadata[i] = data
    
    print("\n" + "="*70)
    print("SCENE MAPPING")
    print("="*70 + "\n")
    
    for scene_dir, meta_idx in sorted(mapping.items()):
        if meta_idx in metadata:
            meta = metadata[meta_idx]
            print(f"Scene: {scene_dir}")
            print(f"  → [{meta_idx}] {meta['caption']}")
            print(f"  Objects: {', '.join(meta['objects'])}")
            print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create scene mapping for evaluation")
    parser.add_argument("images_dir", help="Directory with generated images")
    parser.add_argument("metadata_file", help="Path to valdata.jsonl or metadata.jsonl")
    parser.add_argument("--output", default="scene_mapping.json", help="Output mapping file")
    parser.add_argument("--auto", action="store_true", help="Auto-create mapping by order")
    parser.add_argument("--display", action="store_true", help="Display existing mapping")
    
    args = parser.parse_args()
    
    if args.display:
        if os.path.exists(args.output):
            display_mapping(args.output, args.metadata_file)
        else:
            print(f"Mapping file not found: {args.output}")
    elif args.auto:
        auto_create_scene_mapping(args.images_dir, args.metadata_file, args.output)
    else:
        create_scene_mapping_file(args.images_dir, args.metadata_file, args.output)