"""
Complete Evaluation Runner for Your SG-Adapter Experiments
Adjusted for your JSONL metadata format
"""

import os
import sys
import argparse
import json
from pathlib import Path


def setup_argparse():
    parser = argparse.ArgumentParser(
        description="Evaluate SG-Adapter experiments with JSONL metadata"
    )
    
    parser.add_argument(
        "--repo_dir",
        type=str,
        default=".",
        help="Root directory of the eval repository"
    )
    
    parser.add_argument(
        "--metadata_file",
        type=str,
        default="valdata.jsonl",
        help="Path to metadata JSONL file (valdata.jsonl or metadata.jsonl)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results"
    )
    
    parser.add_argument(
        "--gemini_api_key",
        type=str,
        default=None,
        help="Gemini API key (or set GEMINI_API_KEY env variable)"
    )
    
    parser.add_argument(
        "--rate_limit_delay",
        type=float,
        default=2.0,
        help="Delay between API calls (seconds)"
    )
    
    return parser


def scan_metadata_file(metadata_file: str):
    """Scan and display metadata file info"""
    print("\n" + "="*70)
    print("METADATA FILE INFO")
    print("="*70 + "\n")
    
    if not os.path.exists(metadata_file):
        print(f"✗ File not found: {metadata_file}")
        return None
    
    scenes = {}
    objects = set()
    predicates = set()
    
    with open(metadata_file, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                file_name = data['file_name']
                scenes[file_name] = data
                objects.update(data['objects'])
                predicates.update(rel[1] for rel in data['relations'])
    
    print(f"✓ Metadata file: {metadata_file}")
    print(f"  Total scenes: {len(scenes)}")
    print(f"  Unique objects: {len(objects)}")
    print(f"  Unique predicates: {len(predicates)}")
    print(f"\nPredicates: {', '.join(sorted(predicates))}")
    print(f"\nSample scenes:")
    for i, (file_name, data) in enumerate(list(scenes.items())[:5]):
        print(f"  {file_name}: {data['caption']}")
    
    print("="*70 + "\n")
    
    return scenes


def scan_repo_structure(repo_dir: str):
    """Scan and display repository structure"""
    print("\n" + "="*70)
    print("REPOSITORY STRUCTURE")
    print("="*70 + "\n")
    
    methods_found = []
    
    # Check for gnn_run
    gnn_path = os.path.join(repo_dir, "gnn_run/images-30000/images-30000")
    if os.path.exists(gnn_path):
        # Count images recursively
        import glob
        images = glob.glob(os.path.join(gnn_path, '**', '*.png'), recursive=True) + \
                glob.glob(os.path.join(gnn_path, '**', '*.jpg'), recursive=True)
        # Count scenes (subdirectories)
        scenes = set(os.path.dirname(os.path.relpath(img, gnn_path)) for img in images)
        
        print(f"✓ gnn_run found: {len(scenes)} scenes, {len(images)} images")
        methods_found.append({
            'name': 'gnn_run',
            'images_dir': gnn_path,
            'scenes': len(scenes),
            'images': len(images)
        })
    else:
        print(f"✗ gnn_run not found at {gnn_path}")
    
    # Check for repr_run
    repr_path = os.path.join(repo_dir, "repr_run/images-30000/images-30000")
    if os.path.exists(repr_path):
        import glob
        images = glob.glob(os.path.join(repr_path, '**', '*.png'), recursive=True) + \
                glob.glob(os.path.join(repr_path, '**', '*.jpg'), recursive=True)
        scenes = set(os.path.dirname(os.path.relpath(img, repr_path)) for img in images)
        
        print(f"✓ repr_run found: {len(scenes)} scenes, {len(images)} images")
        methods_found.append({
            'name': 'repr_run',
            'images_dir': repr_path,
            'scenes': len(scenes),
            'images': len(images)
        })
    else:
        print(f"✗ repr_run not found at {repr_path}")
    
    print("="*70 + "\n")
    
    return methods_found


def generate_latex_table(comparison: dict, output_file: str = None):
    """Generate LaTeX table"""
    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{Quantitative Evaluation Results}")
    latex.append("\\label{tab:quantitative_results}")
    latex.append("\\begin{tabular}{lccc}")
    latex.append("\\toprule")
    latex.append("Method & SG-IoU $\\uparrow$ & Entity-IoU $\\uparrow$ & Relation-IoU $\\uparrow$ \\\\")
    latex.append("\\midrule")
    
    for method in sorted(comparison.keys()):
        metrics = comparison[method]
        method_name = method.replace("_", "\\_")
        
        sg_iou = metrics['sg_iou']
        entity_iou = metrics['entity_iou']
        relation_iou = metrics['relation_iou']
        
        latex.append(f"{method_name} & {sg_iou:.3f} & {entity_iou:.3f} & {relation_iou:.3f} \\\\")
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    latex_str = "\n".join(latex)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(latex_str)
        print(f"LaTeX table saved to: {output_file}")
    
    print("\nLaTeX Table:")
    print("-" * 70)
    print(latex_str)
    print("-" * 70)
    
    return latex_str


def generate_markdown_table(comparison: dict, output_file: str = None):
    """Generate Markdown table"""
    md = []
    md.append("# Evaluation Results\n")
    md.append("## Quantitative Metrics\n")
    md.append("| Method | SG-IoU ↑ | Entity-IoU ↑ | Relation-IoU ↑ | Images |")
    md.append("|--------|----------|--------------|----------------|--------|")
    
    for method in sorted(comparison.keys()):
        metrics = comparison[method]
        method_name = method.replace("_", " ").title()
        
        sg_iou = metrics['sg_iou']
        entity_iou = metrics['entity_iou']
        relation_iou = metrics['relation_iou']
        n_images = metrics['n_images']
        
        md.append(f"| {method_name} | {sg_iou:.3f} | {entity_iou:.3f} | {relation_iou:.3f} | {n_images} |")
    
    md_str = "\n".join(md)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(md_str)
        print(f"Markdown table saved to: {output_file}")
    
    return md_str


def main():
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Set API key
    if args.gemini_api_key:
        os.environ["GEMINI_API_KEY"] = args.gemini_api_key
    
    if "GEMINI_API_KEY" not in os.environ:
        print("Error: GEMINI_API_KEY not set!")
        print("Set it via --gemini_api_key argument or GEMINI_API_KEY environment variable")
        print("\nGet your key at: https://makersuite.google.com/app/apikey")
        sys.exit(1)
    
    # Scan metadata file
    metadata_path = os.path.join(args.repo_dir, args.metadata_file)
    scenes = scan_metadata_file(metadata_path)
    
    if scenes is None:
        print(f"Error: Could not load metadata from {metadata_path}")
        sys.exit(1)
    
    # Scan repository structure
    methods_found = scan_repo_structure(args.repo_dir)
    
    if not methods_found:
        print("Error: No method directories found!")
        print("Expected structure:")
        print("  gnn_run/images-30000/images-30000/")
        print("  repr_run/images-30000/images-30000/")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run evaluation
    print("\n" + "="*70)
    print("STARTING EVALUATION")
    print("="*70 + "\n")
    
    # Import here to avoid errors if API key not set
    from sg_adapter_eval import SGAdapterEvaluator
    
    evaluator = SGAdapterEvaluator()
    
    # Prepare methods config
    methods_config = [
        {
            'name': m['name'],
            'images_dir': m['images_dir']
        }
        for m in methods_found
    ]
    
    comparison = evaluator.compare_methods(
        methods_config=methods_config,
        metadata_file=metadata_path,
        output_dir=args.output_dir
    )
    
    # Generate tables
    print("\n" + "="*70)
    print("GENERATING TABLES")
    print("="*70 + "\n")
    
    latex_file = os.path.join(args.output_dir, "results_table.tex")
    generate_latex_table(comparison, latex_file)
    
    md_file = os.path.join(args.output_dir, "results_table.md")
    generate_markdown_table(comparison, md_file)
    
    # Save summary
    summary = {
        "metadata_file": args.metadata_file,
        "methods_evaluated": list(comparison.keys()),
        "num_methods": len(comparison),
        "num_test_scenes": len(scenes),
        "metrics": comparison
    }
    
    summary_file = os.path.join(args.output_dir, "evaluation_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Evaluation complete!")
    print(f"✓ Results saved to: {args.output_dir}")
    print(f"✓ Summary: {summary_file}")
    
    # Print best method
    best_method = max(comparison.items(), key=lambda x: x[1]['sg_iou'])
    print(f"\n🏆 Best Method (by SG-IoU): {best_method[0]} ({best_method[1]['sg_iou']:.3f})")


if __name__ == "__main__":
    main()