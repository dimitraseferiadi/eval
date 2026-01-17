"""
Filter evaluation results to keep only the best image per scene
This matches the paper's evaluation methodology
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


def extract_scene_index(image_name):
    """Extract scene index from image filename"""
    # For files like "000.png", "000_001.png", "000_002.png"
    # Extract the first number before underscore or extension
    base_name = Path(image_name).stem
    scene_idx = base_name.split('_')[0]
    return scene_idx


def filter_best_per_scene(results_file, output_file=None, metric='sg_iou'):
    """
    Filter results to keep only the best image per scene
    
    Args:
        results_file: Path to the *_results.json file
        output_file: Optional path to save filtered results
        metric: Metric to use for selecting best image ('sg_iou', 'entity_iou', or 'relation_iou')
    """
    # Load results
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    per_image_results = data['per_image_results']
    
    # Group images by scene
    scenes = defaultdict(list)
    for result in per_image_results:
        scene_idx = extract_scene_index(result['image'])
        scenes[scene_idx].append(result)
    
    print(f"\n{'='*70}")
    print(f"Processing: {results_file}")
    print(f"{'='*70}")
    print(f"Total images: {len(per_image_results)}")
    print(f"Unique scenes: {len(scenes)}")
    print(f"Selection metric: {metric}")
    
    # Select best image per scene
    best_results = []
    for scene_idx in sorted(scenes.keys()):
        scene_images = scenes[scene_idx]
        
        # Find best image by the specified metric
        best_image = max(scene_images, key=lambda x: x[metric])
        best_results.append(best_image)
        
        # Print details
        print(f"\nScene {scene_idx} ({best_image['caption']}):")
        print(f"  Images: {len(scene_images)}")
        print(f"  Best: {best_image['image']} ({metric}={best_image[metric]:.3f})")
        
        # Show all images for this scene
        for img in sorted(scene_images, key=lambda x: x[metric], reverse=True):
            indicator = "★" if img['image'] == best_image['image'] else " "
            print(f"    {indicator} {img['image']}: SG-IoU={img['sg_iou']:.3f}, "
                  f"Entity-IoU={img['entity_iou']:.3f}, Relation-IoU={img['relation_iou']:.3f}")
    
    # Compute new average metrics
    n_scenes = len(best_results)
    avg_metrics = {
        'sg_iou': sum(r['sg_iou'] for r in best_results) / n_scenes,
        'entity_iou': sum(r['entity_iou'] for r in best_results) / n_scenes,
        'relation_iou': sum(r['relation_iou'] for r in best_results) / n_scenes,
        'n_images': n_scenes
    }
    
    # Create filtered output
    filtered_data = {
        'selection_method': f'best_{metric}_per_scene',
        'average_metrics': avg_metrics,
        'per_image_results': best_results,
        'original_metrics': data['average_metrics']
    }
    
    # Print comparison
    print(f"\n{'='*70}")
    print("METRICS COMPARISON")
    print(f"{'='*70}")
    print(f"{'Metric':<20} {'All Images':>15} {'Best Per Scene':>15} {'Improvement':>15}")
    print(f"{'-'*70}")
    
    for metric_name in ['sg_iou', 'entity_iou', 'relation_iou']:
        original = data['average_metrics'][metric_name]
        filtered = avg_metrics[metric_name]
        improvement = ((filtered - original) / original * 100) if original > 0 else 0
        
        print(f"{metric_name:<20} {original:>15.3f} {filtered:>15.3f} {improvement:>14.1f}%")
    
    print(f"{'n_images':<20} {data['average_metrics']['n_images']:>15} {avg_metrics['n_images']:>15}")
    print(f"{'='*70}\n")
    
    # Save filtered results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(filtered_data, f, indent=2)
        print(f"✓ Filtered results saved to: {output_file}")
    
    return filtered_data


def compare_methods_best_only(methods_results, output_file=None, metric='sg_iou'):
    """
    Compare multiple methods using only best images per scene
    
    Args:
        methods_results: Dict mapping method names to their results files
        output_file: Optional path to save comparison
        metric: Metric to use for selecting best images
    """
    comparison = {}
    
    print(f"\n{'='*70}")
    print("FILTERING ALL METHODS TO BEST IMAGES ONLY")
    print(f"{'='*70}\n")
    
    for method_name, results_file in methods_results.items():
        filtered = filter_best_per_scene(results_file, metric=metric)
        comparison[method_name] = filtered['average_metrics']
    
    # Print final comparison table
    print(f"\n{'='*70}")
    print("FINAL COMPARISON (BEST IMAGES ONLY)")
    print(f"{'='*70}")
    print(f"{'Method':<20} {'SG-IoU':>12} {'Entity-IoU':>12} {'Relation-IoU':>12} {'N':>8}")
    print(f"{'-'*70}")
    
    for method_name in sorted(comparison.keys()):
        metrics = comparison[method_name]
        print(f"{method_name:<20} {metrics['sg_iou']:>12.3f} "
              f"{metrics['entity_iou']:>12.3f} {metrics['relation_iou']:>12.3f} "
              f"{metrics['n_images']:>8}")
    
    print(f"{'='*70}")
    
    # Print paper comparison
    print(f"\n{'='*70}")
    print("COMPARISON WITH PAPER RESULTS")
    print(f"{'='*70}")
    print(f"{'Method':<20} {'SG-IoU':>12} {'Entity-IoU':>12} {'Relation-IoU':>12}")
    print(f"{'-'*70}")
    print(f"{'Paper (SG-Adapter)':<20} {0.623:>12.3f} {0.812:>12.3f} {0.753:>12.3f}")
    
    for method_name in sorted(comparison.keys()):
        metrics = comparison[method_name]
        print(f"{method_name:<20} {metrics['sg_iou']:>12.3f} "
              f"{metrics['entity_iou']:>12.3f} {metrics['relation_iou']:>12.3f}")
    
    print(f"{'='*70}\n")
    
    if output_file:
        output_data = {
            'selection_method': f'best_{metric}_per_scene',
            'methods': comparison,
            'paper_baseline': {
                'SG-Adapter': {
                    'sg_iou': 0.623,
                    'entity_iou': 0.812,
                    'relation_iou': 0.753
                }
            }
        }
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"✓ Comparison saved to: {output_file}")
    
    return comparison


def main():
    parser = argparse.ArgumentParser(
        description="Filter evaluation results to keep only best image per scene"
    )
    
    parser.add_argument(
        "--results_dir",
        type=str,
        default="evaluation_results",
        help="Directory containing evaluation results"
    )
    
    parser.add_argument(
        "--metric",
        type=str,
        default="sg_iou",
        choices=['sg_iou', 'entity_iou', 'relation_iou'],
        help="Metric to use for selecting best image"
    )
    
    parser.add_argument(
        "--methods",
        type=str,
        nargs='+',
        default=['gnn_run', 'repr_run'],
        help="Methods to process"
    )
    
    args = parser.parse_args()
    
    # Build methods results dict
    methods_results = {}
    for method in args.methods:
        results_file = f"{args.results_dir}/{method}_results.json"
        if Path(results_file).exists():
            methods_results[method] = results_file
        else:
            print(f"Warning: Results file not found: {results_file}")
    
    if not methods_results:
        print("Error: No valid results files found!")
        return
    
    # Compare methods with best images only
    comparison_file = f"{args.results_dir}/comparison_best_only.json"
    compare_methods_best_only(
        methods_results,
        output_file=comparison_file,
        metric=args.metric
    )
    
    # Also save individual filtered results
    for method_name, results_file in methods_results.items():
        filtered_file = f"{args.results_dir}/{method_name}_best_only.json"
        filter_best_per_scene(results_file, output_file=filtered_file, metric=args.metric)


if __name__ == "__main__":
    main()