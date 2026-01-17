"""
SG-Adapter Evaluation Script using Gemini API
FIXED VERSION - Properly matches images to metadata
"""

import os
import json
import glob
from pathlib import Path
from typing import List, Dict, Tuple
import google.generativeai as genai
from PIL import Image
import time

# Configure Gemini API
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

class SGAdapterEvaluator:
    def __init__(self, model_name="gemini-2.5-pro"):
        """Initialize the evaluator with Gemini model"""
        self.model = genai.GenerativeModel(model_name)
        
        # These will be extracted from metadata
        self.object_list = set()
        self.predicate_list = set()
    
    def load_metadata(self, metadata_file: str) -> List[Dict]:
        """
        Load metadata from JSONL file
        Returns: list of metadata dicts (preserving order)
        """
        metadata_list = []
        
        with open(metadata_file, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    
                    # Convert relations from index format to actual objects
                    objects = data['objects']
                    relations_idx = data['relations']
                    
                    # Build scene graph with actual object names
                    scene_graph = []
                    for rel in relations_idx:
                        subj_idx = int(rel[0])
                        pred = rel[1]
                        obj_idx = int(rel[2])
                        
                        scene_graph.append([
                            objects[subj_idx],
                            pred,
                            objects[obj_idx]
                        ])
                    
                    metadata_entry = {
                        'file_name': data['file_name'],
                        'caption': data['caption'],
                        'scene_graph': scene_graph,
                        'objects': objects,
                        'relations': relations_idx
                    }
                    
                    metadata_list.append(metadata_entry)
                    
                    # Collect unique objects and predicates
                    self.object_list.update(objects)
                    self.predicate_list.update(rel[1] for rel in relations_idx)
        
        return metadata_list
    
    def extract_scene_graph_from_image(self, image_path: str) -> Dict:
        """Extract scene graph from image using Gemini"""
        
        # Convert sets to sorted lists for prompt
        object_list = sorted(list(self.object_list))
        predicate_list = sorted(list(self.predicate_list))
        
        prompt = f"""Please extract the scene graph of the given image. The scene graph should include the relations of the salient objects.

The objects should be selected from this list: {object_list}

The predicates/relations should be selected from this list: {predicate_list}

Output ONLY a valid JSON object with this exact format:
{{
    "scene_graph": [["subject1", "predicate1", "object1"], ["subject2", "predicate2", "object2"]],
    "entities": ["entity1", "entity2", "entity3"]
}}

Do not include any other text, explanations, or markdown formatting."""

        try:
            img = Image.open(image_path)
            response = self.model.generate_content([prompt, img])
            
            response_text = response.text.strip()
            # Remove markdown code blocks if present
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()
            
            result = json.loads(response_text)
            
            if "scene_graph" not in result or "entities" not in result:
                raise ValueError("Invalid response structure")
            
            return result
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return {"scene_graph": [], "entities": []}
    
    def compute_iou(self, list1: List, list2: List) -> float:
        """Compute Intersection over Union (IoU) for two lists"""
        if not list1 and not list2:
            return 1.0
        if not list1 or not list2:
            return 0.0
        
        # Convert to sets
        if list1 and isinstance(list1[0], list):
            set1 = set(tuple(item) for item in list1)
            set2 = set(tuple(item) for item in list2)
        else:
            set1 = set(list1)
            set2 = set(list2)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def evaluate_image(self, image_path: str, ground_truth_sg: List[List[str]]) -> Dict[str, float]:
        """Evaluate a single image against ground truth scene graph"""
        extracted = self.extract_scene_graph_from_image(image_path)
        predicted_sg = extracted["scene_graph"]
        predicted_entities = extracted["entities"]
        
        # Extract entities and relations from ground truth
        gt_entities = list(set([sg[0] for sg in ground_truth_sg] + [sg[2] for sg in ground_truth_sg]))
        gt_relations = [sg[1] for sg in ground_truth_sg]
        
        # Extract relations from predicted
        predicted_relations = [sg[1] for sg in predicted_sg] if predicted_sg else []
        
        # Compute metrics
        sg_iou = self.compute_iou(ground_truth_sg, predicted_sg)
        entity_iou = self.compute_iou(gt_entities, predicted_entities)
        relation_iou = self.compute_iou(gt_relations, predicted_relations)
        
        return {
            "sg_iou": sg_iou,
            "entity_iou": entity_iou,
            "relation_iou": relation_iou,
            "predicted_sg": predicted_sg,
            "predicted_entities": predicted_entities
        }
    
    def evaluate_method(self, 
                       images_dir: str, 
                       metadata_file: str,
                       output_file: str = None) -> Dict:
        """
        Evaluate all images for a method
        
        Args:
            images_dir: Directory with generated images (e.g., gnn_run/images-30000/images-30000)
            metadata_file: Path to metadata.jsonl or valdata.jsonl file
            output_file: Optional file to save results
        """
        # Load ground truth metadata as a list (preserving order)
        print(f"Loading metadata from: {metadata_file}")
        metadata_list = self.load_metadata(metadata_file)
        print(f"Loaded {len(metadata_list)} entries")
        print(f"Unique objects: {len(self.object_list)}")
        print(f"Unique predicates: {len(self.predicate_list)}")
        
        results = []
        total_metrics = {"sg_iou": 0, "entity_iou": 0, "relation_iou": 0}
        
        # Get all image files
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.JPG']:
            image_files.extend(glob.glob(os.path.join(images_dir, ext)))
        
        # Sort to ensure consistent ordering
        image_files = sorted(image_files)
        
        print(f"\nFound {len(image_files)} images to evaluate")
        
        evaluated = 0
        skipped = 0
        
        for img_path in image_files:
            # Extract the base filename
            base_name = os.path.basename(img_path)
            name_without_ext = os.path.splitext(base_name)[0]
            
            # Extract scene index from filename
            # For files like "000.png", "001_002.png", extract the first number
            scene_idx = None
            try:
                # Split by underscore and take first part
                first_part = name_without_ext.split('_')[0]
                if first_part.isdigit():
                    scene_idx = int(first_part)
            except:
                pass
            
            # Get corresponding metadata
            if scene_idx is not None and scene_idx < len(metadata_list):
                matching_meta = metadata_list[scene_idx]
            else:
                print(f"Warning: No metadata found for {base_name} (extracted index: {scene_idx})")
                skipped += 1
                continue
            
            print(f"Evaluating: {base_name} -> Index {scene_idx} ({matching_meta['caption']})")
            
            gt_scene_graph = matching_meta['scene_graph']
            gt_caption = matching_meta['caption']
            
            # Evaluate image
            metrics = self.evaluate_image(img_path, gt_scene_graph)
            
            results.append({
                "image": base_name,
                "scene_index": scene_idx,
                "caption": gt_caption,
                "ground_truth_sg": gt_scene_graph,
                **metrics
            })
            
            total_metrics["sg_iou"] += metrics["sg_iou"]
            total_metrics["entity_iou"] += metrics["entity_iou"]
            total_metrics["relation_iou"] += metrics["relation_iou"]
            
            evaluated += 1
            
            # Rate limiting
            time.sleep(2)
        
        print(f"\nEvaluated: {evaluated}, Skipped: {skipped}")
        
        # Compute averages
        n_images = len(results)
        avg_metrics = {
            "sg_iou": total_metrics["sg_iou"] / n_images if n_images > 0 else 0,
            "entity_iou": total_metrics["entity_iou"] / n_images if n_images > 0 else 0,
            "relation_iou": total_metrics["relation_iou"] / n_images if n_images > 0 else 0,
            "n_images": n_images
        }
        
        output = {
            "average_metrics": avg_metrics,
            "per_image_results": results
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"Results saved to {output_file}")
        
        return output
    
    def compare_methods(self, 
                       methods_config: List[Dict],
                       metadata_file: str,
                       output_dir: str = "evaluation_results"):
        """
        Compare multiple methods
        
        Args:
            methods_config: List of dicts with 'name' and 'images_dir' keys
            metadata_file: Path to metadata.jsonl file
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        comparison = {}
        
        for config in methods_config:
            method_name = config['name']
            images_dir = config['images_dir']
            
            print(f"\n{'='*50}")
            print(f"Evaluating method: {method_name}")
            print(f"{'='*50}\n")
            
            output_file = os.path.join(output_dir, f"{method_name}_results.json")
            results = self.evaluate_method(
                images_dir, metadata_file, output_file
            )
            
            comparison[method_name] = results["average_metrics"]
            
            print(f"\n{method_name} Results:")
            print(f"  SG-IoU:       {results['average_metrics']['sg_iou']:.3f}")
            print(f"  Entity-IoU:   {results['average_metrics']['entity_iou']:.3f}")
            print(f"  Relation-IoU: {results['average_metrics']['relation_iou']:.3f}")
        
        # Save comparison
        comparison_file = os.path.join(output_dir, "comparison.json")
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        # Print comparison table
        print(f"\n{'='*70}")
        print("COMPARISON TABLE")
        print(f"{'='*70}")
        print(f"{'Method':<20} {'SG-IoU':>12} {'Entity-IoU':>12} {'Relation-IoU':>12}")
        print(f"{'-'*70}")
        for method_name, metrics in comparison.items():
            print(f"{method_name:<20} {metrics['sg_iou']:>12.3f} "
                  f"{metrics['entity_iou']:>12.3f} {metrics['relation_iou']:>12.3f}")
        print(f"{'='*70}\n")
        
        return comparison


if __name__ == "__main__":
    # Example usage
    
    # Set API key
    # os.environ["GEMINI_API_KEY"] = "your-api-key-here"
    
    evaluator = SGAdapterEvaluator()
    
    # Define your methods configuration
    methods_config = [
        {
            'name': 'gnn_run',
            'images_dir': 'gnn_run/images-30000/images-30000'
        },
        {
            'name': 'repr_run',
            'images_dir': 'repr_run/images-30000/images-30000'
        }
    ]
    
    # Run comparison using valdata.jsonl for validation
    comparison = evaluator.compare_methods(
        methods_config=methods_config,
        metadata_file='dataset/MultiRels/valdata.jsonl',
        output_dir='evaluation_results'
    )