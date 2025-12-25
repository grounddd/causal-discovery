"""
Factor-level Evaluation for Causal Discovery
Factor-level evaluation code for assessing matching between discovered factors and benchmark factors

Evaluation metrics:
1. Exact Match Ratio
2. High Similarity Ratio  
3. Strict/Relaxed Recall, Precision, F1
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
import os
from typing import List, Dict


class FactorEvaluator:

    def __init__(self, exact_match_threshold: float = 0.95, high_similarity_threshold: float = 0.7):
        self.exact_match_threshold = exact_match_threshold
        self.high_similarity_threshold = high_similarity_threshold
        self.base_factors = []
        self.predicted_factors = []

    def load_factors_from_file(self, file_path: str) -> List[str]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                factors = data.get('factors', [])
                print(f"Successfully loaded {len(factors)} factors from {file_path}")
                return factors
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return []
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return []
        except Exception as e:
            print(f"Error loading file: {e}")
            return []

    def calculate_similarity_matrix(self) -> np.ndarray:

        similarity_matrix = np.zeros((len(self.predicted_factors), len(self.base_factors)))

        for i, pred_factor in enumerate(self.predicted_factors):
            print(f"Processing predicted factor {i + 1}/{len(self.predicted_factors)}: {pred_factor}")

            try:
                # REPLACE WITH YOUR EMBEDDING FUNCTION IMPLEMENTATION
                pred_embedding = self.get_embedding(pred_factor)

                for j, base_factor in enumerate(self.base_factors):
                    try:
                        # REPLACE WITH YOUR EMBEDDING FUNCTION IMPLEMENTATION
                        base_embedding = self.get_embedding(base_factor)
                        similarity = self.cosine_similarity(pred_embedding, base_embedding)
                        similarity_matrix[i, j] = similarity
                    except Exception as e:
                        print(f"  Error calculating similarity for {base_factor}: {e}")
                        similarity_matrix[i, j] = 0.0

            except Exception as e:
                print(f"Error getting embedding for {pred_factor}: {e}")
                similarity_matrix[i, j] = 0.0

        print("Similarity matrix calculation completed")
        return similarity_matrix

    def get_embedding(self, text: str):
        # IMPLEMENT YOUR EMBEDDING LOGIC HERE
        # Example: return openai.Embedding.create(...)
        raise NotImplementedError("Implement get_embedding method with your embedding approach")

    def cosine_similarity(self, vec1, vec2):
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def find_matches(self, similarity_matrix: np.ndarray) -> Dict:

        best_matches = []
        exact_matches = []
        high_similarity_matches = []

        for i, pred_factor in enumerate(self.predicted_factors):
            max_similarity = np.max(similarity_matrix[i, :])
            best_match_idx = np.argmax(similarity_matrix[i, :])
            best_match_factor = self.base_factors[best_match_idx]

            match_info = {
                'predicted_factor': pred_factor,
                'best_match_factor': best_match_factor,
                'similarity': max_similarity,
                'match_type': 'none'
            }

            if max_similarity >= self.exact_match_threshold:
                match_info['match_type'] = 'exact'
                exact_matches.append(match_info)
            elif max_similarity >= self.high_similarity_threshold:
                match_info['match_type'] = 'high_similarity'
                high_similarity_matches.append(match_info)

            best_matches.append(match_info)

        # Calculate statistics
        total_predicted = len(self.predicted_factors)
        exact_count = len(exact_matches)
        high_similarity_count = len(high_similarity_matches)
        correct_count = exact_count + high_similarity_count

        exact_match_ratio = exact_count / total_predicted if total_predicted > 0 else 0
        high_similarity_ratio = high_similarity_count / total_predicted if total_predicted > 0 else 0
        correct_ratio = correct_count / total_predicted if total_predicted > 0 else 0

        # Benchmark factor perspective metrics (for paper tables)
        base_count = len(self.base_factors)
        base_best_sims = []
        
        if base_count > 0:
            for j in range(base_count):
                if similarity_matrix.shape[0] > 0:
                    best_sim = float(np.max(similarity_matrix[:, j]))
                else:
                    best_sim = 0.0
                base_best_sims.append(best_sim)

        # Calculate paper-defined metrics
        msr_count = sum(1 for s in base_best_sims if s >= self.exact_match_threshold)
        hsr_count = sum(1 for s in base_best_sims if s >= self.high_similarity_threshold)
        msr_avg_sum = sum(s for s in base_best_sims if s >= self.exact_match_threshold)
        hsr_avg_sum = sum(s for s in base_best_sims if s >= self.high_similarity_threshold)
        overall_sim_sum = sum(base_best_sims)

        msr = (msr_count / base_count) if base_count > 0 else 0.0
        hsr = (hsr_count / base_count) if base_count > 0 else 0.0
        msr_avg = (msr_avg_sum / msr_count) if msr_count > 0 else 0.0
        hsr_avg = (hsr_avg_sum / hsr_count) if hsr_count > 0 else 0.0
        avg_similarity = (overall_sim_sum / base_count) if base_count > 0 else 0.0

        return {
            'best_matches': best_matches,
            'exact_matches': exact_matches,
            'high_similarity_matches': high_similarity_matches,
            'statistics': {
                'total_predicted': total_predicted,
                'total_base': len(self.base_factors),
                'exact_count': exact_count,
                'high_similarity_count': high_similarity_count,
                'correct_count': correct_count,
                'exact_match_ratio': exact_match_ratio,
                'high_similarity_ratio': high_similarity_ratio,
                'correct_ratio': correct_ratio,
                'MSR': msr,
                'MSR_avg': msr_avg,
                'HSR': hsr,
                'HSR_avg': hsr_avg,
                'avg_similarity': avg_similarity
            }
        }


def evaluate_methods_factors(input_json_path: str,
                             output_json_path: str = "methods_factors_evaluation.json",
                             exact_match_threshold: float = 0.8,
                             high_similarity_threshold: float = 0.5) -> Dict:
    try:
        # REPLACE WITH YOUR ACTUAL INPUT JSON PATH
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to read {input_json_path}: {e}")
        return {}

    if 'base' not in data or not isinstance(data['base'], list):
        print("Input JSON missing 'base' list")
        return {}

    base_factors = data['base']
    results: Dict[str, Dict] = {}

    for method_name, factors in data.items():
        if method_name == 'base':
            continue
        if not isinstance(factors, list):
            continue

        evaluator = FactorEvaluator(
            exact_match_threshold=exact_match_threshold,
            high_similarity_threshold=high_similarity_threshold,
        )
        evaluator.base_factors = base_factors
        evaluator.predicted_factors = factors

        similarity_matrix = evaluator.calculate_similarity_matrix()
        match_results = evaluator.find_matches(similarity_matrix)

        stats = match_results['statistics']
        total_pred = stats['total_predicted']
        exact_cnt = stats['exact_count']
        high_cnt = stats['high_similarity_count']
        none_cnt = max(total_pred - exact_cnt - high_cnt, 0)

        best_matches = match_results['best_matches']
        none_matches = [m for m in best_matches if m.get('match_type') == 'none']

        results[method_name] = {
            'MSR': round(stats.get('MSR', 0.0), 4),
            'MSR_avg': round(stats.get('MSR_avg', 0.0), 4),
            'HSR': round(stats.get('HSR', 0.0), 4),
            'HSR_avg': round(stats.get('HSR_avg', 0.0), 4),
            'avg_similarity': round(stats.get('avg_similarity', 0.0), 4),
        }

    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f" evaluation results saved to: {output_json_path}")
    except Exception as e:
        print(f"Failed to save results: {e}")

    return results


if __name__ == "__main__":
    try:
        # REPLACE WITH YOUR ACTUAL INPUT JSON PATH
        input_json_path = '/path/to/your/methods_factors.json'
        output_json_path = 'methods_factors_evaluation.json'
        
        print("\n=== Running Multi-method Factor Evaluation ===")
        evaluate_methods_factors(
            input_json_path=input_json_path,
            output_json_path=output_json_path,
            exact_match_threshold=0.8,
            high_similarity_threshold=0.5,
        )
    except Exception as e:
        print(f"Multi-method evaluation failed: {e}")