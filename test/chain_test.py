import json
from collections import defaultdict


class ChainEvaluator:
    def __init__(self, factor_mapping):
        self.factor_mapping = factor_mapping

    def normalize_chain(self, chain_str):
        """
        Normalize chain string: split and map factors
        """
        factors = chain_str.split(' -> ')
        normalized_factors = []
        for factor in factors:
            factor = factor.strip()
            normalized_factor = self.factor_mapping.get(factor, factor)
            normalized_factors.append(normalized_factor)
        return normalized_factors

    def normalize_chains(self, chains):
        """Normalize all chains"""
        return [self.normalize_chain(chain) for chain in chains]

    def extract_direct_edges(self, normalized_chains):
        """Extract all direct edges from normalized chains"""
        edges = set()
        for chain in normalized_chains:
            for i in range(len(chain) - 1):
                edge = (chain[i], chain[i + 1])
                edges.add(edge)
        return edges

    def calculate_edge_metrics(self, base_edges, method_edges):
        """Calculate edge-level precision, recall, and F1"""
        base_set = set(base_edges)
        method_set = set(method_edges)

        true_positives = len(base_set & method_set)

        precision = true_positives / len(method_set) if len(method_set) > 0 else 0
        recall = true_positives / len(base_set) if len(base_set) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return precision, recall, f1

    def calculate_cSHD(self, base_chains, method_chains):
        """
        Calculate cSHD: Chain set difference based on edit distance
        Simplified implementation: Calculate edge-level difference
        """
        base_edges = self.extract_direct_edges(base_chains)
        method_edges = self.extract_direct_edges(method_chains)

        unique_to_base = base_edges - method_edges
        unique_to_method = method_edges - base_edges
        cSHD = len(unique_to_base) + len(unique_to_method)

        return cSHD

    def calculate_cSID(self, base_chains, method_chains):
        """
        Calculate cSID: Difference based on intervention effects
        Simplified implementation: Calculate reachable path differences
        """

        def get_reachability(chains):
            """Get reachability between all factor pairs"""
            reachable = set()
            for chain in chains:
                for i in range(len(chain)):
                    for j in range(i + 1, len(chain)):
                        reachable.add((chain[i], chain[j]))
            return reachable

        base_reachable = get_reachability(base_chains)
        method_reachable = get_reachability(method_chains)

        unique_to_base = base_reachable - method_reachable
        unique_to_method = method_reachable - base_reachable
        cSID = len(unique_to_base) + len(unique_to_method)

        return cSID

    def calculate_cSHD_error_rate(self, base_chains, method_chains):
        """
        Calculate cSHD error rate: Wrong edges / Total edges
        """
        base_edges = self.extract_direct_edges(base_chains)
        method_edges = self.extract_direct_edges(method_chains)

        union_edges = base_edges | method_edges
        intersection_edges = base_edges & method_edges

        error_count = len(union_edges) - len(intersection_edges)
        total_possible_edges = len(union_edges)

        error_rate = error_count / total_possible_edges if total_possible_edges > 0 else 1.0
        return error_rate, error_count, total_possible_edges

    def calculate_cSID_error_rate(self, base_chains, method_chains):
        """
        Calculate cSID error rate: Wrong predicted intervention effects / Total predicted effects
        """

        def get_reachability(chains):
            reachable = set()
            for chain in chains:
                for i in range(len(chain)):
                    for j in range(i + 1, len(chain)):
                        reachable.add((chain[i], chain[j]))
            return reachable

        base_reachable = get_reachability(base_chains)
        method_reachable = get_reachability(method_chains)

        union_reachable = base_reachable | method_reachable
        intersection_reachable = base_reachable & method_reachable

        error_count = len(union_reachable) - len(intersection_reachable)
        total_predicted_effects = len(union_reachable)

        error_rate = error_count / total_predicted_effects if total_predicted_effects > 0 else 1.0
        return error_rate, error_count, total_predicted_effects


def main():
    fallback_factor_mapping = {
        # REPLACE WITH YOUR ACTUAL FACTOR MAPPINGS
    }

    # Try to load mapping from evaluation file
    factor_mapping = None
    eval_json_path = '/path/to/your/methods_factors_evaluation.json'  # REPLACE WITH YOUR ACTUAL PATH
    
    try:
        with open(eval_json_path, 'r', encoding='utf-8') as f:
            eval_data = json.load(f)

        eval_map = {}
        for method_name, payload in eval_data.items():
            if not isinstance(payload, dict):
                continue
            details = payload.get('details', {})
            for key in ['exact_matches', 'high_similarity_matches']:
                matches = details.get(key, []) or []
                for item in matches:
                    src = item.get('predicted_factor')
                    dst = item.get('best_match_factor')
                    if src and dst:
                        eval_map[src] = dst
                        eval_map[src.strip()] = dst

        factor_mapping = fallback_factor_mapping.copy()
        factor_mapping.update(eval_map)
    except Exception as e:
        print(f"Failed to load mapping from evaluation file, using fallback. Reason: {e}")
        factor_mapping = fallback_factor_mapping

    # Load data
    input_json_path = '/path/to/your/methods_chains.json'  # REPLACE WITH YOUR ACTUAL PATH
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    evaluator = ChainEvaluator(factor_mapping)

    base_chains = evaluator.normalize_chains(data['base_chains'])
    base_edges = evaluator.extract_direct_edges(base_chains)

    results = {}
    methods = []

    for method in methods:
        if not data[method]:
            continue

        method_chains = evaluator.normalize_chains(data[method])
        method_edges = evaluator.extract_direct_edges(method_chains)

        edge_precision, edge_recall, edge_f1 = evaluator.calculate_edge_metrics(base_edges, method_edges)
        cSHD = evaluator.calculate_cSHD(base_chains, method_chains)
        cSID = evaluator.calculate_cSID(base_chains, method_chains)
        cSHDr = evaluator.calculate_cSHD_error_rate(base_chains, method_chains)
        cSIDr = evaluator.calculate_cSID_error_rate(base_chains, method_chains)

        results[method] = {
            'edge_precision': round(edge_precision, 4),
            'edge_recall': round(edge_recall, 4),
            'edge_f1': round(edge_f1, 4),
            'cSHD': cSHD,
            'cSID': cSID,
            'cSHDr': cSHDr,
            'cSIDr': cSIDr
        }

    output_path = '/path/to/your/evaluation_results.json'  # REPLACE WITH YOUR DESIRED OUTPUT PATH
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Evaluation completed! Results saved to evaluation_results.json")


if __name__ == '__main__':
    main()