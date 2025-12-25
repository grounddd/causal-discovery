"""
PC Baseline Causal Discovery Experiment
Baseline experiment code for causal discovery using PC algorithm
"""
import numpy as np
import pandas as pd
import networkx as nx
import json
from datetime import datetime
from causallearn.search.ConstraintBased.PC import pc
from causallearn.graph.Endpoint import Endpoint


class PCCausalAnalyzer:
    """
    PC Causal Discovery Analyzer
    Discovers causal structure using PC algorithm and analyzes causal chains pointing to is_hot
    """

    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.causal_graph = None
        self.directed_edges = []
        self.edges_to_is_hot = []
        self.edges_from_is_hot = []
        self.causal_chains_to_is_hot = []

    def discover_causal_structure(self, data):
        """
        Discover causal structure using PC algorithm
        """
        col_names = list(data.columns)
        print(f"Variable list: {col_names}")

        # Convert to numpy array
        data_np = data.values

        # Run PC algorithm
        print("Running PC algorithm...")
        cg = pc(data_np, alpha=self.alpha, verbose=False)

        # Extract directed edges - use correct API
        directed_edges = []
        try:
            # Use correct API: cg.G.get_graph_edges()
            for edge in cg.G.get_graph_edges():
                if edge.endpoint1 == Endpoint.TAIL and edge.endpoint2 == Endpoint.ARROW:
                    try:
                        src_idx = int(edge.node1.get_name().replace('X', '')) - 1
                        dst_idx = int(edge.node2.get_name().replace('X', '')) - 1
                        if 0 <= src_idx < len(col_names) and 0 <= dst_idx < len(col_names):
                            src = col_names[src_idx]
                            dst = col_names[dst_idx]
                            directed_edges.append((src, dst))
                            print(f"Found edge: {src} -> {dst}")
                    except Exception as e:
                        print(f'Edge mapping error: {e}')
                        continue
        except Exception as e:
            print(f'Error extracting edges: {e}')
            directed_edges = []

        self.causal_graph = cg
        self.directed_edges = directed_edges

        print(f"Number of directed edges found: {len(directed_edges)}")
        print(f"Directed edges: {directed_edges}")

        return directed_edges

    def analyze_is_hot_connections(self):
        """
        Analyze edges pointing to is_hot and edges from is_hot
        """
        # Edges pointing to is_hot
        self.edges_to_is_hot = [(src, dst) for src, dst in self.directed_edges if dst == 'is_hot']

        # Edges from is_hot
        self.edges_from_is_hot = [(src, dst) for src, dst in self.directed_edges if src == 'is_hot']

        print(f"Number of edges pointing to is_hot: {len(self.edges_to_is_hot)}")
        print(f"Edges pointing to is_hot: {self.edges_to_is_hot}")
        print(f"Number of edges from is_hot: {len(self.edges_from_is_hot)}")
        print(f"Edges from is_hot: {self.edges_from_is_hot}")

        return self.edges_to_is_hot, self.edges_from_is_hot

    def find_causal_chains_to_is_hot(self, max_chain_length=5):
        """
        Find causal chains pointing to is_hot
        """
        from collections import defaultdict, deque

        # Build adjacency list
        graph = defaultdict(list)
        for src, dst in self.directed_edges:
            graph[src].append(dst)

        # Use BFS to find all paths pointing to is_hot
        chains = []
        queue = deque()

        # Initialize: start from all nodes directly pointing to is_hot
        for src, dst in self.edges_to_is_hot:
            queue.append([src, dst])

        while queue:
            path = queue.popleft()

            if len(path) > max_chain_length:
                continue

            # If path reaches max length or no predecessors, save path
            current_node = path[0]
            if len(path) == max_chain_length or current_node not in graph:
                chains.append(path)
            else:
                # Extend path
                for predecessor in graph[current_node]:
                    if predecessor not in path:  # Avoid cycles
                        new_path = [predecessor] + path
                        queue.append(new_path)

        self.causal_chains_to_is_hot = chains

        return chains

    def save_results_to_file(self, output_file="pc_causal_report.txt"):
        """
        Save all causal discovery results to file
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("PC Causal Discovery Experiment Report\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated at: {timestamp}\n")
            f.write(f"Algorithm parameters: alpha = {self.alpha}\n\n")

            #  Overall causal graph information
            f.write(" Overall Causal Graph Information\n")
            f.write("-" * 40 + "\n")
            all_variables = set()
            for src, dst in self.directed_edges:
                all_variables.add(src)
                all_variables.add(dst)
            f.write(f"Total variables: {len(all_variables)}\n")
            f.write(f"Total edges: {len(self.directed_edges)}\n\n")

            #  Causal chains pointing to is_hot
            f.write(" Causal Chains Pointing to is_hot\n")
            f.write("-" * 40 + "\n")
            f.write(f"Number of causal chains: {len(self.causal_chains_to_is_hot)}\n")
            for i, chain in enumerate(self.causal_chains_to_is_hot, 1):
                f.write(f"{i:3d}. {' -> '.join(chain)}\n")
            f.write("\n")


def main():
    """
    Main function for PC causal discovery
    """
    print("Starting PC Causal Discovery Experiment...")

    # Load data
    # REPLACE WITH YOUR ACTUAL DATA PATH
    data_path = '/path/to/your/filtered_data.csv'
    print(f"Loading data: {data_path}")

    try:
        df = pd.read_csv(data_path)
        print(f"Data shape: {df.shape}")
        print(f"Column names: {list(df.columns)}")
    except FileNotFoundError:
        print(f"Error: Cannot find data file {data_path}")
        print("Please check the file path")
        return

    # Create PC analyzer
    analyzer = PCCausalAnalyzer(alpha=0.1)

    # 1. Discover causal structure
    print("\nStep 1: Discovering causal structure using PC algorithm...")
    directed_edges = analyzer.discover_causal_structure(df)

    # 2. Analyze is_hot connections
    print("\nStep 2: Analyzing is_hot connections...")
    edges_to_is_hot, edges_from_is_hot = analyzer.analyze_is_hot_connections()

    # 3. Find causal chains
    print("\nStep 3: Finding causal chains pointing to is_hot...")
    causal_chains = analyzer.find_causal_chains_to_is_hot(max_chain_length=5)

    # 4. Save results
    print("\nStep 4: Saving results...")
    analyzer.save_results_to_file("pc_causal_report.txt")

if __name__ == "__main__":
    main()