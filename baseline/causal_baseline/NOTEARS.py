"""
NOTEARS Baseline Causal Discovery Experiment
Baseline experiment code for causal discovery using NOTEARS algorithm (supports GPU and specified GPUs)
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import json
from datetime import datetime
import torch
from notears.linear import notears_linear
from notears.nonlinear import notears_nonlinear, NotearsMLP


class NOTEARSCausalAnalyzer:
    """
    NOTEARS Causal Discovery Analyzer
    Discovers causal structure using NOTEARS algorithm and analyzes causal chains pointing to is_hot
    """

    def __init__(self, lambda1=0.01, threshold=0.001, use_nonlinear=True, gpu_id=0):
        self.lambda1 = lambda1
        self.threshold = threshold
        self.use_nonlinear = use_nonlinear
        self.weight_matrix = None
        self.directed_edges = []
        self.edges_to_is_hot = []
        self.edges_from_is_hot = []
        self.causal_chains_to_is_hot = []

        # Set device: prioritize torch settings rather than relying solely on CUDA_VISIBLE_DEVICES
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            if gpu_id is None:
                gpu_id = 0
            if gpu_id < 0 or gpu_id >= n_gpus:
                print(f"Warning: Requested gpu_id={gpu_id} is not in available range [0, {n_gpus-1}], will use gpu 0")
                gpu_id = 0
            try:
                torch.cuda.set_device(gpu_id)
            except Exception as e:
                print(f"Failed to set GPU: {e}, will try to continue (may use default GPU)")
            self.device = torch.device("cuda")
            # Try to get device name (non-fatal if fails)
            try:
                dev_name = torch.cuda.get_device_name(gpu_id)
            except Exception:
                dev_name = f"cuda:{gpu_id}"
            print(f"Using GPU: {dev_name} (ID={gpu_id})")
        else:
            self.device = torch.device("cpu")
            print("No GPU detected, running on CPU")

    def discover_causal_structure(self, data):
        """
        Discover causal structure using NOTEARS algorithm
        """
        col_names = list(data.columns)
        print(f"Variable list: {col_names}")

        # Data preprocessing: standardization (numpy)
        data_np = data.values.astype(np.float32)
        stds = data_np.std(axis=0)
        stds[stds == 0] = 1.0
        data_normalized = (data_np - data_np.mean(axis=0)) / stds
        data_normalized = data_normalized.astype(np.float32)

        print("Running NOTEARS algorithm (non-linear, GPU accelerated)...")

        if self.use_nonlinear:
            d = data_normalized.shape[1]
            # Build model and move to specified device
            model = NotearsMLP(dims=[d, 10, 1], bias=True).to(self.device)
            # Note: do not pass device parameter
            W_est = notears_nonlinear(model, data_normalized, lambda1=self.lambda1)
            # Ensure return is numpy
            if isinstance(W_est, torch.Tensor):
                W_est = W_est.detach().cpu().numpy()
        else:
            # Linear version only supports CPU
            W_est = notears_linear(data_normalized, lambda1=self.lambda1, loss_type='l2')

        self.weight_matrix = W_est

        # Extract directed edges
        directed_edges = []
        for i in range(len(col_names)):
            for j in range(len(col_names)):
                if i != j and abs(W_est[i, j]) > self.threshold:
                    src = col_names[i]
                    dst = col_names[j]
                    weight = float(W_est[i, j])
                    directed_edges.append((src, dst, weight))
                    print(f"{src} --> {dst} (weight: {weight:.4f})")

        self.directed_edges = directed_edges
        print(f"Number of directed edges found: {len(directed_edges)}")

        return directed_edges

    def analyze_is_hot_connections(self):
        """
        Analyze edges pointing to is_hot and edges from is_hot
        """
        # Edges pointing to is_hot
        self.edges_to_is_hot = [(src, dst, weight) for src, dst, weight in self.directed_edges if dst == 'is_hot']

        # Edges from is_hot
        self.edges_from_is_hot = [(src, dst, weight) for src, dst, weight in self.directed_edges if src == 'is_hot']

        print(f"Number of edges pointing to is_hot: {len(self.edges_to_is_hot)}")
        print(f"Edges pointing to is_hot:")
        for src, dst, weight in self.edges_to_is_hot:
            print(f"  {src} --> {dst} (weight: {weight:.4f})")

        print(f"Number of edges from is_hot: {len(self.edges_from_is_hot)}")
        print(f"Edges from is_hot:")
        for src, dst, weight in self.edges_from_is_hot:
            print(f"  {src} --> {dst} (weight: {weight:.4f})")

        return self.edges_to_is_hot, self.edges_from_is_hot

    def find_causal_chains_to_is_hot(self, max_chain_length=5):
        """
        Find causal chains pointing to is_hot
        """
        from collections import defaultdict, deque

        # Build adjacency list (only consider edges with weight above threshold)
        graph = defaultdict(list)
        for src, dst, weight in self.directed_edges:
            if abs(weight) > self.threshold:
                graph[src].append((dst, weight))

        # Use BFS to find all paths pointing to is_hot
        chains = []
        queue = deque()

        # Initialize: start from all nodes directly pointing to is_hot
        for src, dst, weight in self.edges_to_is_hot:
            queue.append([(src, weight), (dst, 0)])  # (node, weight)

        while queue:
            path = queue.popleft()

            if len(path) > max_chain_length:
                continue

            # If path reaches max length or no predecessors, save path
            current_node = path[0][0]
            if len(path) == max_chain_length or current_node not in graph:
                # Extract node names
                chain_nodes = [node for node, _ in path]
                chains.append(chain_nodes)
            else:
                # Extend path
                for successor, weight in graph[current_node]:
                    # Avoid cycles
                    if not any(successor == node for node, _ in path):
                        new_path = [(successor, weight)] + path
                        queue.append(new_path)

        self.causal_chains_to_is_hot = chains

        return chains

    def save_results_to_file(self, output_file="notears_causal_report.txt"):
        """
        Save all causal discovery results to file
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("NOTEARS Causal Discovery Experiment Report\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated at: {timestamp}\n")
            f.write(f"Algorithm parameters: lambda1 = {self.lambda1}, threshold = {self.threshold}\n")
            f.write(f"Algorithm type: {'Non-linear NOTEARS' if self.use_nonlinear else 'Linear NOTEARS'}\n\n")

            #  Overall causal graph information
            f.write(" Overall Causal Graph Information\n")
            f.write("-" * 40 + "\n")
            all_variables = set()
            for src, dst, _ in self.directed_edges:
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
    Main function for NOTEARS causal discovery
    """
    parser = argparse.ArgumentParser(description="NOTEARS Causal Discovery (optional GPU ID specification)")
    parser.add_argument("--data", type=str,
                        default='/path/to/your/filtered_data.csv',  # REPLACE WITH YOUR ACTUAL DATA PATH
                        help="Input CSV data path")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use (use CPU if no GPU available)")
    parser.add_argument("--lambda1", type=float, default=0.01, help="NOTEARS lambda1 regularization")
    parser.add_argument("--threshold", type=float, default=0.001, help="Threshold for extracting directed edges")
    parser.add_argument("--nonlinear", action="store_true", default=True, help="Use non-linear NOTEARS (default True, supports GPU)")
    parser.add_argument("--linear", action="store_true", help="Use linear NOTEARS (CPU only)")
    args = parser.parse_args()

    # Determine which algorithm to use: default is non-linear (GPU), unless linear is explicitly specified
    use_nonlinear = args.nonlinear and not args.linear
    print("Starting NOTEARS Causal Discovery Experiment...")
    # Load data
    data_path = args.data
    print(f"Loading data: {data_path}")

    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Cannot find data file {data_path}")
        print("Please check the file path")
        return

    # Create NOTEARS analyzer
    analyzer = NOTEARSCausalAnalyzer(
        lambda1=args.lambda1,
        threshold=args.threshold,
        use_nonlinear=use_nonlinear,
        gpu_id=args.gpu
    )

    # 1. Discover causal structure
    print("\nStep 1: Discovering causal structure using NOTEARS algorithm...")
    directed_edges = analyzer.discover_causal_structure(df)

    # 2. Analyze is_hot connections
    print("\nStep 2: Analyzing is_hot connections...")
    edges_to_is_hot, edges_from_is_hot = analyzer.analyze_is_hot_connections()

    # 3. Find causal chains
    print("\nStep 3: Finding causal chains pointing to is_hot...")
    causal_chains = analyzer.find_causal_chains_to_is_hot(max_chain_length=5)

    # 4. Save results
    print("\nStep 4: Saving results...")
    analyzer.save_results_to_file("notears_causal_report.txt")


if __name__ == "__main__":
    main()