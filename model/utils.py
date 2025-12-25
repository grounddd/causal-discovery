"""
Utility Functions Module
Helper functions for various operations
"""

import json
import numpy as np
from typing import List, Dict


def save_chains_to_file(chains: list, 
                       filename: str = 'output_chains.txt',
                       col2invmap: dict = None) -> None:
    """
    Save chains to text file
    
    Args:
        chains: List of chains to save
        filename: Output filename
        col2invmap: Optional inverse mapping for decoding
    """
    with open(filename, 'w', encoding='utf-8') as f:
        for i, (path, scores, score) in enumerate(chains):
            if col2invmap:
                from data_loader import DataLoader
                loader = DataLoader()
                decoded_path = loader.decode_chain(path, col2invmap)
                chain_str = ' -> '.join(decoded_path)
            else:
                chain_str = ' -> '.join(path)
            
            f.write(f"{i + 1}. Chain: {chain_str}, Score: {score:.3f}, Steps: {scores}\n")
    
    print(f"Chains saved to {filename}")


def save_top_factors_summary(summary: List[Dict], 
                           filename: str = 'top_factors_summary.json') -> None:
    """
    Save top factors summary to JSON file
    
    Args:
        summary: List of factor summaries
        filename: Output filename
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({'summary': summary}, f, ensure_ascii=False, indent=2)
        print(f"Top factors summary saved to {filename}")
        
    except Exception as e:
        print(f"Error saving summary: {e}")


def compute_summary_statistics(chains: list, data: pd.DataFrame) -> Dict:
    """
    Compute summary statistics for chains
    
    Args:
        chains: List of chains
        data: Input data
        
    Returns:
        Dictionary with statistics
    """
    from chain_scoring import ChainScorer
    scorer = ChainScorer()
    
    stats = {
        'total_chains': len(chains),
        'average_chain_length': np.mean([len(chain[0]) for chain in chains]) if chains else 0,
        'unique_factors': len(set([node.split('=')[0] for chain in chains for node in chain[0]])) if chains else 0
    }
    
    # Compute entropy gains for top chains
    if chains:
        entropy_gains = []
        for chain in chains[:10]:  # Top 10 chains
            gain, coverage, p_hot, coverage_rate = scorer.compute_chain_entropy_gain(chain[0], data)
            entropy_gains.append({
                'chain': ' -> '.join(chain[0]),
                'entropy_gain': gain,
                'coverage': coverage,
                'p_hot': p_hot,
                'coverage_rate': coverage_rate
            })
        
        stats['top_chains_entropy'] = entropy_gains
    
    return stats


def print_summary_table(chains: list, top_n: int = 10) -> None:
    """
    Print summary table of top chains
    
    Args:
        chains: List of chains
        top_n: Number of top chains to display
    """
    print(f"\n{'='*80}")
    print(f"TOP {min(top_n, len(chains))} CAUSAL CHAINS")
    print(f"{'='*80}")
    
    for i, (path, scores, score) in enumerate(chains[:top_n], 1):
        print(f"{i:2d}. Chain: {' -> '.join(path)}")
        print(f"    Score: {score:.3f} | Steps: {len(path)} | Avg Score: {np.mean(scores) if scores else 0:.3f}")
        print()
