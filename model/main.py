"""
Main Program Entry Point
Orchestrates the complete causal discovery pipeline
"""

import pandas as pd
import json
from factor_filter import FactorFilter
from causal_analyzer import CausalChainAnalyzer
from chain_scoring import ChainScorer
from data_loader import DataLoader
from utils import save_chains_to_file, save_top_factors_summary, print_summary_table


def main(target_factor_count: int = 27):
    """
    Main function - orchestrates complete pipeline
    
    Args:
        target_factor_count: Number of factors to keep
    """
    print("🚀 Causal Chain Discovery Pipeline")
    print("=" * 80)
    
    # Step 1: Load data
    print("\n📊 Step 1: Loading data...")
    loader = DataLoader()
    X, y, full_data = loader.load_weibo_data(
        '/path/to/your/test_data.csv'  # REPLACE WITH YOUR PATH
    )
    
    # Load inverse mapping
    col2invmap = loader.load_inverse_mapping(
        '/path/to/your/col2invmap.json'  # REPLACE WITH YOUR PATH
    )
    
    # Step 2: Filter factors
    print("\n🔍 Step 2: Filtering factors...")
    factor_filter = FactorFilter()
    filtered_factors = factor_filter.filter_factors(
        full_data, 
        target_factor_count=target_factor_count
    )
    
    # Save filtered data
    filtered_data = factor_filter.save_filtered_data(
        full_data, 
        filtered_factors,
        output_path='filtered_data.csv'
    )
    
    # Update X with filtered factors
    X = X[list(filtered_factors)]
    
    # Step 3: Split data
    print("\n✂️ Step 3: Splitting data...")
    X_train, X_test, y_train, y_test = loader.split_data(X, y)
    X_train['is_hot'] = y_train.values
    
    # Step 4: Causal analysis
    print("\n🔗 Step 4: Discovering causal chains...")
    analyzer = CausalChainAnalyzer(max_chain_length=6)
    
    # Find causal edges with LiNGAM
    element_edges = analyzer.find_direct_causal_pairs_lingam(X_train)
    print(f"Found {len(element_edges)} causal edges")
    
    # Reweight edges with contrastive analysis
    reweighted_edges = analyzer.contrast_reweight_edges(
        X_train, element_edges, 'is_hot'
    )
    analyzer.causal_pairs = reweighted_edges
    
    # Build multi-step chains
    element_chains = analyzer.build_multi_step_chains('is_hot')
    print(f"Built {len(element_chains)} element-level chains")
    
    # Step 5: Score chains
    print("\n📈 Step 5: Scoring chains...")
    scorer = ChainScorer()
    
    # Build causal graph
    causal_graph = scorer.learn_causal_graph(analyzer.causal_pairs)
    
    # Score chains with SCM
    scored_element_chains = scorer.score_chains_with_scm(
        X_train, element_chains, causal_graph
    )
    
    # Filter to top chains
    top_element_chains = [item[0] for item in scored_element_chains[:K]]
    
    # Build value-level chains
    value_chains = scorer.build_value_chains_backward(
        X_train, top_element_chains, 'is_hot', 1
    )
    
    # Filter unique chains
    unique_chains = scorer.filter_unique_chains(value_chains)
    print(f"Final unique chains: {len(unique_chains)}")
    
    # Step 6: Save results
    print("\n💾 Step 6: Saving results...")
    save_chains_to_file(
        unique_chains, 
        'sorted_chains.txt',
        col2invmap
    )
    
    # Generate and save top factors summary
    summary = analyzer.summarize_top_factors_and_values(
        X_train, 
        target_col='is_hot', 
        top_factors=5, 
        top_values=3
    )
    
    save_top_factors_summary(summary, 'top_factors_summary.json')
    
    # Print summary
    print_summary_table(unique_chains, top_n=10)
    
    print("\n✅ Pipeline completed successfully!")


if __name__ == "__main__":
    # Run with default parameters
    main(target_factor_count=27)