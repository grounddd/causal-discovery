"""
Causal Analyzer Module
Main causal discovery and chain analysis functionality
"""

import numpy as np
import pandas as pd
from collections import defaultdict, deque
from sklearn.linear_model import LinearRegression
from scipy import stats
from lingam import DirectLiNGAM


class CausalChainAnalyzer:
    def __init__(self, max_chain_length: int = 5):
        """
        Initialize causal chain analyzer
        
        Args:
            max_chain_length: Maximum length of causal chains
        """
        self.max_chain_length = max_chain_length
        self.causal_pairs = []
        self.causal_chains = []
        self.causal_graph = {}
        self.structural_equations = {}
    
    def find_direct_causal_pairs_lingam(self, data: pd.DataFrame) -> list:
        """
        Find direct causal pairs using LiNGAM algorithm
        
        Args:
            data: Input dataframe
            
        Returns:
            List of causal edges (source, target, weight)
        """
        col_names = list(data.columns)
        data_np = data.values
        
        # Run LiNGAM algorithm
        model = DirectLiNGAM()
        model.fit(data_np)
        adj_matrix = model.adjacency_matrix_
        
        lingam_edges = []
        threshold = 0.01
        
        for i in range(len(col_names)):
            for j in range(len(col_names)):
                if i != j and abs(adj_matrix[i, j]) > threshold:
                    src = col_names[i]
                    dst = col_names[j]
                    weight = adj_matrix[i, j]
                    lingam_edges.append((src, dst, weight))
        
        return lingam_edges
    
    def _compute_feature_contrast(self, data: pd.DataFrame, feature: str, target_col: str) -> float:
        """
        Compute contrast between hot and cold groups for a feature
        """
        hot = data[data[target_col] == 1][feature].dropna()
        cold = data[data[target_col] == 0][feature].dropna()
        
        if hot.empty or cold.empty:
            return 0.0
        
        # Numerical features with many values: t-test + Cohen's d
        if hot.dtype.kind in ("i", "u", "f") and cold.dtype.kind in ("i", "u", "f"):
            if max(hot.nunique(), cold.nunique()) > 5:
                try:
                    pooled = np.sqrt((hot.std(ddof=1) ** 2 + cold.std(ddof=1) ** 2) / 2.0)
                    if pooled == 0 or np.isnan(pooled):
                        return 0.0
                    d = (hot.mean() - cold.mean()) / pooled
                    return float(abs(d))
                except Exception:
                    return 0.0
        
        # Binary features: log odds ratio
        if max(hot.nunique(), cold.nunique()) == 2:
            vc_h = hot.value_counts()
            vc_c = cold.value_counts()
            keys = sorted(list(set(vc_h.index.tolist() + vc_c.index.tolist())))
            
            if len(keys) != 2:
                return 0.0
            
            a = vc_h.get(keys[1], 0)
            b = vc_h.get(keys[0], 0)
            c = vc_c.get(keys[1], 0)
            d = vc_c.get(keys[0], 0)
            or_ = (a * d + 0.5) / ((b * c) + 0.5)
            return float(abs(np.log(or_)))
        
        # Multi-class categorical: Cramér's V
        if max(hot.nunique(), cold.nunique()) <= 10:
            try:
                x = pd.crosstab(hot, cold)
                chi2, _, _, _ = stats.chi2_contingency(x)
                n = x.values.sum()
                r, k = x.shape
                phi2 = chi2 / max(n, 1)
                v = np.sqrt(phi2 / max(min(k - 1, r - 1), 1))
                return float(v)
            except Exception:
                return 0.0
        
        # Ordinal/discrete: Mann-Whitney approximate r
        try:
            u, _ = stats.mannwhitneyu(hot, cold, alternative="two-sided")
            n1, n2 = len(hot), len(cold)
            mu_u = n1 * n2 / 2
            sigma_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
            
            if sigma_u == 0:
                return 0.0
            
            z = (u - mu_u) / sigma_u
            r = abs(z) / np.sqrt(n1 + n2)
            return float(r)
        except Exception:
            return 0.0
    
    def contrast_reweight_edges(self, 
                               data: pd.DataFrame, 
                               element_edges: list, 
                               target_col: str) -> list:
        """
        Reweight edges based on contrastive analysis
        """
        # Compute contrast scores for all features
        features = [col for col in data.columns if col != target_col]
        feat2contrast = {f: self._compute_feature_contrast(data, f, target_col) 
                        for f in features}
        
        # Normalize contrast scores
        vals = np.array(list(feat2contrast.values())) if feat2contrast else np.array([0.0])
        vmax = float(vals.max()) if vals.size else 0.0
        norm = {f: (0.1 + 0.9 * feat2contrast[f] / vmax if vmax > 0 else 0.1) 
                for f in feat2contrast}
        
        # Reweight edges
        eps = 1e-6
        reweighted = []
        
        for (src, dst, w) in element_edges:
            base = abs(float(w)) if w is not None else 0.0
            
            if dst == target_col:
                b_contrast = norm.get(src, 0.1)
            else:
                b_contrast = norm.get(dst, 0.1)
            
            score = base * (eps + b_contrast) + base * 0.3
            reweighted.append((src, dst, score))
        
        return reweighted
    
    def build_multi_step_chains(self, target_col: str) -> list:
        """
        Build multi-step causal chains
        """
        pred = defaultdict(list)
        for A, B, score in self.causal_pairs:
            pred[B].append((A, score))
        
        chains = []
        queue = deque()
        
        # Initialize with chains directly pointing to target
        for A, score in pred[target_col]:
            queue.append(([A, target_col], [score]))
        
        # BFS to build chains
        while queue:
            path, scores = queue.popleft()
            
            if len(path) > self.max_chain_length:
                continue
            
            last = path[0]
            
            if len(path) == self.max_chain_length or last not in pred:
                chains.append((path, scores))
            else:
                for pre, s in pred[last]:
                    if pre in path:
                        continue
                    queue.appendleft(([pre] + path, [s] + scores))
        
        self.causal_chains = chains
        return chains