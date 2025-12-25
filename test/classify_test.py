import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier
import random
import warnings

warnings.filterwarnings('ignore')


def extract_chains(chain_file):
    chains = []
    with open(chain_file, 'r', encoding='utf-8') as f:
        for line in f:
            if '因果链:' in line:
                chain_part = line.split('因果链:')[1].split(',')[0]
                nodes = [node.strip() for node in chain_part.split('->')]
                chain_vars = []
                for node in nodes:
                    if '=' in node:
                        var = node.split('=')[0]
                    else:
                        var = node
                    chain_vars.append(var)
                chains.append(chain_vars)
    return chains


def load_data(data_file, exclude_columns=None):
    df = pd.read_csv(data_file)

    default_exclude = []

    if exclude_columns is None:
        exclude_columns = default_exclude
    else:
        exclude_columns = list(set(default_exclude + exclude_columns))

    feature_columns = [col for col in df.columns if col not in exclude_columns and col != 'is_hot']

    X = df[feature_columns]
    y = df['is_hot']
    return X, y, df


def build_chain_features(X, chains):
    X_chain = X.copy()
    for idx, chain in enumerate(chains):
        chain_vars = [v for v in chain if v in X.columns]
        if len(chain_vars) < 2:
            continue
        prod_feat = X[chain_vars].prod(axis=1)
        X_chain[f'chain{idx + 1}_prod'] = prod_feat
        sum_feat = X[chain_vars].sum(axis=1)
        X_chain[f'chain{idx + 1}_sum'] = sum_feat
    return X_chain


def evaluate_xgboost(X, y, feature_set, random_state=42):
    feature_set = [f for f in feature_set if f in X.columns]
    X_sel = X[feature_set]
    X_train, X_test, y_train, y_test = train_test_split(
        X_sel, y, test_size=0.3, random_state=random_state, stratify=y
    )

    clf = XGBClassifier(random_state=random_state, eval_metric='logloss')
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    return {
        'accuracy': acc,
        'auc': auc
    }


def summarize_results(all_results):
    summary = {}

    for scenario, metrics in all_results.items():
        summary[scenario] = {
            'accuracy': metrics['accuracy'],
            'auc': metrics['auc']
        }

    return summary


if __name__ == "__main__":
    # REPLACE WITH YOUR ACTUAL FILE PATHS
    chain_top5_file = '/path/to/your/final_top5_chains.txt'
    filter_data_file = '/path/to/your/filtered_data.csv'
    all_data_file = '/path/to/your/all_data.csv'

    X_filter, y_filter, df_filter = load_data(filter_data_file)
    filter_factors = list(X_filter.columns)

    exclude_problematic = [
    ]
    X_all, y_all, df_all = load_data(all_data_file, exclude_columns=exclude_problematic)
    all_factors = list(X_all.columns)

    chains = extract_chains(chain_top5_file)
    X_chain = build_chain_features(X_filter, chains)
    chain_features = [col for col in X_chain.columns if col not in X_filter.columns]

    random.seed(1)
    if len(filter_factors) <= len(all_factors):
        random_factors = random.sample(all_factors, len(filter_factors))
    else:
        random_factors = all_factors.copy()

    all_results = {}

    results3 = evaluate_xgboost(X_all, y_all, random_factors)
    all_results['_Random_Factors'] = results3

    chain_factors = sorted(set(v for lst in chains for v in lst))
    chain_factors_in_data = [f for f in chain_factors if f in X_filter.columns]

    if len(chain_factors_in_data) > 0:
        results5 = evaluate_xgboost(X_filter, y_filter, chain_factors_in_data)
        all_results['_Top5_Chain_Factors'] = results5
    else:
        all_results['_Top5_Chain_Factors'] = {'accuracy': 0.0, 'auc': 0.0}

    if len(chain_factors_in_data) > 0:
        chain_factors_plus_chain = chain_factors_in_data + chain_features
        results6 = evaluate_xgboost(X_chain, y_filter, chain_factors_plus_chain)
        all_results['_Top5_Chain_Factors_Chain'] = results6
    else:
        all_results['_Top5_Chain_Factors_Chain'] = {'accuracy': 0.0, 'auc': 0.0}

    summary = summarize_results(all_results)
    
    print("XGBoost Performance Summary:")
    print("=" * 60)
    for scenario, metrics in summary.items():
        print(f"{scenario:35} | Accuracy: {metrics['accuracy']:.4f} | AUC: {metrics['auc']:.4f}")