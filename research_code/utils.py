import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'

SUBSTRATE_NEIGHBORS = 10
ENZYME_NEIGHBORS = 10


def get_train_and_test():
    data = pd.read_csv('../data/features.csv')

    wo_empty_rows_x = data[data['sasa_area'].notna()]

    train, test = train_test_split(wo_empty_rows_x, test_size=0.5, random_state=42)

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    return train, test


def standardize(train, test):
    scaler = StandardScaler()
    scaler.fit(train.drop(columns=['Substrate ID', 'SMILES']))

    train_transformed = scaler.transform(train.drop(columns=['Substrate ID', 'SMILES']))
    test_transformed = scaler.transform(test.drop(columns=['Substrate ID', 'SMILES']))

    train_transformed = pd.DataFrame(train_transformed, columns=train.drop(columns=['Substrate ID', 'SMILES']).columns)
    train_transformed['Substrate ID'] = train['Substrate ID']
    train_transformed['SMILES'] = train['SMILES']

    test_transformed = pd.DataFrame(test_transformed, columns=test.drop(columns=['Substrate ID', 'SMILES']).columns)
    test_transformed['Substrate ID'] = test['Substrate ID']
    test_transformed['SMILES'] = test['SMILES']

    return train_transformed, test_transformed


def get_initial_pca(data, path='final_plots/pca_plot_'):
    pca = PCA(n_components=5, random_state=42)

    data_transformed = pca.fit_transform(data.drop(columns=['Substrate ID', 'SMILES']))

    plt.figure(figsize=(5, 5))
    plt.subplot(151)
    plt.title('PC1')
    plt.barh(range(1, 1 + len(pca.components_[0])), pca.components_[0])
    plt.yticks(range(1, 1 + len(pca.components_[0])),
               data.drop(columns=['Substrate ID', 'SMILES']).columns)

    for i in range(1, 5):
        plt.subplot(1, 5, i + 1)
        plt.title(f'PC{i + 1}')
        plt.barh(range(1, 1 + len(pca.components_[i])), pca.components_[i])
        plt.yticks([])

    plt.savefig(
        path + 'components.png',
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

    plt.figure(figsize=(5, 5))
    plt.scatter(*data_transformed.T[:2], s=5)

    plt.xlabel('PC1')
    plt.ylabel('PC2')

    plt.savefig(
        path + 'scatter.png',
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

    return pca


def load_interactions():
    interactions = pd.read_csv('../data/reaction_table.csv')
    return interactions


def load_sequence_similarity():
    as_table = pd.read_csv('../data/alignment_scores.csv')
    as_table = as_table[as_table['AS %'].notna()]
    as_table['AS %'] = as_table['AS %'].map(lambda x: float(x[:-1]) / 100)

    return as_table


def calculate_pca(data, pca):
    return pca.transform(data.drop(columns=['Substrate ID', 'SMILES'])[pca.feature_names_in_])


def get_k_neighbors(row, data, k, offset=0):
    distances = np.linalg.norm(data - row, axis=1)

    neighbors = np.argsort(distances)[offset:k + offset]
    neighbor_distances = distances[neighbors]

    return neighbors, neighbor_distances


def get_candiates(neighbor_substrate_ids, neighbor_distances, interactions, as_table, neighbor_pca, substrate_pca):
    results_to_rank = []

    for neighbor_substrate_id, neighbor_distance, neighbor_pca_ in zip(neighbor_substrate_ids, neighbor_distances,
                                                                       neighbor_pca):
        enzymes_found = interactions[interactions['Substrate = 119'] == neighbor_substrate_id]
        enzymes_found = enzymes_found['Enzyme = 163'].unique()

        expanded_enzymes = []

        for enzyme_id in enzymes_found:
            expanded_enzymes.append((
                enzyme_id,
                as_table[as_table['node 1'] == enzyme_id][
                    ['node 2', 'AS %']
                ].sort_values(by='AS %', ascending=False)[:10]
            ))

        expanded_enzymes = pd.concat([x[1] for x in expanded_enzymes])
        expanded_enzymes['distance'] = neighbor_distance

        for i in range(neighbor_pca.shape[1]):
            expanded_enzymes[f'neighbor_pca_{i}'] = np.linalg.norm(neighbor_pca_[i] - substrate_pca[i])

        results_to_rank.append(expanded_enzymes)

    results_to_rank = pd.concat(results_to_rank)

    return results_to_rank


def get_metrics(hits, total_examples):
    metrics = {'precision': hits.mean(), 'recall': hits.sum() / total_examples,
               'ndcg': ndcg_score([hits], [np.arange(len(hits))[::-1]])}

    metrics['enrichment'] = metrics['precision'] / (total_examples / 314)

    for i in trange(2, 51):
        metrics[f'precision@{i}'] = hits[:i].mean()
        metrics[f'recall@{i}'] = hits[:i].sum() / total_examples
        metrics[f'ndcg@{i}'] = ndcg_score([hits[:i]], [np.arange(len(hits))[:i][::-1]])
        metrics[f'enrichment@{i}'] = metrics[f'precision@{i}'] / (total_examples / 314)

    print(len(hits))
    metrics['rank_first_hit'] = np.where(hits)[0][0] + 1 if np.any(hits) else np.nan
    return metrics


def get_score(dataset, interactions, as_table, pca, scoring_formula, mode='training'):
    substrates_with_interactions = dataset[
        dataset['Substrate ID'].isin(interactions['Substrate = 119'].values)
    ].reset_index(drop=True)

    pca_features = calculate_pca(substrates_with_interactions, pca)

    all_metrics = []

    if mode == 'training':
        training_data = []

    if mode == 'evaluation':
        predictions = []

    for row_id, row in substrates_with_interactions.iterrows():
        substrate_id = row['Substrate ID']
        substrate_features = pca_features[row_id]

        neighbors, neighbor_distances = get_k_neighbors(substrate_features, pca_features, SUBSTRATE_NEIGHBORS, 1)
        neighbor_substrate_ids = substrates_with_interactions.iloc[neighbors]['Substrate ID'].values

        results_to_rank = get_candiates(neighbor_substrate_ids, neighbor_distances, interactions, as_table,
                                        pca_features[neighbors], substrate_features)

        results_to_rank['hit'] = results_to_rank['node 2'].apply(
            lambda x: ((interactions['Enzyme = 163'] == x) & (interactions['Substrate = 119'] == substrate_id)).any()
        )

        if mode == 'evaluation':
            results_to_rank['score'] = scoring_formula(results_to_rank)
            results_to_rank = results_to_rank.sort_values(by='score', ascending=False).reset_index(drop=True)
            results_to_rank = results_to_rank.drop_duplicates(subset='node 2').reset_index(drop=True)

            results_to_rank['Substrate ID'] = substrate_id
            predictions.append(results_to_rank)

            metrics = get_metrics(results_to_rank['hit'].values,
                                  (interactions['Substrate = 119'] == substrate_id).sum())
            all_metrics.append(metrics)

        if mode == 'training':
            results_to_rank['hit'] = results_to_rank['hit'].astype(int)
            training_data.append(results_to_rank)

    if mode == 'training':
        return training_data
    else:
        all_metrics = pd.DataFrame(all_metrics)
        predictions = pd.concat(predictions)
        return all_metrics.mean(), predictions, all_metrics['rank_first_hit'].values


def get_formula(ranker):
    def formula(data):
        return ranker.predict(data.drop(columns=['node 2', 'hit']))

    return formula


def get_formula_baseline():
    def formula(data):
        return - 0.1 * data['distance'] - 0.9 * (1 - data['AS %'] ** 2) ** 2

    return formula
