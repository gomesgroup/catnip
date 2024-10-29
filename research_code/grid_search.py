import utils
from tqdm import tqdm
import pandas as pd
from catboost import CatBoostRanker, Pool

if __name__ == "__main__":
    train, test = utils.get_train_and_test()
    train, test = utils.standardize(train, test)

    interactions = utils.load_interactions()
    as_table = utils.load_sequence_similarity()

    pca = utils.get_initial_pca(train, path='final_plots/pca_plot_')

    dataset = utils.get_score(train, interactions, as_table, pca, None, mode='training')
    for i in range(len(dataset)):
        dataset[i]['query'] = i

    dataset = pd.concat(dataset)

    iterations = [10, 50, 100, 500, 1000]
    depths = [1, 2, 5, 6, 8]

    all_metrics = []

    baseline_metrics, _, fh = utils.get_score(
        test, interactions, as_table, pca, utils.get_formula_baseline(), mode='evaluation')

    for iteration in tqdm(iterations):
        for depth in tqdm(depths):
            ranker = CatBoostRanker(iterations=iteration, depth=depth, loss_function='YetiRank')
            ranker.fit(dataset.drop(columns=['node 2', 'hit', 'query']), dataset['hit'], group_id=dataset['query'])
            ranker.save_model(f'for_backend/scoring_formula_{iteration}_{depth}.cb')

            test_metrics, _, _ = utils.get_score(
                test, interactions, as_table, pca, utils.get_formula(ranker), mode='evaluation')
            test_metrics -= baseline_metrics
            test_metrics['iterations'] = iteration
            test_metrics['depth'] = depth

            for feature_name, importance in zip(
                    dataset.drop(columns=['node 2', 'hit', 'query']).columns, ranker.get_feature_importance(
                        Pool(dataset.drop(columns=['node 2', 'hit', 'query']), dataset['hit'],
                             group_id=dataset['query'])
                    )):
                test_metrics['f_' + feature_name] = importance

            all_metrics.append(test_metrics)

    all_metrics = pd.DataFrame(all_metrics)
    all_metrics.to_csv('final_plots/metrics.csv', index=False)
