import utils
import pandas as pd
from catboost import CatBoostRanker, Pool

if __name__ == "__main__":
    train, test = utils.get_train_and_test()
    train, test = utils.standardize(train, test)

    interactions = utils.load_interactions()
    as_table = utils.load_sequence_similarity()

    pca = utils.get_initial_pca(train, path='final_plots/pca_plot_')

    ranker = CatBoostRanker().load_model('../backend/scoring_formula_50_2.cb')
    test_metrics, predictions, fh = utils.get_score(
        test, interactions, as_table, pca, utils.get_formula(ranker), mode='evaluation')
    predictions.to_csv(
        'predictions.csv', index=False
    )

    dataset = utils.get_score(train, interactions, as_table, pca, None, mode='training')
    for i in range(len(dataset)):
        dataset[i]['query'] = i

    dataset = pd.concat(dataset)

    for feature_name, importance in zip(
            dataset.drop(columns=['node 2', 'hit', 'query']).columns, ranker.get_feature_importance(
                Pool(dataset.drop(columns=['node 2', 'hit', 'query']), dataset['hit'],
                     group_id=dataset['query'])
            )):
        test_metrics['f_' + feature_name] = importance

    baseline_metrics, _, fh2 = utils.get_score(
        test, interactions, as_table, pca, utils.get_formula_baseline(), mode='evaluation')

    pd.DataFrame([test_metrics, baseline_metrics]).to_csv('metrics_comparison.csv', index=False)
