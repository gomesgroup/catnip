from redis import Redis
from rq import get_current_job
from msa_and_reranking import do_msa, find_enzyme_neighbors, extract_top_10, get_neighbor_substrates, get_substrate_neighbors, expand_matches, get_nearest_neighbors_all, expand_matches_all, catboost_rerank


def run_enzyme_analysis(sequence):
    job = get_current_job()
    
    do_msa(sequence)
    job.meta['msa_completed'] = True
    job.save_meta()
    
    find_enzyme_neighbors()
    extract_top_10()
    
    job.meta['retrieval_results'] = []
        
    job.meta['retrieval_completed'] = True
    job.save_meta()
    
    get_neighbor_substrates()
    get_substrate_neighbors()
    expand_matches()
    get_nearest_neighbors_all()
    expand_matches_all()
    final_results_df = catboost_rerank()
    
    job.meta['reranking_results'] = []
        
    job.meta['reranking_completed'] = True
    job.save_meta()
    
    try:
        substrate_predictions = []
        
        for _, row in final_results_df.head(10).iterrows():
            substrate_predictions.append({
                "smiles": row['SMILES'],
                "enzyme_neighbor": row['Enzyme2'],
                "substrate": row['Enzyme2_neighbor'],
                "substrate_neighbor": row['Enzyme2_neighbor'],
                "score": row['Predicted'] 
            })
            
        job.meta['substrate_predictions'] = substrate_predictions
    except Exception as e:
        print(f"Error reading substrate predictions: {e}")

        job.meta['substrate_predictions'] = []
    
    job.meta['analysis_completed'] = True
    job.save_meta()
    
    return {"status": "completed"}

