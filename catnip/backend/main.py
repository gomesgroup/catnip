import uuid
import os
import csv
import json
from io import StringIO
import pickle as pkl

import numpy as np
import pandas as pd
from catboost import CatBoostRanker

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

import redis
from rq import Worker, Queue, Connection

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

redis_url = os.environ.get('REDIS_HOST')
conn = redis.from_url(redis_url)

q = Queue(connection=conn)

SUBSTRATE_NEIGHBORS = 10

as_table = pd.read_csv('alignment_scores.csv')
as_table = as_table[as_table['AS %'].notna()]
as_table['AS %'] = as_table['AS %'].map(lambda x: float(x[:-1]) / 100)

with open('pca.pkl', 'rb') as f:
    pca = pkl.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pkl.load(f)

with open('interactions.pkl', 'rb') as f:
    interactions = pkl.load(f)

with open('dataset.pkl', 'rb') as f:
    dataset = pkl.load(f)
    substrate_id2split = {row['Substrate ID']: row['split'] for _, row in dataset.iterrows()}

ranker = CatBoostRanker().load_model('scoring_formula_50_2.cb')


def get_formula(ranker):
    def formula(data):
        return ranker.predict(data.drop(columns=['node 2']))

    return formula


scoring_formula = get_formula(ranker)

enzyme_sequences = pd.read_csv('enzyme_sequences.csv')


class MoleculeData(BaseModel):
    data: str


class EnzymeData(BaseModel):
    sequence: str


class CompatibilityData(BaseModel):
    sequence: str
    substrate: str


@app.post("/submit")
async def submit_data(molecule_data: MoleculeData):
    job = q.enqueue('calculate_features.run_calculations', molecule_data.data.split('.')[0], result_ttl=3600 * 24 * 7)

    job.meta['smiles_full'] = molecule_data.data
    job.meta['smiles'] = molecule_data.data.split('.')[0]
    job.save_meta()

    return {"run_id": job.id}


@app.post("/submit-enzyme")
async def submit_enzyme(enzyme_data: EnzymeData):
    job = q.enqueue('process_enzyme.run_enzyme_analysis', enzyme_data.sequence, result_ttl=3600 * 24 * 7)

    job.meta['sequence'] = enzyme_data.sequence
    job.save_meta()

    return {"run_id": job.id}


def calculate_pca(data, pca):
    return pca.transform(data.drop(columns=['Substrate ID', 'SMILES', 'split']))


substrates_with_interactions = dataset[
    dataset['Substrate ID'].isin(interactions['Substrate = 119'].values)
].reset_index(drop=True)

pca_features = calculate_pca(substrates_with_interactions, pca)


def get_k_neighbors(row, data, k):
    distances = np.linalg.norm(data - row, axis=1)
    print(min(distances))
    print(data[np.argmin(distances)])

    if min(distances) < 0.05:
        offset = 1
        zero_neighbor = np.argmin(distances)
    else:
        offset = 0
        zero_neighbor = None

    neighbors = np.argsort(distances)[offset:k + offset]
    neighbor_distances = distances[neighbors]

    return neighbors, neighbor_distances, offset, zero_neighbor


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


def process_results(output_data, job):
    features = pd.Series(json.loads(job.meta['features']))
    features_transformed = scaler.transform([features[scaler.feature_names_in_]])[0]

    features = pd.Series(features_transformed, index=scaler.feature_names_in_)
    features = features[list(pca.feature_names_in_)]

    substrate_features = pca.transform(pd.DataFrame([features]))[0]

    output_data['pca_weights'] = substrate_features.tolist()

    neighbors, neighbor_distances, offset, zero_neighbor = get_k_neighbors(
        substrate_features, pca_features, SUBSTRATE_NEIGHBORS)
    neighbor_substrate_ids = substrates_with_interactions.iloc[neighbors]['Substrate ID'].values

    output_data['neighbors'] = [f"{sid} ({substrate_id2split[sid]})" for sid in neighbor_substrate_ids]

    results_to_rank = get_candiates(neighbor_substrate_ids, neighbor_distances, interactions, as_table,
                                    pca_features[neighbors], substrate_features)

    results_to_rank['score'] = scoring_formula(results_to_rank)
    results_to_rank = results_to_rank.sort_values(by='score', ascending=False).reset_index(drop=True)
    results_to_rank = results_to_rank.drop_duplicates(subset='node 2').reset_index(drop=True)

    results_to_rank['id'] = np.arange(len(results_to_rank)) + 1
    results_to_rank = results_to_rank[['id', 'node 2', 'score']]

    results_to_rank.columns = ['#', 'Enzyme ID', 'Score']
    results_to_rank = pd.merge(results_to_rank, enzyme_sequences, on='Enzyme ID', how='left')
    output_data['ranking'] = json.loads(results_to_rank.to_json(orient='records'))

    return offset, zero_neighbor


def exact_match(output_data, offset, zero_neighbor):
    if offset == 1:
        output_data['exact_match'] = {
            'found': True,
        }

        interacting_enzymes = \
            interactions[interactions['Substrate = 119'] == substrates_with_interactions.iloc[zero_neighbor][
                'Substrate ID']][['Substrate = 119', 'Enzyme = 163']][
                'Enzyme = 163'].unique()

        sequence_info = enzyme_sequences[enzyme_sequences['Enzyme ID'].isin(interacting_enzymes)]
        output_data['exact_match']['seqs'] = json.loads(sequence_info.to_json(orient='records'))
    else:
        output_data['exact_match'] = {
            'found': False
        }


@app.get("/run/{run_id}")
async def get_run_data(run_id: str):
    job = q.fetch_job(run_id)

    if not job:
        return {"error": "Run ID not found"}

    output_data = process_job(job, run_id)
    return output_data


def process_job(job, run_id):
    if 'features' in job.meta:
        table = json.loads(job.meta['features']).items()
        table = list(table)
    else:
        table = []

    output_data = {
        "run_id": run_id,
        "data": job.meta['ph_corrected_smiles'] if 'ph_corrected_smiles' in job.meta else job.meta['smiles'],
        "overall_status": "in_progress",
        "status": [
            {"name": "Step 1", "description": "Generating 3D geometry",
             "status": "pending",
             "data_type": "molecule"},
            {"name": "Step 2", "description": "Optimizing molecular geometry",
             "status": "pending",
             "data_type": "molecule"},
            {"name": "Step 3", "description": "Calculating MORFEUS descriptors",
             "status": "pending",
             "data_type": "table", "table": table},
            {"name": "Step 4", "description": "Wrapping up",
             "status": "pending",
             "data_type": "nothing"},
        ],
        'exact_match': {
            'found': False
        },
        'neighbors': [],
        'ranking': [],
        'pca_weights': [],
    }

    if "unoptimized_sdf" in job.meta:
        output_data["status"][0]["status"] = "complete"
        output_data["status"][0]["progress"] = 100
        output_data["status"][0]["molecule"] = job.meta["unoptimized_sdf"]

        output_data["status"][1]["status"] = "in_progress"
        output_data["status"][1]["progress"] = -1

    if "optimized_sdf" in job.meta:
        output_data["status"][1]["status"] = "complete"
        output_data["status"][1]["progress"] = 100
        output_data["status"][1]["molecule"] = job.meta["optimized_sdf"]

        output_data["status"][2]["status"] = "in_progress"
        output_data["status"][2]["progress"] = int(round(100 * (len(table) / 21)))

    if len(table) == 21:
        output_data["status"][2]["status"] = "in_progress"
        output_data["status"][2]["progress"] = 100

        output_data["status"][3]["status"] = "in_progress"
        output_data["status"][3]["progress"] = -1

    if job.result:
        output_data["status"][3]["status"] = "complete"
        output_data["status"][3]["progress"] = 100

        offset, zero_neighbor = process_results(output_data, job)
        exact_match(output_data, offset, zero_neighbor)

    return output_data


@app.get("/enzyme-run/{run_id}")
async def get_enzyme_run_data(run_id: str):
    job = q.fetch_job(run_id)

    if not job:
        return {"error": "Run ID not found"}

    output_data = process_enzyme_job(job, run_id)
    return output_data


def process_enzyme_job(job, run_id):
    output_data = {
        "run_id": run_id,
        "sequence": job.meta.get('sequence', ''),
        "overall_status": "in_progress",
        "status": [
            {"name": "Step 1", "description": "Multiple Sequence Alignment",
             "status": "pending"},
            {"name": "Step 2", "description": "Initial retrieval",
             "status": "pending"},
            {"name": "Step 3", "description": "Reranking",
             "status": "pending", "table": []},
            {"name": "Step 4", "description": "Final results",
             "status": "pending"},
        ],
        'substrates': [],
    }

    # Update status based on job meta
    if "msa_completed" in job.meta:
        output_data["status"][0]["status"] = "complete"
        output_data["status"][0]["progress"] = 100

        output_data["status"][1]["status"] = "in_progress"
        output_data["status"][1]["progress"] = -1

    if "retrieval_completed" in job.meta:
        output_data["status"][1]["status"] = "complete"
        output_data["status"][1]["progress"] = 100

        if "retrieval_results" in job.meta:
            output_data["status"][1]["table"] = job.meta["retrieval_results"]

        output_data["status"][2]["status"] = "in_progress"
        output_data["status"][2]["progress"] = -1

    if "reranking_completed" in job.meta:
        output_data["status"][2]["status"] = "complete"
        output_data["status"][2]["progress"] = 100

        if "reranking_results" in job.meta:
            output_data["status"][2]["table"] = job.meta["reranking_results"]

        output_data["status"][3]["status"] = "in_progress"
        output_data["status"][3]["progress"] = -1

    if "analysis_completed" in job.meta:
        output_data["status"][3]["status"] = "complete"
        output_data["status"][3]["progress"] = 100
        output_data["overall_status"] = "done"

        if "substrate_predictions" in job.meta:
            output_data["substrates"] = job.meta["substrate_predictions"]

    return output_data


@app.get("/download/{run_id}/csv")
async def download_csv(run_id: str):
    job = q.fetch_job(run_id)

    if not job:
        raise HTTPException(status_code=404, detail="Run ID not found")

    output_data = process_job(job, run_id)
    if output_data['status'][3]['status'] != 'complete':
        raise HTTPException(status_code=404, detail="Run not complete")

    ranking = output_data['ranking']
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Enzyme ID', 'Score', 'Enzyme Name', 'Organism', 'AA Sequence'])
    for row in ranking:
        writer.writerow([row['Enzyme ID'], row['Score'], row['Enzyme Name'], row['Organism'], row['AA Sequence']])

    output.seek(0)

    return Response(content=output.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=run_{run_id}_features.csv"})


@app.get("/download/enzyme/{run_id}/csv")
async def download_enzyme_csv(run_id: str):
    job = q.fetch_job(run_id)

    if not job or "substrate_predictions" not in job.meta:
        raise HTTPException(status_code=404, detail="Results not found")

    predictions = job.meta["substrate_predictions"]

    # Convert to DataFrame
    df = pd.DataFrame(predictions)

    # Create StringIO buffer
    buffer = StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    # Return CSV as response
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=enzyme_substrates_{run_id}.csv"}
    )
