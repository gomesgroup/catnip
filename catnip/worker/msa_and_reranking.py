import os
import csv
import subprocess
import shutil

from Bio import AlignIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from catboost import CatBoost
from sklearn.preprocessing import MinMaxScaler


def calculate_similarity_for_user(alignment, user_id):
    user_index = None
    labels = [record.id for record in alignment]
    for i, label in enumerate(labels):
        if label == user_id:
            user_index = i
            break

    if user_index is None:
        raise ValueError(f"User sequence ID '{user_id}' not found in alignment.")

    user_seq = alignment[user_index].seq
    similarity_data = []
    for i, record in enumerate(alignment):
        seq2 = record.seq
        if len(user_seq) != len(seq2):
            raise ValueError("Sequences are not aligned properly. Check input.")
        matches = sum(res1 == res2 for res1, res2 in zip(user_seq, seq2))
        similarity = matches / len(user_seq) * 100
        similarity_data.append([user_id, record.id, round(similarity, 2)])  # Round to 2 decimals
    return similarity_data


def do_msa(user_sequence):
    temp_input_fasta = os.path.join("DataPrep/user_sequence.fasta")
    temp_combined_fasta = os.path.join("DataPrep/temp_combined.fasta")
    temp_output_fasta = os.path.join("DataPrep/temp_aligned.fasta")
    prealigned_fasta = os.path.join("DataInput/Enzyme_Sequences.fasta")
    csv_output = os.path.join("UserData/user_similarity_results.csv")
    clustal_exe = "./clustalo"

    user_seq_id = "USER"

    user_record = SeqRecord(Seq(user_sequence), id=user_seq_id, description="User input enzyme")
    with open(temp_input_fasta, "w") as user_fasta:
        SeqIO.write(user_record, user_fasta, "fasta")

    with open(temp_combined_fasta, "w") as combined_fasta:
        with open(prealigned_fasta) as prealigned:
            combined_fasta.write(prealigned.read())
        with open(temp_input_fasta) as user_fasta:
            combined_fasta.write(user_fasta.read())

    clustal_command = f"{clustal_exe} -i \"{temp_combined_fasta}\" -o \"{temp_output_fasta}\" --force -v"
    print(f"Running: {clustal_command}")
    result = subprocess.run(clustal_command, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error running Clustal Omega: {result.stderr}")
    else:
        print("Clustal Omega alignment completed.")

    try:
        alignment = AlignIO.read(temp_output_fasta, "fasta")
        print(f"Alignment read successfully. {len(alignment)} sequences aligned.")
    except FileNotFoundError:
        print(f"Error: Output file {temp_output_fasta} not found.")
        alignment = None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        alignment = None

    if alignment:
        try:
            similarity_data = calculate_similarity_for_user(alignment, user_seq_id)
            print("Saving user similarity results...")

            with open(csv_output, mode="w", newline="") as csv_file:
                writer = csv.writer(csv_file)
                # Write header
                writer.writerow(["Enzyme1", "Enzyme2", "Percent_Similarity"])
                # Write rows
                writer.writerows(similarity_data)

            print(f"User similarity results saved to {csv_output}")
        except Exception as e:
            print(f"Error calculating or saving similarity data frame: {e}")

    os.remove(temp_input_fasta)
    os.remove(temp_combined_fasta)
    os.remove(temp_output_fasta)


def find_enzyme_neighbors(all=False):
    # File paths
    similarity_results_csv = "UserData/user_similarity_results.csv"
    unique_enzymes_csv = "DataPrep/EnzymeID_unique.csv" if not all else "Score_Model/Unique_Enzymes.csv"
    output_csv = "UserData/user_similarity_filtered.csv"

    try:
        # Load user similarity results
        similarity_data = pd.read_csv(similarity_results_csv)
        print(f"Loaded {similarity_results_csv} successfully.")

        # Load unique enzyme IDs
        unique_enzymes = pd.read_csv(unique_enzymes_csv)
        print(f"Loaded {unique_enzymes_csv} successfully.")

        # Extract enzyme IDs from column 1 of EnzymeID_unique.csv
        valid_enzymes = unique_enzymes.iloc[:, 0].tolist()  # Assuming IDs are in the first column

        # Filter rows where Enzyme2 exists in the valid enzymes
        filtered_data = similarity_data[similarity_data["Enzyme2"].isin(valid_enzymes)]

        # Save filtered results to a new CSV file
        filtered_data.to_csv(output_csv, index=False)
        print(f"Filtered data saved to {output_csv}.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def extract_top_10():
    input_csv = "UserData/user_similarity_filtered.csv"
    output_csv = "UserData/UserE_top10E.csv"

    try:
        # Load the filtered similarity results
        similarity_data = pd.read_csv(input_csv)
        print(f"Loaded {input_csv} successfully.")

        # Sort by Percent_Similarity in descending order
        sorted_data = similarity_data.sort_values(by="Percent_Similarity", ascending=False)
        print("Data sorted by Percent_Similarity.")

        # Retrieve the top 10 rows
        top_10_data = sorted_data.head(314)
        print("Top 10 rows retrieved.")

        # Save the top 10 results to a new CSV file
        top_10_data.to_csv(output_csv, index=False)
        print(f"Top 10 similarity results saved to {output_csv}.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def get_neighbor_substrates(all=False):
    topmatches = pd.read_csv('UserData/UserE_top10E.csv')
    min_rxn_table = pd.read_csv(f'DataInput/RxnTable{"_all" if all else ""}.csv')

    # Initialize an empty list to hold the expanded rows
    expanded_rows = []

    # Iterate through each enzyme in the "Enzyme2" column of topmatches
    for index, enzyme in topmatches['Enzyme2'].items():
        # Find matching rows in min_rxn_table
        matching_rows = min_rxn_table[min_rxn_table['Enzyme'] == enzyme]

        # If matches are found, create a new row for each substrate
        if not matching_rows.empty:
            for substrate_id in matching_rows['Substrate_ID']:
                expanded_row = topmatches.loc[index].copy()  # Copy the current row
                expanded_row['E2_Sub'] = substrate_id  # Assign the substrate ID as a value
                expanded_rows.append(expanded_row)

    # Create a new DataFrame from the expanded rows
    expanded_topmatches = pd.DataFrame(expanded_rows)

    # Convert the 'E2_substrates' column to a list of values (if needed)
    expanded_topmatches['E2_Sub'] = expanded_topmatches['E2_Sub'].apply(pd.to_numeric, errors='ignore')

    # Save the expanded DataFrame to a new CSV
    expanded_topmatches.to_csv('UserData/UserE_top10E_E2Sub.csv', index=False)

    # Load the CSV file into a pandas DataFrame
    expanded_topmatches_with_neighbors = pd.read_csv('UserData/UserE_top10E_E2Sub.csv')

    # Remove any percentage signs and convert the AS% column to numeric values
    expanded_topmatches_with_neighbors['Percent_Similarity'] = expanded_topmatches_with_neighbors[
        'Percent_Similarity'].replace('%', '', regex=True)

    # Convert the AS% column to numeric (decimal format)
    expanded_topmatches_with_neighbors['Percent_Similarity'] = pd.to_numeric(
        expanded_topmatches_with_neighbors['Percent_Similarity'], errors='coerce')

    # If needed, divide by 100 to convert percentage values to decimals
    expanded_topmatches_with_neighbors['Percent_Similarity'] = expanded_topmatches_with_neighbors[
                                                                   'Percent_Similarity'] / 100

    # Save the updated DataFrame back to a CSV
    expanded_topmatches_with_neighbors.to_csv('UserData/UserE_top10E_E2Sub.csv', index=False)

    print("Updated file saved as 'UserData/UserE_top10E_E2Sub.csv'.")


def get_substrate_neighbors():
    # Load the CSV file
    data = pd.read_csv('DataInput/Substrates_PCs.csv')

    # Load the RxnTable.csv file and get the valid substrate IDs
    rxn_table = pd.read_csv('DataInput/RxnTable.csv')
    valid_substrate_ids = set(rxn_table['Substrate_ID'])

    # Filter data to keep only rows where Substrate_ID exists in RxnTable's Substrate column
    data = data[data['Substrate_ID'].isin(valid_substrate_ids)]

    # Drop rows with invalid (NaN or infinite) values in PC_0 to PC_4
    valid_columns = ['PC_0', 'PC_1', 'PC_2', 'PC_3', 'PC_4']
    data = data[np.isfinite(data[valid_columns]).all(axis=1)]

    # Check if there are enough data points for nearest neighbor computation
    if data.shape[0] < 2:
        raise ValueError("Not enough valid data points to perform nearest neighbor search.")

    # Extract coordinates (PC_0 to PC_4) and substrate IDs
    coordinates = data[valid_columns].values
    substrate_ids = data['Substrate_ID'].values

    # Build a KDTree for efficient nearest neighbor search
    tree = cKDTree(coordinates)

    # Initialize a list to hold expanded results
    expanded_results = []

    # Find the 300 nearest neighbors for each substrate
    k = min(300, len(coordinates))  # Ensure k doesn't exceed the number of points
    for idx, substrate in enumerate(coordinates):
        # Query the tree for the k nearest neighbors (including itself)
        distances, indices = tree.query(substrate, k=k)

        # Create a new row for each nearest substrate
        for neighbor_idx, distance in zip(indices, distances):
            expanded_results.append({
                'Substrate_ID': substrate_ids[idx],
                'Nearest_Substrate': substrate_ids[neighbor_idx],
                'Distance': distance
            })

    # Convert the expanded results to a DataFrame and save to a CSV
    expanded_df = pd.DataFrame(expanded_results)

    # Calculate the maximum distance
    max_distance = expanded_df['Distance'].max()

    # Normalize distances by dividing each by the maximum distance
    expanded_df['Normalized_Distance'] = expanded_df['Distance'] / max_distance

    # Save the normalized results to a CSV
    expanded_df.to_csv('DataPrep/nearest_substrates_expanded_normalized.csv', index=False)

    print("Results saved to 'nearest_substrates_expanded_normalized.csv'.")


def expand_matches():
    # Load the CSV files into pandas DataFrames
    topmatches_expanded = pd.read_csv('UserData/UserE_top10E_E2Sub.csv')
    nearest_substrates_expanded = pd.read_csv('DataPrep/nearest_substrates_expanded_normalized.csv')

    # Initialize an empty list to hold the expanded rows
    expanded_rows = []

    # Iterate through each row in the topmatches_expanded DataFrame
    for index, row in topmatches_expanded.iterrows():
        # Get the value in the 'Enzyme2_substrates' column (substrate ID)
        substrate_id = row['E2_Sub']

        # Find matching rows in the nearest_substrates_expanded DataFrame
        matching_rows = nearest_substrates_expanded[nearest_substrates_expanded['Substrate_ID'] == substrate_id]

        # If matching rows are found, create a new row for each nearest substrate
        if not matching_rows.empty:
            for _, match in matching_rows.iterrows():
                expanded_row = row.copy()  # Copy the current row
                expanded_row['Enzyme2_neighbor'] = match['Nearest_Substrate']  # Assign nearest substrate
                expanded_row['Distance'] = match['Distance']  # Assign distance
                expanded_rows.append(expanded_row)  # Add the expanded row to the list

    # Create a new DataFrame from the expanded rows
    expanded_topmatches = pd.DataFrame(expanded_rows)

    # Save the expanded DataFrame to a new CSV
    expanded_topmatches.to_csv('UserData/topmatches_expanded_with_neighbors.csv', index=False)

    # Load the CSV file into a pandas DataFrame
    expanded_topmatches_with_neighbors = pd.read_csv('UserData/topmatches_expanded_with_neighbors.csv')

    # Ensure the 'AS%' column is numeric and convert invalid entries to NaN
    expanded_topmatches_with_neighbors['Percent_Similarity'] = pd.to_numeric(
        expanded_topmatches_with_neighbors['Percent_Similarity'], errors='coerce')

    # Check if 'AS%' column has any invalid values (NaN or non-numeric) and handle them
    # Optionally, you can set NaN values to a default value like 100 if it's reasonable in your case
    expanded_topmatches_with_neighbors['Percent_Similarity'].fillna(0, inplace=True)

    # Ensure the 'Distance' column is numeric and convert invalid entries to NaN
    expanded_topmatches_with_neighbors['Distance'] = pd.to_numeric(expanded_topmatches_with_neighbors['Distance'],
                                                                   errors='coerce')

    # Check for NaN values in 'Distance' and fill them with a default value (e.g., 0)
    expanded_topmatches_with_neighbors['Distance'].fillna(1, inplace=True)

    # Calculate the 'Score' as a function of 'Percent_Similarity' and 'Distance'
    # expanded_topmatches_with_neighbors['Score'] = (1.0000001 - expanded_topmatches_with_neighbors['Percent_Similarity']) * expanded_topmatches_with_neighbors['Distance']
    expanded_topmatches_with_neighbors['Score'] = np.sqrt(
        expanded_topmatches_with_neighbors['Distance'] ** 2 +
        (1 - expanded_topmatches_with_neighbors['Percent_Similarity']) ** 2
    )

    # Save the updated DataFrame with the 'Score' column to a new CSV file
    expanded_topmatches_with_neighbors.to_csv('UserData/topmatches_expanded_with_neighbors_and_scores.csv', index=False)

    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv('UserData/topmatches_expanded_with_neighbors_and_scores.csv')

    # Sort the DataFrame by 'node 1' first, then by 'Score' in decreasing order
    df_sorted = df.sort_values(by=['Enzyme1', 'Score'], ascending=[True, True])

    # Save the sorted DataFrame to a new CSV file
    df_sorted.to_csv('UserData/topmatches_expanded_with_neighbors_and_scores_ranked.csv', index=False)

    # Clean up files
    os.remove('UserData/topmatches_expanded_with_neighbors.csv')
    os.remove('UserData/topmatches_expanded_with_neighbors_and_scores.csv')

    print("Sorted file saved as 'topmatches_expanded_with_neighbors_and_scores_ranked.csv'.")


def get_nearest_neighbors_all():
    # Load the CSV file
    data = pd.read_csv('DataInput/Substrates_PCs.csv')

    # Drop rows with invalid (NaN or infinite) values in PC 0 or PC 1
    data = data[data[['PC_0', 'PC_1']].applymap(lambda x: pd.notnull(x) and x != float('inf')).all(axis=1)]

    # Extract coordinates (PC 0 and PC 1) and substrate IDs
    coordinates = data[['PC_0', 'PC_1']].values
    substrate_ids = data['Substrate_ID'].values

    # Build a KDTree for efficient nearest neighbor search
    tree = cKDTree(coordinates)

    # Initialize a list to hold expanded results
    expanded_results = []

    # Find the 10 nearest neighbors for each substrate
    for idx, substrate in enumerate(coordinates):
        # Query the tree for the 11 nearest neighbors (including itself)
        distances, indices = tree.query(substrate, k=100)

        # Exclude itself (distance == 0), keep only the nearest 10
        valid_indices = [(i, d) for i, d in zip(indices, distances)][:100]

        # Create a new row for each nearest substrate
        for neighbor_idx, distance in valid_indices:
            expanded_results.append({
                'Substrate_ID': substrate_ids[idx],
                'Nearest_Substrate': substrate_ids[neighbor_idx],
                'Distance': distance
            })

    # Convert the expanded results to a DataFrame and save to a CSV
    expanded_df = pd.DataFrame(expanded_results)
    expanded_df.to_csv('DataPrep/nearest_substrates_all.csv', index=False)

    print("Results saved to 'nearest_substrates_all.csv'.")


def expand_matches_all():
    topmatches_expanded = pd.read_csv('UserData/UserE_top10E_E2Sub.csv')
    nearest_substrates_expanded = pd.read_csv('DataPrep/nearest_substrates_all.csv')

    expanded_rows = []

    for index, row in topmatches_expanded.iterrows():
        substrate_id = row['E2_Sub']

        matching_rows = nearest_substrates_expanded[nearest_substrates_expanded['Substrate_ID'] == substrate_id]

        if not matching_rows.empty:
            for _, match in matching_rows.iterrows():
                expanded_row = row.copy()  # Copy the current row
                expanded_row['Enzyme2_neighbor'] = match['Nearest_Substrate']  # Assign nearest substrate
                expanded_row['Distance'] = match['Distance']  # Assign distance
                expanded_rows.append(expanded_row)  # Add the expanded row to the list

    expanded_topmatches = pd.DataFrame(expanded_rows)

    expanded_topmatches.to_csv('UserData/topmatches_expanded_with_allneighbors.csv', index=False)

    expanded_topmatches_with_neighbors = pd.read_csv('UserData/topmatches_expanded_with_allneighbors.csv')

    expanded_topmatches_with_neighbors['Percent_Similarity'] = pd.to_numeric(
        expanded_topmatches_with_neighbors['Percent_Similarity'], errors='coerce')

    expanded_topmatches_with_neighbors['Percent_Similarity'].fillna(100, inplace=True)

    expanded_topmatches_with_neighbors['Distance'] = pd.to_numeric(expanded_topmatches_with_neighbors['Distance'],
                                                                   errors='coerce')

    expanded_topmatches_with_neighbors['Distance'].fillna(0, inplace=True)
    expanded_topmatches_with_neighbors['Score'] = np.sqrt(
        expanded_topmatches_with_neighbors['Distance'] ** 2 +
        (1 - expanded_topmatches_with_neighbors['Percent_Similarity']) ** 2
    )

    expanded_topmatches_with_neighbors.to_csv('UserData/topmatches_expanded_with_allneighbors_and_scores.csv',
                                              index=False)

    df = pd.read_csv('UserData/topmatches_expanded_with_allneighbors_and_scores.csv')
    df_sorted = df.sort_values(by=['Enzyme1', 'Score'], ascending=[True, True])
    df_sorted.to_csv('UserData/topmatches_expanded_with_allneighbors_and_scores_ranked.csv', index=False)

    # Clean up files
    os.remove('UserData/topmatches_expanded_with_allneighbors.csv')
    os.remove('UserData/topmatches_expanded_with_allneighbors_and_scores.csv')


def catboost_rerank():
    INPUT_PATH = "UserData/topmatches_expanded_with_allneighbors_and_scores_ranked.csv"

    model = CatBoost().load_model('../revision/inference/2025-02-11_model.cb')
    substrates = pd.read_csv('../revision/inference/Substrates.csv')

    input_data = pd.read_csv(INPUT_PATH)
    input_data['Normalized_Distance'] = MinMaxScaler().fit_transform(input_data[['Distance']])

    input_x = input_data[['Percent_Similarity', 'Normalized_Distance', 'Score']]
    input_data['Predicted'] = model.predict(input_x)

    input_data = input_data.sort_values('Predicted', ascending=False).drop_duplicates(
        subset=['Enzyme2_neighbor']
    )
    input_data = pd.merge(input_data, substrates, left_on='Enzyme2_neighbor', right_on='Substrate_ID', how='left')
    return input_data[['Enzyme2', 'Enzyme2_neighbor', 'Predicted', 'SMILES']]


def main(seq):
    if not os.path.exists('UserData'):
        os.makedirs('UserData')

    do_msa(seq)
    find_enzyme_neighbors()
    extract_top_10()
    get_neighbor_substrates()
    get_substrate_neighbors()
    expand_matches()
    get_nearest_neighbors_all()
    expand_matches_all()

    output = catboost_rerank()
    # delete user data folder
    shutil.rmtree('UserData')

    return output


if __name__ == "__main__":
    print(main('MARTGLLLALLAAGLAG'))
