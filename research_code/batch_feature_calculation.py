import argparse
import pandas as pd
from joblib import Parallel, delayed
from calculate_features import Substrate  # Make sure that worker.py is in the same directory as this script

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", help="Path to input csv file")
    parser.add_argument("--output_csv", help="Path to output csv file")

    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)


    def process_row(row):
        try:
            substrate = Substrate(row["SMILES"])
            substrate.get_geometry_and_basic_features()
            substrate.calculate_features()

            for feature_name, feature_value in substrate.features.items():
                row[feature_name] = feature_value

        except Exception as e:
            print(f"Error processing {row['SMILES']}, {e}")

        return row


    output_data = Parallel(n_jobs=-1)(delayed(process_row)(row) for _, row in df.iterrows())
    output_data = [row for row in output_data if row is not None]

    output_df = pd.DataFrame(output_data)
    output_df.to_csv(args.output_csv, index=False)
