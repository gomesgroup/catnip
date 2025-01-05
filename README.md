# 🌿 CATNIP

## Generation of connections between protein sequence space and chemical space to enable a predictive model for biocatalysis
The application of biocatalysis in synthesis has the potential to offer dramatically streamlined routes toward target molecules, exquisite and tunable catalyst-controlled selectivity, as well as more sustainable processes. Despite these advantages, biocatalytic synthetic strategies can be high risk to implement. Successful execution of these approaches requires identifying an enzyme capable of performing chemistry on a specific intermediate in a synthesis which often calls for extensive screening of enzymes and protein engineering. Strategies for predicting which enzyme is most likely to be compatible with a given small molecule have been hindered by the lack of well-studied biocatalytic reactions. The under exploration of connections between chemical and protein sequence spaces constrains navigation between these two landscapes. Herein, this longstanding challenge is overcome in a two-phase effort relying on high throughput experimentation to populate connections between substrate chemical space and biocatalyst sequence space, and the subsequent development of machine learning models that enable the navigation between these two landscapes. Using a curated library of α-ketoglutarate-dependent non-heme iron (NHI) enzymes, the <code>BioCatSet1</code> dataset was generated to capture the reactivity of each biocatalyst with >100 substrates. In addition to the discovery of novel chemistry, <code>BioCatSet1</code> was leveraged to develop a predictive workflow that provides a ranked list of enzymes that have the greatest compatibility with a given substrate. To make this tool accessible to the community, we built <b>CATNIP</b>, an open access web interface to our predictive workflows. We anticipate our approach can be readily expanded to additional enzyme and transformation classes, and will derisk the application of biocatalysis in chemical synthesis.

https://catnip.cheme.cmu.edu

## Data availability
Data to run the app (including trained model) is available at https://huggingface.co/gomesgroup/catnip/tree/main. Additional files to reproduce model training are available in the repository.

## How to run the code?
### Setting up the environemnt
```bash
conda create -n catnip autode==1.4 xtb xtb-python rq=1.15 python=3.10 rdkit=2022.09.5 -c conda-forge
conda activate catnip
pip install morfeus-ml
pip install joblib
pip install dimorphite-dl

git clone https://github.com/gomesgroup/catnip.git
cd catnip
export PYTHONPATH="catnip/worker"
python research_code/batch_feature_calculation.py --input_csv data/subsrates_quick_test.csv --output_csv features.csv
```

### Calculate MORFEUS features
Run the following command, replacing the paths:
```bash
python research_code/batch_feature_calculation.py --input_csv substrates.csv --output_csv features.csv
```
You can expect this to take up to an hour to run. Feel free to use `data/subsrates_quick_test.csv` for testing.

### Run grid search
From `research_code` directory:
```bash
python grid_search.py
```
It will take up to 15 minutes to run.

### Running evaluation
From `research_code` directory:
```bash
python evaluate.py
```

### Running on your data
You can modify input files to add your own data:
1. Add your substrates to `substrates.csv`
2. Run the feature calculation script
3. Add known reactions to `reaction_table.csv`
4. Retrain the model

## Dependencies
### Model training
The code was run in an environment with Python 3.9.13, Catboost 1.2.2, NumPy 1.24.4. However, we expect these to work with most installations of these with Python 3.

### Feature calculation
This requires autodE (1.4.0 in our case), dimorphite-dl (1.3.2) and xtb (6.5.1) to run.
