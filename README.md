# Index Recommendation System
Learning-based index recommendation via workload-aware ranking on PostgreSQL + TPC-H.

## Prerequisites
- Python 3.8+
- Docker

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/raulf21/index-recommendation.git
cd index-recommendation
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up your .env file
Copy the example file and fill in your own values:
```bash
cp .env.example .env
```
Update `DB_USER` and `DB_PASSWORD` to whatever you want your local credentials to be.

### 5. Start the database
Make sure Docker Desktop is running, then:
```bash
docker-compose up -d
```

This spins up Postgres 16 with HypoPG already installed. To stop it:
```bash
docker-compose down
```
### 6. Generate and load TPC-H data

Clone and build the data generator:
```bash
git clone https://github.com/electrum/tpch-dbgen.git
cd tpch-dbgen
make
./dbgen -s 1
```

Move the generated files into the data folder:
```bash
mv *.tbl ../data/
cd ..
```

Load the data into the database:
```bash
./sql/load_data.sh
```

Verify it worked:
```bash
docker exec -i tpch-db psql -U postgres -d tpch -c "SELECT COUNT(*) FROM lineitem;"
```

You should see `6001215` rows.
### 7. Run the full pipeline
From the repo root, run the scripts below in order:

#### A) Generate HypoPG labels
```bash
python3 src/hypopg_labeler.py
```

#### B) Build train/val/test dataset
```bash
python3 src/training_dataset.py --labels data/labels.csv
```

#### C) Train model
- Full grid search + final refit on train+val:
```bash
python3 src/ml_model.py --train
```
- Skip grid search and train with `TUNED_XGB_PARAMS`:
```bash
python3 src/ml_model.py --train --no-grid-search
```
- Reproducible training (fixed seed + single-thread training):
```bash
python3 src/ml_model.py --train --seed 42 --reproducible
```

#### D) Print top-k recommendations
```bash
python3 src/ml_model.py --recommend --top-k 10
```

#### E) Optional physical evaluation with real indexes
Create top-k physical indexes, compare planner costs before/after, then drop eval indexes:
```bash
python3 src/evaluate_indexes.py --top-k 5 --drop-after
```

To inspect index impact without creating indexes:
```bash
python3 src/evaluate_indexes.py --top-k 5 --dry-run
```




## Project Structure
```
index-recommendation/
├── .env                  ← your local config, never committed
├── .env.example          ← template, safe to commit
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── init.sql              ← enables HypoPG on startup
├── README.md
├── requirements.txt
├── data/                 ← TPC-H .tbl files, never committed
├── notebooks/
├── queries/              ← TPC-H SQL query files (q1.sql - q22.sql)
├── sql/
│   └── schema.sql
│   └── load_data.sh      ← loads TPC-H data into Docker
└── src/
    ├── db_utils.py               ← shared connection and DB logic
    ├── workload_parser.py        ← extracts AST query parameters 
    ├── candidate_generator.py    ← heuristic & cost-aware pruning
    ├── feature_extractor.py      ← fetches optimizer costs & physical table stats
    ├── hypopg_labeler.py         ← virtual index simulation via HypoPG
    ├── training_dataset.py       ← dataset weaver & log1p scaler
    └── ml_model.py               ← XGBoost training and recommendation engine
```

## Team
- [Raul Flores](https://github.com/raulf21)
- [Sindhu Satish ]()