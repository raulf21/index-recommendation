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
    ├── workload_parser.py
    ├── candidate_generator.py
    ├── feature_extractor.py
    ├── hypopg_labeler.py
    └── ml_model.py
```

## Team
- [Raul Flores](https://github.com/raulf21)
- [Sindhu Satish ]()