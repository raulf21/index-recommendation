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
├── sql/
│   └── schema.sql
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