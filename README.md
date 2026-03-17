# Index Recommendation System
Learning-based index recommendation via workload-aware ranking on PostgreSQL + TPC-H.

## Prerequisites
- Python 3.8+

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
Create a `.env` file in the root of the project (never commit this):
```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=tpch
DB_USER=postgres
DB_PASSWORD=yourpassword
```
Update `DB_USER` and `DB_PASSWORD` to match your local Postgres setup.


## Project Structure
```
index-recommendation/
├── .env                  ← your local config, never committed
├── .gitignore
├── README.md
├── requirements.txt
├── data/                 ← TPC-H .tbl files, never committed
├── notebooks/            ← exploratory analysis
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
- [Your Name](https://github.com/your-username)
- [Teammate Name](https://github.com/teammate-username)