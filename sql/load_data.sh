#!/bin/bash

echo "Stripping trailing pipes from .tbl files..."
for f in data/*.tbl; do sed -i '' 's/|$//' "$f"; done

echo "Loading schema..."
docker exec -i tpch-db psql -U postgres -d tpch < sql/schema.sql

echo "Loading data..."
docker exec -i tpch-db psql -U postgres -d tpch -c "\copy nation FROM '/data/nation.tbl' DELIMITER '|' CSV;"
docker exec -i tpch-db psql -U postgres -d tpch -c "\copy region FROM '/data/region.tbl' DELIMITER '|' CSV;"
docker exec -i tpch-db psql -U postgres -d tpch -c "\copy part FROM '/data/part.tbl' DELIMITER '|' CSV;"
docker exec -i tpch-db psql -U postgres -d tpch -c "\copy supplier FROM '/data/supplier.tbl' DELIMITER '|' CSV;"
docker exec -i tpch-db psql -U postgres -d tpch -c "\copy partsupp FROM '/data/partsupp.tbl' DELIMITER '|' CSV;"
docker exec -i tpch-db psql -U postgres -d tpch -c "\copy customer FROM '/data/customer.tbl' DELIMITER '|' CSV;"
docker exec -i tpch-db psql -U postgres -d tpch -c "\copy orders FROM '/data/orders.tbl' DELIMITER '|' CSV;"
docker exec -i tpch-db psql -U postgres -d tpch -c "\copy lineitem FROM '/data/lineitem.tbl' DELIMITER '|' CSV;"

echo "Done! Database is ready."