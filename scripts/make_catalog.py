"""Generate a catalog.jsonl for the retail RAG project from Instacart data.

Reuses the products.csv / aisles.csv / departments.csv already downloaded for
the langgraph-retail-assistant project to give the MCP product_search and the
RAG agent real, coherent data to retrieve from.
"""
import json
import os
import sys

import pandas as pd

SRC = r"H:\面试项目\langgraph-retail-assistant\dataset"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "catalog.jsonl")


def main():
    products = pd.read_csv(os.path.join(SRC, "products.csv"))
    aisles = pd.read_csv(os.path.join(SRC, "aisles.csv"))
    departments = pd.read_csv(os.path.join(SRC, "departments.csv"))

    products["aisle_id"] = products["aisle_id"].astype(int)
    products["department_id"] = products["department_id"].astype(int)
    merged = products.merge(aisles, on="aisle_id", how="left").merge(
        departments, on="department_id", how="left"
    )

    rows = []
    for _, r in merged.iterrows():
        rows.append(
            {
                "id": str(r["product_id"]),
                "text": (
                    f"{r['product_name']}, found in the {r['aisle']} aisle "
                    f"of the {r['department']} department."
                ),
                "metadata": {
                    "product_id": int(r["product_id"]),
                    "product_name": str(r["product_name"]),
                    "aisle": str(r["aisle"]),
                    "department": str(r["department"]),
                },
            }
        )

    with open(OUT, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} product chunks -> {OUT}")


if __name__ == "__main__":
    main()