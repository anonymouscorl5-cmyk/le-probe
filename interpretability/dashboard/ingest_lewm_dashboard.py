import json
import os
import subprocess
from pathlib import Path


def run_sql_batch(sqls):
    batch_file = Path("ingest_batch.sql")
    with open(batch_file, "w") as f:
        f.write("\n".join(sqls))

    # Copy to container and run
    subprocess.run(
        f"docker cp {batch_file} neuronpedia-postgres-1:/tmp/ingest_batch.sql",
        shell=True,
        check=True,
    )
    subprocess.run(
        f"docker exec -i neuronpedia-postgres-1 psql -U postgres -d postgres -f /tmp/ingest_batch.sql",
        shell=True,
        check=True,
    )

    batch_file.unlink()


def ingest():
    dashboard_path = Path(__file__).parent / "lewm_dashboard.json"
    if not dashboard_path.exists():
        print(f"❌ Dashboard file not found at {dashboard_path}")
        return

    with open(dashboard_path, "r") as f:
        dashboard = json.load(f)

    model_id = dashboard["model_name"]
    layer_id = dashboard["layer"]
    creator_id = "clkht01d40000jv08hvalcvly"

    print(f"🚀 Ingesting {len(dashboard['features'])} features for {model_id}...")

    all_sqls = []
    for feature in dashboard["features"]:
        f_idx = str(feature["feature_index"])

        # 1. Neuron
        all_sqls.append(f"""
        INSERT INTO "Neuron" ("modelId", "layer", "index", "creatorId", "createdAt")
        VALUES ('{model_id}', '{layer_id}', '{f_idx}', '{creator_id}', NOW())
        ON CONFLICT ("modelId", "layer", "index") DO NOTHING;
        """)

        # 2. Activations
        for i, act in enumerate(feature["activations"]):
            act_id = f"{model_id}_{layer_id}_{f_idx}_{i}"
            token = act["token"]
            val = act["value"]
            tokens_pg = "{" + f'"{token}"' + "}"
            values_pg = "{" + str(val) + "}"

            all_sqls.append(f"""
            INSERT INTO "Activation" 
            ("id", "modelId", "layer", "index", "tokens", "values", "maxValue", "maxValueTokenIndex", "minValue", "creatorId", "createdAt")
            VALUES 
            ('{act_id}', '{model_id}', '{layer_id}', '{f_idx}', '{tokens_pg}', '{values_pg}', {val}, 0, 0, '{creator_id}', NOW())
            ON CONFLICT (id) DO NOTHING;
            """)

    # Run in batches of 1000 SQL statements
    batch_size = 1000
    for i in range(0, len(all_sqls), batch_size):
        print(f"  📦 Running batch {i//batch_size + 1}...")
        run_sql_batch(all_sqls[i : i + batch_size])

    print("✨ Ingestion complete!")


if __name__ == "__main__":
    ingest()
