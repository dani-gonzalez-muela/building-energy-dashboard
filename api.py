# api.py - FastAPI service for building energy predictions
# Run: uvicorn api:app --reload

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import json
from pathlib import Path

app = FastAPI(title="Building Energy API", version="1.0")

# Allow Streamlit to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load predictions on startup
DATA_PATH = Path("predictions.parquet")  # Same folder as api.py

if DATA_PATH.exists():
    df = pd.read_parquet(DATA_PATH)
    print(f"✅ Loaded {len(df)} buildings")
else:
    # Fallback to CSV
    csv_path = Path("predictions.csv")
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} buildings from CSV")
    else:
        raise FileNotFoundError("predictions.parquet or predictions.csv not found")


@app.get("/")
def root():
    return {"status": "ok", "buildings": len(df)}


@app.get("/buildings")
def list_buildings():
    """List all buildings for dropdown"""
    buildings = df[['building_id', 'building_type', 'priority_rank']].to_dict('records')
    return buildings


@app.get("/building/{building_id}")
def get_building(building_id: str):
    """Get single building details + SHAP"""
    row = df[df['building_id'] == building_id]
    
    if row.empty:
        raise HTTPException(status_code=404, detail="Building not found")
    
    record = row.iloc[0].to_dict()
    
    # Parse SHAP JSON if present
    if 'shap_json' in record and record['shap_json']:
        try:
            record['shap_values'] = json.loads(record['shap_json'])
        except:
            record['shap_values'] = {}
    
    return record


@app.get("/summary")
def get_summary():
    """Dashboard summary stats"""
    return {
        "total_buildings": len(df),
        "anomalies": int(df['is_anomaly'].sum()),
        "underperformers": int(df['underperformer'].sum()) if 'underperformer' in df.columns else 0,
        "clusters": df['cluster'].value_counts().to_dict(),
        "top_priority": df.nsmallest(10, 'priority_rank')[
            ['building_id', 'building_type', 'recommendation', 'priority_rank']
        ].to_dict('records')
    }


@app.get("/cluster/{cluster_id}")
def get_cluster(cluster_id: int):
    """Get buildings in a cluster"""
    cluster_df = df[df['cluster'] == cluster_id]
    
    if cluster_df.empty:
        raise HTTPException(status_code=404, detail="Cluster not found")
    
    return {
        "cluster_id": cluster_id,
        "count": len(cluster_df),
        "avg_baseload": float(cluster_df['baseload'].mean()),
        "avg_weekend_ratio": float(cluster_df['weekend_ratio'].mean()) if 'weekend_ratio' in cluster_df.columns else None,
        "buildings": cluster_df['building_id'].tolist()
    }
