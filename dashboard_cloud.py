# dashboard_cloud.py - Streamlit Cloud version (no API needed)
# Loads data directly from parquet file

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
import os

PLOTS_PATH = "plots"

st.set_page_config(
    page_title="Building Energy Insights | DNV", 
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------------------------
# Custom CSS - Clean and elegant
# -----------------------------------------------------------------------------

st.markdown("""
<style>
    /* Make main container wider */
    .block-container {
        max-width: 1200px;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    
    /* Tab content padding */
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Load Data Directly (no API)
# -----------------------------------------------------------------------------

@st.cache(ttl=300, allow_output_mutation=True)
def load_data():
    try:
        df = pd.read_parquet("predictions.parquet")
        return df
    except:
        try:
            df = pd.read_csv("predictions.csv")
            return df
        except:
            return None

df = load_data()

if df is None:
    st.error("Cannot load predictions.parquet or predictions.csv")
    st.stop()

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def get_summary():
    return {
        "total_buildings": len(df),
        "anomalies": int(df['is_anomaly'].sum()),
        "underperformers": int(df['underperformer'].sum()) if 'underperformer' in df.columns else 0,
        "clusters": df['cluster'].value_counts().to_dict(),
        "top_priority": df.nsmallest(10, 'priority_rank')[
            ['building_id', 'building_type', 'recommendation', 'priority_rank']
        ].to_dict('records')
    }

def get_building(building_id):
    row = df[df['building_id'] == building_id]
    if row.empty:
        return None
    record = row.iloc[0].to_dict()
    
    # Parse SHAP JSON
    if 'shap_json' in record and record['shap_json'] and pd.notna(record['shap_json']):
        try:
            record['shap_values'] = json.loads(record['shap_json'])
        except:
            record['shap_values'] = {}
    else:
        record['shap_values'] = {}
    
    return record

def get_cluster(cluster_id):
    cluster_df = df[df['cluster'] == cluster_id]
    if cluster_df.empty:
        return None
    return {
        "cluster_id": cluster_id,
        "count": len(cluster_df),
        "avg_baseload": float(cluster_df['baseload'].mean()),
        "avg_weekend_ratio": float(cluster_df['weekend_ratio'].mean()) if 'weekend_ratio' in cluster_df.columns else None
    }

# -----------------------------------------------------------------------------
# Cluster descriptions
# -----------------------------------------------------------------------------

def get_cluster_description(cluster_id, cluster_stats):
    """Generate actionable cluster description"""
    descriptions = {
        "high_baseload": {
            "name": "🔴 High Baseload (24/7 Operators)",
            "description": "Buildings with consistently high energy use, even during off-hours.",
            "action": "**Action:** Priority targets for energy audits and demand-side management programs."
        },
        "efficient": {
            "name": "🟢 Efficient Buildings",
            "description": "Well-managed buildings with good weekend/night shutdown patterns.",
            "action": "**Action:** Use as benchmarks. Document best practices for other buildings."
        },
        "standard": {
            "name": "🟡 Standard Operations",
            "description": "Typical consumption patterns with some optimization potential.",
            "action": "**Action:** Secondary priority for retro-commissioning programs."
        }
    }
    
    clusters_by_baseload = sorted(cluster_stats.items(), key=lambda x: x[1]['avg_baseload'], reverse=True)
    
    for i, (cid, stats) in enumerate(clusters_by_baseload):
        if int(cid) == cluster_id:
            if i == 0:
                return descriptions["high_baseload"]
            elif i == len(clusters_by_baseload) - 1:
                return descriptions["efficient"]
            else:
                return descriptions["standard"]
    
    return descriptions["standard"]

# -----------------------------------------------------------------------------
# Dashboard Header
# -----------------------------------------------------------------------------

st.title("Building Energy Insights")
st.caption("ML-Powered Energy Analysis | Azure Databricks + MLflow + Streamlit")

summary = get_summary()

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------

tab1, tab2 = st.tabs(["📋 Context & Data", "📊 Portfolio Insights"])

# =============================================================================
# TAB 1: Context & Data
# =============================================================================

with tab1:
    
    # The Problem
    st.header("🎯 The Problem")
    st.markdown("""
    We have **hourly energy consumption data** from **1,500 commercial buildings**. 
    The goal is to **identify red flags** (buildings that waste energy) so energy managers 
    can prioritize them for **demand-side efficiency programs**.
    """)
    
    st.markdown("---")
    
    # The Solution
    st.header("💡 The Solution")
    st.markdown("""
    I built a pipeline with **3 complementary ML models**, then combined their outputs 
    into a single **priority score** to rank which buildings need attention first. 
    Buildings flagged by multiple models rank highest.
    """)
    
    st.markdown("")
    
    # Three models in columns
    col1, col2, col3 = st.columns(3, gap="medium")
    
    with col1:
        st.subheader("📊 K-Means")
        st.markdown("Groups buildings by behavior. Identifies the high consumption cluster (buildings that never shut down).")
    
    with col2:
        st.subheader("🔍 Isolation Forest")
        st.markdown("Detects outliers (buildings with unusual patterns that don't fit any normal group).")
    
    with col3:
        st.subheader("🤖 XGBoost")
        st.markdown("Predicts expected efficiency vs actual. Flags underperformers (buildings doing worse than peers).")
    
    st.markdown("---")
    
    # Data Overview
    st.header("📈 Data Overview")
    
    if os.path.exists(PLOTS_PATH):
        # Summary stats
        if os.path.exists(f"{PLOTS_PATH}/summary_stats.json"):
            with open(f"{PLOTS_PATH}/summary_stats.json") as f:
                stats = json.load(f)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Buildings", summary['total_buildings'])
            col2.metric("Avg Consumption", f"{stats.get('avg_consumption_mean', 0):.0f} kWh")
            col3.metric("Avg Baseload", f"{stats.get('baseload_mean', 0):.0f} kWh")
            col4.metric("Building Types", stats.get('building_types', 0))
            
            st.markdown("")
        
        # 4 plots in 2x2 grid - properly aligned
        col1, col2 = st.columns(2)
        
        with col1:
            if os.path.exists(f"{PLOTS_PATH}/consumption_distribution.png"):
                st.image(f"{PLOTS_PATH}/consumption_distribution.png", use_column_width=True)
                st.caption("Consumption Distribution")
            
            if os.path.exists(f"{PLOTS_PATH}/weekend_vs_weekday.png"):
                st.image(f"{PLOTS_PATH}/weekend_vs_weekday.png", use_column_width=True)
                st.caption("Weekend vs Weekday Patterns")
        
        with col2:
            if os.path.exists(f"{PLOTS_PATH}/consumption_by_type.png"):
                st.image(f"{PLOTS_PATH}/consumption_by_type.png", use_column_width=True)
                st.caption("Consumption by Building Type")
            
            if os.path.exists(f"{PLOTS_PATH}/correlation_heatmap.png"):
                st.image(f"{PLOTS_PATH}/correlation_heatmap.png", use_column_width=True)
                st.caption("Feature Correlations")
    else:
        st.info(f"Plots folder not found at `{PLOTS_PATH}/`. Add a plots/ folder with EDA images.")

# =============================================================================
# TAB 2: Portfolio Insights
# =============================================================================

with tab2:
    
    # Portfolio Summary
    st.header("Portfolio Summary")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Buildings", summary['total_buildings'])
    col2.metric("Anomalies (Isolation Forest)", summary['anomalies'])
    col3.metric("Underperformers (XGBoost)", summary['underperformers'])
    
    st.markdown("---")
    
    # -------------------------------------------------------------------------
    # Cluster Analysis
    # -------------------------------------------------------------------------
    st.header("Cluster Analysis")
    st.caption("Buildings grouped by consumption patterns using K-Means clustering")
    
    # Get cluster details
    cluster_stats = {}
    for cluster_id in summary['clusters'].keys():
        cluster_data = get_cluster(int(cluster_id))
        if cluster_data:
            cluster_stats[cluster_id] = cluster_data
    
    # Display each cluster
    cols = st.columns(len(summary['clusters']))
    
    for i, (cluster_id, count) in enumerate(summary['clusters'].items()):
        with cols[i]:
            stats = cluster_stats.get(cluster_id, {})
            desc = get_cluster_description(int(cluster_id), cluster_stats)
            
            st.subheader(desc["name"])
            st.metric("Buildings", count)
            
            if stats:
                st.write(f"**Avg Baseload:** {stats.get('avg_baseload', 0):.1f} kWh")
                st.write(f"**Avg Weekend Ratio:** {stats.get('avg_weekend_ratio', 0):.2f}")
            
            st.write(desc["description"])
            st.info(desc["action"])
    
    st.markdown("---")
    
    # -------------------------------------------------------------------------
    # Top Priority Buildings - centered table
    # -------------------------------------------------------------------------
    st.header("Top Priority Buildings")
    st.caption("These buildings have the highest combination of anomaly flags, efficiency gaps, and baseload.")
    
    priority_buildings = summary['top_priority']
    priority_df = pd.DataFrame(priority_buildings)
    
    # Center the dataframe
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.dataframe(priority_df)
    
    # Building Deep Dive
    st.markdown("")
    st.header("Building Deep Dive")
    st.caption("Select a building to understand why it was flagged")
    
    if priority_buildings:
        building_options = {
            f"#{b['priority_rank']} - {b['building_id']} ({b['building_type']})": b['building_id'] 
            for b in priority_buildings
        }
        
        selected = st.selectbox(
            "Select a priority building:",
            options=list(building_options.keys())
        )
        
        if selected:
            building_id = building_options[selected]
            building = get_building(building_id)
            
            if building:
                st.markdown("---")
                
                # Recommendation as the main callout
                recommendation = building.get('recommendation', 'Conduct energy audit to identify optimization opportunities.')
                st.success(f"**Recommendation:** {recommendation}")
                
                # Get cluster data for comparison
                cluster_id = building.get('cluster')
                cluster_data = get_cluster(int(cluster_id)) if cluster_id is not None else None
                weekend_ratio = building.get('weekend_ratio')
                
                # Building vs Cluster Comparison
                st.subheader("How does this building compare to its peers?")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**This Building**")
                    st.metric("Avg Consumption", f"{building.get('avg_consumption', 0):.1f} kWh")
                    st.metric("Baseload", f"{building.get('baseload', 0):.1f} kWh")
                    if weekend_ratio:
                        st.metric("Weekend Ratio", f"{weekend_ratio:.2f}")
                
                with col2:
                    st.write(f"**Cluster {cluster_id} Average**")
                    if cluster_data:
                        st.metric("Avg Consumption", f"{cluster_data.get('avg_baseload', 0):.1f} kWh", 
                                  delta=f"{building.get('avg_consumption', 0) - cluster_data.get('avg_baseload', 0):.1f}")
                        st.metric("Baseload", f"{cluster_data.get('avg_baseload', 0):.1f} kWh",
                                  delta=f"{building.get('baseload', 0) - cluster_data.get('avg_baseload', 0):.1f}")
                        if cluster_data.get('avg_weekend_ratio'):
                            st.metric("Weekend Ratio", f"{cluster_data.get('avg_weekend_ratio', 0):.2f}",
                                      delta=f"{(weekend_ratio or 0) - cluster_data.get('avg_weekend_ratio', 0):.2f}")
                    else:
                        st.write("Cluster data not available")
                
                # SHAP Explanation
                shap_values = building.get('shap_values', {})
                
                if shap_values:
                    st.markdown("---")
                    st.subheader("Why was this building flagged?")
                    st.caption("The chart below shows which factors contributed most to the model's prediction. Longer bars = bigger impact.")
                    
                    sorted_shap = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
                    features = [x[0].replace('_', ' ').title() for x in sorted_shap]
                    values = [x[1] for x in sorted_shap]
                    colors = ['#ef4444' if v > 0 else '#22c55e' for v in values]
                    
                    fig = go.Figure(go.Bar(
                        x=values,
                        y=features,
                        orientation='h',
                        marker_color=colors
                    ))
                    fig.update_layout(
                        xaxis_title="Impact on prediction",
                        yaxis_title="",
                        height=max(300, len(features) * 35),
                        margin=dict(l=10, r=10, t=10, b=40),
                        yaxis=dict(autorange="reversed"),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                    )
                    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e2e8f0')
                    fig.update_yaxes(showgrid=False)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.caption("🔴 Red = pushes toward worse efficiency | 🟢 Green = pushes toward better efficiency")

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------

st.markdown("---")
st.caption("Built with Azure Databricks • MLflow • FastAPI • Streamlit | Dani Muela © 2025")
