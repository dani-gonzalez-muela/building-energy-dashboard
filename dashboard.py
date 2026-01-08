# dashboard.py - Streamlit dashboard for building energy insights
# Run: streamlit run dashboard.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import os

# Config
API_URL = "http://localhost:8000"
PLOTS_PATH = "plots"  # Folder with EDA images

st.set_page_config(page_title="Building Energy Dashboard", layout="wide")

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

@st.cache(ttl=60, allow_output_mutation=True)
def get_summary():
    try:
        return requests.get(f"{API_URL}/summary").json()
    except:
        return None

@st.cache(ttl=60, allow_output_mutation=True)
def get_buildings():
    try:
        return requests.get(f"{API_URL}/buildings").json()
    except:
        return []

def get_building(building_id):
    try:
        return requests.get(f"{API_URL}/building/{building_id}").json()
    except:
        return None

def get_cluster(cluster_id):
    try:
        return requests.get(f"{API_URL}/cluster/{cluster_id}").json()
    except:
        return None

# -----------------------------------------------------------------------------
# Cluster descriptions (based on your model results)
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
    
    # Determine cluster type based on baseload (you can adjust thresholds)
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
# Dashboard
# -----------------------------------------------------------------------------

st.title("🏢 Building Energy Insights")

# Check API connection
summary = get_summary()
if not summary:
    st.error(f"❌ Cannot connect to API at {API_URL}")
    st.info("Make sure to run: `uvicorn api:app --reload`")
    st.stop()

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------

tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Data Exploration", "🔍 Building Explorer"])

# =============================================================================
# TAB 1: Overview
# =============================================================================

with tab1:
    # Summary metrics
    st.header("Portfolio Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Buildings", summary['total_buildings'])
    col2.metric("Anomalies", summary['anomalies'], delta=f"{summary['anomalies']/summary['total_buildings']*100:.0f}%")
    col3.metric("Underperformers", summary['underperformers'])
    col4.metric("Clusters", len(summary['clusters']))
    
    st.markdown("---")
    
    # -------------------------------------------------------------------------
    # Cluster Analysis with Descriptions
    # -------------------------------------------------------------------------
    st.header("🎯 Cluster Analysis")
    
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
    # Priority Buildings
    # -------------------------------------------------------------------------
    st.header("🎯 Top Priority Buildings")
    st.caption("These buildings have the highest combination of anomaly flags, efficiency gaps, and baseload.")
    
    priority_df = pd.DataFrame(summary['top_priority'])
    st.dataframe(priority_df)

# =============================================================================
# TAB 2: Data Exploration (EDA)
# =============================================================================

with tab2:
    st.header("📈 Data Exploration")
    st.caption("Understanding the dataset before diving into model results.")
    
    # Check if plots exist
    if os.path.exists(PLOTS_PATH):
        col1, col2 = st.columns(2)
        
        with col1:
            if os.path.exists(f"{PLOTS_PATH}/consumption_distribution.png"):
                st.subheader("Consumption Distribution")
                st.image(f"{PLOTS_PATH}/consumption_distribution.png")
                st.caption("Most buildings cluster in lower consumption ranges, with some high-energy outliers.")
            
            if os.path.exists(f"{PLOTS_PATH}/consumption_by_type.png"):
                st.subheader("Consumption by Building Type")
                st.image(f"{PLOTS_PATH}/consumption_by_type.png")
                st.caption("Building type significantly impacts energy consumption patterns.")
        
        with col2:
            if os.path.exists(f"{PLOTS_PATH}/weekend_vs_weekday.png"):
                st.subheader("Weekend vs Weekday")
                st.image(f"{PLOTS_PATH}/weekend_vs_weekday.png")
                st.caption("Gap between weekday and weekend consumption indicates shutdown efficiency.")
            
            if os.path.exists(f"{PLOTS_PATH}/baseload_vs_peak.png"):
                st.subheader("Baseload vs Peak Ratio")
                st.image(f"{PLOTS_PATH}/baseload_vs_peak.png")
                st.caption("Buildings with high baseload AND high peak ratio are priority targets.")
        
        # Full width for correlation
        if os.path.exists(f"{PLOTS_PATH}/correlation_heatmap.png"):
            st.subheader("Feature Correlations")
            st.image(f"{PLOTS_PATH}/correlation_heatmap.png")
            st.caption("Strong correlations help identify which features drive consumption patterns.")
        
        # Summary stats
        if os.path.exists(f"{PLOTS_PATH}/summary_stats.json"):
            import json
            with open(f"{PLOTS_PATH}/summary_stats.json") as f:
                stats = json.load(f)
            
            st.subheader("📊 Key Statistics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Avg Consumption", f"{stats.get('avg_consumption_mean', 0):.1f} kWh")
            col2.metric("Avg Baseload", f"{stats.get('baseload_mean', 0):.1f} kWh")
            col3.metric("Building Types", stats.get('building_types', 0))
    else:
        st.warning(f"📁 Plots folder not found at `{PLOTS_PATH}/`")
        st.info("Run the EDA stage in Databricks and download the plots folder.")

# =============================================================================
# TAB 3: Building Explorer (Priority Buildings Only)
# =============================================================================

with tab3:
    st.header("🔍 Building Explorer")
    st.caption("Explore details for priority buildings identified by the models.")
    
    # Only show priority buildings
    priority_buildings = summary['top_priority']
    
    if not priority_buildings:
        st.warning("No priority buildings found.")
        st.stop()
    
    building_options = {
        f"#{b['priority_rank']} - {b['building_id']} ({b['building_type']})": b['building_id'] 
        for b in priority_buildings
    }
    
    selected = st.selectbox("Select a priority building:", options=list(building_options.keys()))
    
    if selected:
        building_id = building_options[selected]
        building = get_building(building_id)
        
        if building:
            # Get cluster stats for comparison
            cluster_id = building.get('cluster')
            cluster_data = get_cluster(cluster_id) if cluster_id is not None else None
            
            # -------------------------------------------------------------------------
            # Building vs Cluster Comparison
            # -------------------------------------------------------------------------
            st.subheader("📊 Building vs Cluster Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**This Building**")
                st.metric("Avg Consumption", f"{building.get('avg_consumption', 0):.1f} kWh")
                st.metric("Baseload", f"{building.get('baseload', 0):.1f} kWh")
                weekend_ratio = building.get('weekend_ratio')
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
            
            st.markdown("---")
            
            # -------------------------------------------------------------------------
            # Classification & Flags
            # -------------------------------------------------------------------------
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Classification")
                st.write(f"**Cluster:** {building.get('cluster')}")
                st.write(f"**Anomaly:** {'🚨 Yes' if building.get('is_anomaly') else '✅ No'}")
                st.write(f"**Priority Rank:** {building.get('priority_rank')}")
            
            with col2:
                st.subheader("Weekend Efficiency")
                predicted = building.get('predicted_weekend_ratio')
                actual = building.get('weekend_ratio')
                gap = building.get('weekend_gap')
                
                if predicted and actual:
                    st.write(f"**Actual:** {actual:.2f}")
                    st.write(f"**Predicted:** {predicted:.2f}")
                    st.write(f"**Gap:** {gap:.2f}" + (" 🔴" if gap and gap > 0.1 else " 🟢"))
            
            with col3:
                st.subheader("Building Info")
                st.write(f"**Type:** {building.get('building_type')}")
                st.write(f"**Night Ratio:** {building.get('night_ratio', 0):.2f}")
            
            # -------------------------------------------------------------------------
            # Recommendation
            # -------------------------------------------------------------------------
            st.subheader("📋 Recommendation")
            st.info(building.get('recommendation', 'No recommendation'))
            
            # -------------------------------------------------------------------------
            # SHAP Explanation
            # -------------------------------------------------------------------------
            shap_values = building.get('shap_values', {})
            
            if shap_values:
                st.subheader("🔬 SHAP Explanation")
                st.caption("Why did the model predict this weekend ratio?")
                
                # Sort by absolute value
                sorted_shap = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
                features = [x[0] for x in sorted_shap]
                values = [x[1] for x in sorted_shap]
                colors = ['#ff6b6b' if v > 0 else '#4ecdc4' for v in values]
                
                fig = go.Figure(go.Bar(
                    x=values,
                    y=features,
                    orientation='h',
                    marker_color=colors
                ))
                fig.update_layout(
                    title="Feature Contributions to Weekend Ratio Prediction",
                    xaxis_title="SHAP Value (impact on prediction)",
                    yaxis_title="Feature",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.caption("🔴 Red = increases prediction (worse efficiency) | 🟢 Green = decreases prediction (better efficiency)")