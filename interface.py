import streamlit as st
import pandas as pd
import sqlite3
import folium
from streamlit_folium import st_folium
import os
import agent  # Imports your agent.py

# --- PAGE CONFIG ---
st.set_page_config(layout="wide", page_title="Live Safety Agent")

# --- INITIALIZE STATE ---
if "stage" not in st.session_state:
    st.session_state.stage = "idle" # idle, finding_target, finding_neighbors, report
if "agent_state" not in st.session_state:
    st.session_state.agent_state = {} 
if "map_center" not in st.session_state:
    st.session_state.map_center = [48.85, 2.35] # Default (Europe)
if "map_zoom" not in st.session_state:
    st.session_state.map_zoom = 4

# --- HELPER: GET DATA FOR MAP ---
def get_map_data(town_names):
    """Get coords for a list of towns from DB"""
    if not town_names: return []
    conn = sqlite3.connect(agent.DB_NAME)
    placeholders = ','.join('?' for _ in town_names)
    try:
        df = pd.read_sql(
            f"SELECT name, lat, lon, safety_index FROM towns WHERE name IN ({placeholders})",
            conn, params=town_names
        )
    except:
        return []
    finally:
        conn.close()
    return df.to_dict('records')

# --- SIDEBAR ---
with st.sidebar:
    st.title("🕵️‍♂️ Safety Agent")
    target_input = st.text_input("Target City", "London")
    
    # The Button Logic
    if st.button("🚀 Start Investigation", type="primary"):
        # Reset everything
        agent.init_db()
        st.session_state.stage = "finding_target"
        st.session_state.agent_state = {"target_town": target_input}
        st.rerun()

    st.divider()
    
    # Progress Indicators
    if st.session_state.stage == "finding_target":
        st.spinner("🕵️ Scouring database & web for target...")
    elif st.session_state.stage == "finding_neighbors":
        st.success("✅ Target Locked.")
        st.spinner("🛰️ Scanning for comparable cities...")
    elif st.session_state.stage == "analyzing":
        st.success("✅ Network Built.")
        st.spinner("🧠 Analyzing correlations...")
    elif st.session_state.stage == "finished":
        st.success("✅ Analysis Complete!")

# --- MAIN LOGIC (PHASED EXECUTION) ---

# PHASE 1: Find Target
if st.session_state.stage == "finding_target":
    # 1. Check DB
    res = agent.check_db_node(st.session_state.agent_state)
    st.session_state.agent_state.update(res)
    
    # 2. If not in DB, Scrape
    if not st.session_state.agent_state.get('target_town_data'):
        res = agent.scrape_target_node(st.session_state.agent_state)
        st.session_state.agent_state.update(res)
    
    # 3. Update Map Focus
    data = st.session_state.agent_state.get('target_town_data')
    
    if data and pd.notna(data.get('lat')) and pd.notna(data.get('lon')):
        st.session_state.map_center = [data['lat'], data['lon']]
        st.session_state.map_zoom = 7
        st.session_state.stage = "finding_neighbors" # Move to next stage
        st.rerun() 
    else:
        st.error(f"Could not find coordinates for {target_input}. Please try another city.")
        st.session_state.stage = "idle"

# PHASE 2: Find Neighbors
elif st.session_state.stage == "finding_neighbors":
    # Run the neighbors node
    res = agent.find_neighbors_node(st.session_state.agent_state)
    st.session_state.agent_state.update(res)
    
    st.session_state.stage = "analyzing" # Move to next stage
    st.rerun()

# PHASE 3: Analyze & Report
elif st.session_state.stage == "analyzing":
    # Run Analysis
    res = agent.analysis_node(st.session_state.agent_state)
    st.session_state.agent_state.update(res)
    
    # Run Report
    res = agent.generate_pdf_report_node(st.session_state.agent_state)
    st.session_state.agent_state.update(res)
    
    st.session_state.stage = "finished"
    st.rerun()

# --- VISUALIZATION (Renders on every Rerun) ---

st.subheader(f"Live Intelligence Map: {target_input}")

# 1. Setup Map
m = folium.Map(location=st.session_state.map_center, zoom_start=st.session_state.map_zoom)

# 2. Plot Data based on current state
target_data = st.session_state.agent_state.get('target_town_data')
neighbor_names = st.session_state.agent_state.get('final_dataset_names', [])

target_coords = None

# A. Plot Target (Red)
if target_data and pd.notna(target_data.get('lat')):
    target_coords = (target_data['lat'], target_data['lon'])
    folium.Marker(
        target_coords,
        popup=f"TARGET: {target_data['name']}",
        icon=folium.Icon(color="red", icon="home", prefix='fa')
    ).add_to(m)

# B. Plot Neighbors & Lines (Blue)
if target_coords and neighbor_names:
    neighbors = get_map_data(neighbor_names)
    
    # Open connection for any necessary repairs
    conn = sqlite3.connect(agent.DB_NAME) 
    cursor = conn.cursor()
    
    for city in neighbors:
        # Skip if it is the target
        if city['name'] == target_data['name']: continue
        
        lat = city.get('lat')
        lon = city.get('lon')
        
        # --- AUTO-REPAIR MISSING COORDINATES ---
        # If the DB has the city but no coords (NaN/None), fix it NOW.
        if pd.isna(lat) or pd.isna(lon) or lat is None:
            # Show a small spinner so the user knows why it's pausing
            with st.spinner(f"Fixing map data for {city['name']}..."):
                new_lat, new_lon = agent.get_coordinates(city['name'])
                if new_lat and new_lon:
                    lat, lon = new_lat, new_lon
                    # Update DB immediately so it works next time without delay
                    try:
                        cursor.execute("UPDATE towns SET lat=?, lon=? WHERE name=?", (lat, lon, city['name']))
                        conn.commit()
                    except:
                        pass
        # ---------------------------------------

        # Only plot if we have valid coords (either from DB or fresh repair)
        if lat and lon and pd.notna(lat) and pd.notna(lon):
            n_coords = (lat, lon)
            
            # Marker
            folium.Marker(
                n_coords,
                popup=f"{city['name']}\nSafety: {city['safety_index']}",
                icon=folium.Icon(color="blue", icon="info-sign")
            ).add_to(m)
            
            # Line connecting to target
            folium.PolyLine(
                locations=[target_coords, n_coords],
                color="blue",
                weight=2,
                opacity=0.5,
                dash_array='5, 10'
            ).add_to(m)
            
    conn.close()

# Render Map
st_folium(m, width="100%", height=500)

# --- RESULTS SECTION ---
if st.session_state.stage == "finished":
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("📊 Key Correlations Identified")
        corrs = st.session_state.agent_state.get('top_correlations', {})
        st.write(corrs)
        
    with col2:
        st.success("📄 Strategy Report Ready")
        pdf_path = st.session_state.agent_state.get('pdf_path')
        if pdf_path and os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                st.download_button("Download PDF", f, file_name=pdf_path)