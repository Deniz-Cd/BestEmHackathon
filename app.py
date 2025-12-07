import streamlit as st
import pydeck as pdk
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import pandas as pd
import sqlite3
import numpy as np
import time
import os
import subprocess
import re
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypedDict, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from xgboost import XGBRegressor

# --- SELENIUM IMPORTS ---
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# --- CONFIGURATION & SETUP ---
st.set_page_config(layout="wide", page_title="SafeCity Agent")
load_dotenv()
DB_NAME = "towns_detailed.db"
MAX_WORKERS = 5

# --- GEOCODING HELPER ---
@st.cache_data
def get_coordinates(city_name):
    # User agent helps identify your script to the geocoding service
    geolocator = Nominatim(user_agent="safecity_agent_v1")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    try:
        location = geocode(city_name)
        if location:
            return {"lat": location.latitude, "lon": location.longitude}
    except:
        return None
    return None

# --- ORIGINAL DB & SCRAPER FUNCTIONS ---

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS towns (
            name TEXT PRIMARY KEY, country TEXT, safety_index FLOAT,
            level_of_crime FLOAT, crime_increasing_5y FLOAT, worry_home_broken FLOAT,
            worry_mugged_robbed FLOAT, worry_car_stolen FLOAT, worry_car_items_stolen FLOAT,
            worry_attacked FLOAT, worry_insulted FLOAT, worry_discrimination FLOAT,
            problem_drugs FLOAT, problem_property_crimes FLOAT, problem_violent_crimes FLOAT,
            problem_corruption FLOAT, safety_walking_day FLOAT, safety_walking_night FLOAT
        )
    ''')
    conn.commit()
    conn.close()

def get_ua():
    return "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

def clean_value(text_value):
    if not text_value: return 0.0
    try: return float(text_value.strip().split()[0])
    except: return 0.0

def search_city_selenium(city, country=""):    
    # [LOGGING] Start
    print(f"   🔎 [Scraper] Starting: {city}...")
    
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument(f'user-agent={get_ua()}')
    options.add_argument("--log-level=3")
    
    city_url = city.replace(" ", "-").title()
    driver = webdriver.Chrome(options=options)
    
    try:
        driver.get(f"https://www.numbeo.com/crime/in/{city_url}")
        time.sleep(1)
        if "Cannot find city" in driver.page_source:
            print(f"   ❌ [Scraper] 404 Not Found: {city}")
            driver.quit()
            return None

        scraped_data = {}
        mapping = {
            "Level of crime": "level_of_crime",
            "Crime increasing in the past 3 years": "crime_increasing_5y",
            "Worries home broken and things stolen": "worry_home_broken",
            "Worries being mugged or robbed": "worry_mugged_robbed",
            "Worries car stolen": "worry_car_stolen",
            "Worries things from car stolen": "worry_car_items_stolen",
            "Worries attacked": "worry_attacked",
            "Worries being insulted": "worry_insulted",
            "Worries being subject to a physical attack because of your skin color, ethnic origin, gender or religion": "worry_discrimination",
            "Problem people using or dealing drugs": "problem_drugs",
            "Problem property crimes such as vandalism and theft": "problem_property_crimes",
            "Problem violent crimes such as assault and armed robbery": "problem_violent_crimes",
            "Problem corruption and bribery": "problem_corruption",
            "Safety walking alone during daylight": "safety_walking_day",
            "Safety walking alone during night": "safety_walking_night"
        }

        rows = driver.find_elements(By.CSS_SELECTOR, "table.data_wide_table tbody tr")
        for row in rows:
            try:
                category = row.find_element(By.CSS_SELECTOR, "td").text.strip()
                for map_key, map_val in mapping.items():
                    if map_key in category:
                        raw_val = row.find_element(By.CSS_SELECTOR, "td.indexValueTd").text.strip()
                        scraped_data[map_val] = clean_value(raw_val)
                        break
            except: continue
        
        driver.quit()

        # Defaults
        for col in mapping.values():
            if col not in scraped_data: scraped_data[col] = 0.0

        vals = list(scraped_data.values())
        safety_index = (100 - (sum(vals) / len(vals))) if vals else 0.0

        # [LOGGING] Success
        print(f"   ✅ [Scraper] Success: {city} | Safety Index: {safety_index:.2f}")

        return {
            "name": city,
            "country": "Unknown",
            "safety_index": round(safety_index, 2),
            **scraped_data
        }
    except Exception as e:
        print(f"   ⚠️ [Scraper] Error for {city}: {e}")
        driver.quit()
        return None

def save_to_db(data: dict):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT OR IGNORE INTO towns VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        ''', (
            data['name'], data['country'], data['safety_index'],
            data['level_of_crime'], data['crime_increasing_5y'],
            data['worry_home_broken'], data['worry_mugged_robbed'],
            data['worry_car_stolen'], data['worry_car_items_stolen'],
            data['worry_attacked'], data['worry_insulted'],
            data['worry_discrimination'], data['problem_drugs'],
            data['problem_property_crimes'], data['problem_violent_crimes'],
            data['problem_corruption'], data['safety_walking_day'],
            data['safety_walking_night']
        ))
        conn.commit()
    except Exception: pass
    conn.close()

# --- AGENT TOOLS & NODES ---

def get_similar_towns_from_llm(town_name: str):
    try:
        print(f"   🤖 [LLM] Asking GPT-4o for neighbors of {town_name}...")
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        prompt = (f"Find 30 similar towns to {town_name} regarding culture and safety. "
                  "Return ONLY a raw comma-separated list of strings. Example: 'Paris, Lyon'.")
        response = llm.invoke(prompt)
        towns = [t.strip() for t in response.content.split(',')]
        print(f"   🤖 [LLM] Returned: {towns}")
        return towns
    except:
        return ["London", "Berlin", "Madrid", "Rome"]

class AgentState(TypedDict):
    target_town: str
    target_town_data: dict
    final_dataset_names: List[str]
    top_correlations: dict
    map_data: List[dict] 

def check_db_node(state: AgentState):
    town = state['target_town']
    print(f"\n--- 1. Checking Database for {town} ---")
    conn = sqlite3.connect(DB_NAME)
    try:
        df = pd.read_sql(f"SELECT * FROM towns WHERE name = '{town}'", conn)
    except: df = pd.DataFrame()
    conn.close()
    
    map_data = []
    coords = get_coordinates(town)
    if coords:
        map_data.append({"name": town, "lat": coords['lat'], "lon": coords['lon'], "type": "target", "safety": 0})

    if not df.empty:
        print(f"   ✅ Found {town} in DB.")
        if map_data: map_data[0]['safety'] = df.iloc[0]['safety_index']
        return {"target_town_data": df.iloc[0].to_dict(), "map_data": map_data}
    
    print(f"   ❌ {town} not in DB.")
    return {"target_town_data": None, "map_data": map_data}

def scrape_target_node(state: AgentState):
    town = state['target_town']
    print(f"\n--- 2. Scraping Target: {town} ---")
    data = search_city_selenium(town)
    
    if data:
        save_to_db(data)
        updated_map = state['map_data']
        if updated_map:
            updated_map[0]['safety'] = data['safety_index']
        return {"target_town_data": data, "map_data": updated_map}
    return {"target_town_data": {}}

def find_neighbors_node(state: AgentState):
    target_data = state['target_town_data']
    if not target_data: return {"final_dataset_names": []}

    target_index = target_data['safety_index']
    target_name = target_data['name']
    
    print(f"\n--- 3. Finding Neighbors (Target Index: {target_index}) ---")
    
    similar_towns = get_similar_towns_from_llm(target_name)
    similar_towns = [t for t in list(set(similar_towns)) if t != target_name]
    
    valid_towns = [target_name]
    map_update = state['map_data']
    towns_to_scrape = []
    
    conn = sqlite3.connect(DB_NAME)
    print("   ... Checking cache for neighbors ...")
    for town in similar_towns:
        existing = pd.read_sql(f"SELECT * FROM towns WHERE name = '{town}'", conn)
        if not existing.empty:
            t_data = existing.iloc[0].to_dict()
            # If safety index is 100, it's likely a bad scrape from before, but we accept it logic-wise
            if t_data['safety_index'] > target_index:
                print(f"   💾 [Cache Hit] {town} is safer ({t_data['safety_index']})")
                valid_towns.append(town)
                coords = get_coordinates(town)
                if coords:
                    map_update.append({
                        "name": town, "lat": coords['lat'], "lon": coords['lon'], 
                        "type": "neighbor", "safety": t_data['safety_index']
                    })
            else:
                print(f"   💾 [Cache Hit] {town} is NOT safer ({t_data['safety_index']})")
        else:
            towns_to_scrape.append(town)
    conn.close()
    
    if towns_to_scrape:
        print(f"   ... Spawning workers for {len(towns_to_scrape)} towns ...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_town = {executor.submit(search_city_selenium, town): town for town in towns_to_scrape}
            for future in as_completed(future_to_town):
                town = future_to_town[future]
                try:
                    data = future.result()
                    if data and data['safety_index'] > target_index:
                        save_to_db(data)
                        valid_towns.append(town)
                        coords = get_coordinates(town)
                        if coords:
                            map_update.append({
                                "name": town, "lat": coords['lat'], "lon": coords['lon'], 
                                "type": "neighbor", "safety": data['safety_index']
                            })
                except: pass

    return {"final_dataset_names": valid_towns, "map_data": map_update}

def analysis_node(state: AgentState):
    town_names = state['final_dataset_names']
    print(f"\n--- 4. Analyzing {len(town_names)} Towns ---")
    if len(town_names) < 3: 
        print("   ⚠️ Not enough data for XGBoost.")
        return {"top_correlations": {}}

    conn = sqlite3.connect(DB_NAME)
    placeholders = ','.join('?' for _ in town_names)
    df = pd.read_sql(f"SELECT * FROM towns WHERE name IN ({placeholders})", conn, params=town_names)
    conn.close()
    
    numeric_df = df.select_dtypes(include=[np.number])
    X = numeric_df.drop(columns=['safety_index'], errors='ignore').dropna()
    y = numeric_df['safety_index'].loc[X.index]
    
    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    top = importance[importance > 0.05]
    
    print("   📊 Analysis Complete. Top features found.")
    return {"top_correlations": top.to_dict()}

def generate_pdf_report_node(state: AgentState):
    town = state['target_town']
    correlations = state['top_correlations']
    print(f"\n--- 5. Generating Report for {town} ---")
    if not correlations: return

    data_summary = "\n".join([f"- {k}: {v:.4f}" for k,v in correlations.items()])
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = f"""Generate LaTeX report for {town}. Top safety factors: \n{data_summary}. 
    Return ONLY LaTeX code inside ```latex ... ```."""
    
    response = llm.invoke(prompt)
    latex_code = response.content.replace("```latex", "").replace("```", "").strip()
    
    filename = f"{town}_report.tex"
    with open(filename, "w", encoding="utf-8") as f: f.write(latex_code)
    
    try:
        subprocess.run(["pdflatex", "-interaction=nonstopmode", filename], stdout=subprocess.DEVNULL)
        print(f"   📄 PDF compiled: {filename.replace('.tex', '.pdf')}")
    except: 
        print("   ⚠️ Could not compile PDF (pdflatex missing or path error).")
    
    return

# --- GRAPH SETUP ---
def decide_route(state: AgentState):
    if not state['target_town_data']: return "scrape_target"
    return "find_neighbors"

workflow = StateGraph(AgentState)
workflow.add_node("check_db", check_db_node)
workflow.add_node("scrape_target", scrape_target_node)
workflow.add_node("find_neighbors", find_neighbors_node)
workflow.add_node("analyze", analysis_node)
workflow.add_node("generate_report", generate_pdf_report_node)

workflow.set_entry_point("check_db")
workflow.add_conditional_edges("check_db", decide_route, {"scrape_target": "scrape_target", "find_neighbors": "find_neighbors"})
workflow.add_edge("scrape_target", "find_neighbors")
workflow.add_edge("find_neighbors", "analyze")
workflow.add_edge("analyze", "generate_report")
workflow.add_edge("generate_report", END)

app = workflow.compile()

# --- STREAMLIT UI ---
def render_map(map_data):
    if not map_data:
        return
    
    target = next((x for x in map_data if x['type'] == 'target'), None)
    df = pd.DataFrame(map_data)
    
    layers = []
    
    layers.append(pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position='[lon, lat]',
        get_color='[type == "target" ? 255 : 0, type == "target" ? 0 : 255, 100, 160]',
        get_radius=20000,
        pickable=True,
    ))
    
    if target:
        df['target_lon'] = target['lon']
        df['target_lat'] = target['lat']
        arcs_df = df[df['type'] == 'neighbor']
        
        layers.append(pdk.Layer(
            "ArcLayer",
            data=arcs_df,
            get_source_position='[lon, lat]',
            get_target_position='[target_lon, target_lat]',
            get_source_color=[0, 255, 100, 100],
            get_target_color=[255, 0, 0, 100],
            get_width=3,
        ))

    view_state = pdk.ViewState(
        latitude=target['lat'] if target else 48.8,
        longitude=target['lon'] if target else 2.3,
        zoom=4,
        pitch=40,
    )

    r = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip={"text": "{name}\nSafety Index: {safety}"}
    )
    st.pydeck_chart(r)

def main():
    init_db()
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.title("🛡️ SafeCity Agent")
        st.markdown("Enter a city to analyze safety and find safer alternatives.")
        town_input = st.text_input("Target City", "Bucharest")
        start_btn = st.button("Start Analysis")
        status_area = st.empty()
        
    with col2:
        map_area = st.empty()

    if start_btn and town_input:
        inputs = {"target_town": town_input, "map_data": []}
        for output in app.stream(inputs):
            node_name = list(output.keys())[0]
            current_state = output[node_name]
            
            status_area.info(f"Agent Status: Finished step **{node_name}**")
            
            # --- Check if state is valid ---
            if current_state and "map_data" in current_state and current_state['map_data']:
                with map_area:
                    render_map(current_state['map_data'])

            if node_name == "analyze" and current_state and "top_correlations" in current_state:
                with col1:
                    st.success("Analysis Complete!")
                    st.subheader("Top Safety Factors")
                    st.write(current_state['top_correlations'])
                    
            if node_name == "generate_report":
                with col1:
                    st.markdown(f"📄 **PDF Report Generated for {town_input}**")

if __name__ == "__main__":
    main()