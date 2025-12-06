import os
import sqlite3
import pandas as pd
import numpy as np
import random
import time
import textwrap
import re
from typing import TypedDict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from xgboost import XGBRegressor

# --- SELENIUM IMPORTS ---
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# --- NEW IMPORTS FOR INTERFACE ---
from geopy.geocoders import Nominatim
from fpdf import FPDF

# --- CONFIGURATION ---
load_dotenv()
DB_NAME = "towns_detailed.db"
MAX_WORKERS = 5

# --- HELPER: GEOCODING (Required for Map) ---
def get_coordinates(city_name, country=""):
    """Fetch Lat/Lon for mapping."""
    time.sleep(1.0) # Respect API limits
    try:
        geolocator = Nominatim(user_agent=f"safety_agent_{random.randint(1000,9999)}")
        query = f"{city_name}, {country}" if country else city_name
        location = geolocator.geocode(query, timeout=10)
        if location:
            return location.latitude, location.longitude
    except Exception as e:
        print(f"Geo Error: {e}")
    return None, None

# --- 1. SETUP DATABASE ---
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS towns (
            name TEXT PRIMARY KEY,
            country TEXT,
            safety_index FLOAT,
            level_of_crime FLOAT,
            crime_increasing_5y FLOAT,
            worry_home_broken FLOAT,
            worry_mugged_robbed FLOAT,
            worry_car_stolen FLOAT,
            worry_car_items_stolen FLOAT,
            worry_attacked FLOAT,
            worry_insulted FLOAT,
            worry_discrimination FLOAT,
            problem_drugs FLOAT,
            problem_property_crimes FLOAT,
            problem_violent_crimes FLOAT,
            problem_corruption FLOAT,
            safety_walking_day FLOAT,
            safety_walking_night FLOAT,
            lat FLOAT,
            lon FLOAT
        )
    ''')
    conn.commit()
    conn.close()

# --- 2. SCRAPER FUNCTIONS ---

def get_ua(filename="user_agents.txt"):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            user_agents = [line.strip() for line in f if line.strip()]
        return random.choice(user_agents)
    except FileNotFoundError:
        return "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

def clean_value(text_value):
    if not text_value: return 0.0
    try:
        return float(text_value.strip().split()[0])
    except:
        return 0.0

def search_city_selenium(city, country=""):    
    print(f"   [Worker] Starting scrape for {city}...")
    
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument(f'user-agent={get_ua()}')
    options.page_load_strategy = 'eager'
    options.add_argument("--log-level=3")

    city_url = city.replace(" ", "-").title()
    country_url = country.replace(" ", "-").title() if country else ""
    
    driver = webdriver.Chrome(options=options)
    
    try:
        # 1. Scrape Numbeo
        driver.get(f"https://www.numbeo.com/crime/in/{city_url}")
        time.sleep(1) 
        
        if "Cannot find city" in driver.page_source:
            if country:
                driver.get(f"https://www.numbeo.com/crime/in/{city_url}-{country_url}")
                if "Cannot find city" in driver.page_source:
                    driver.quit()
                    return None
            else:
                driver.quit()
                return None

        scraped_data = {}
        mapping = {
            "Level of crime": "level_of_crime",
            "Crime increasing in the past 3 years": "crime_increasing_5y",
            "Worries home broken": "worry_home_broken",
            "Worries being mugged": "worry_mugged_robbed",
            "Worries car stolen": "worry_car_stolen",
            "Worries things from car stolen": "worry_car_items_stolen",
            "Worries attacked": "worry_attacked",
            "Worries being insulted": "worry_insulted",
            "Worries being subject to a physical attack": "worry_discrimination",
            "Problem people using or dealing drugs": "problem_drugs",
            "Problem property crimes": "problem_property_crimes",
            "Problem violent crimes": "problem_violent_crimes",
            "Problem corruption": "problem_corruption",
            "Safety walking alone during daylight": "safety_walking_day",
            "Safety walking alone during night": "safety_walking_night"
        }

        rows = driver.find_elements(By.CSS_SELECTOR, "table.data_wide_table tbody tr")
        for row in rows:
            txt = row.text
            for k, v in mapping.items():
                if k in txt:
                    try:
                        val_elem = row.find_element(By.CSS_SELECTOR, "td.indexValueTd")
                        scraped_data[v] = clean_value(val_elem.text)
                    except:
                        pass
        
        driver.quit()

        # Fill defaults
        for v in mapping.values():
            if v not in scraped_data: scraped_data[v] = 0.0

        # Calc Index
        vals = list(scraped_data.values())
        safety_index = (100 - (sum(vals) / len(vals))) if vals else 0.0
        
        # 2. Get Geolocation (Added for Map)
        lat, lon = get_coordinates(city, country)

        return {
            "name": city,
            "country": country if country else "Unknown",
            "safety_index": round(safety_index, 2),
            "lat": lat, "lon": lon,
            **scraped_data
        }

    except Exception as e:
        print(f"   [Worker] Error scraping {city}: {e}")
        driver.quit()
        return None

def save_to_db(data: dict):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    try:
        # Added lat/lon to insert
        cursor.execute('''
            INSERT OR REPLACE INTO towns VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
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
            data['safety_walking_night'],
            data.get('lat'), data.get('lon')
        ))
        conn.commit()
    except Exception as e:
        print(f"DB Error: {e}")
    conn.close()

# --- 3. AGENT TOOLS ---

def get_similar_towns_from_llm(town_name: str):
    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        prompt = (
            f"Find 30 similar towns to {town_name} regarding culture, religion, and laws. "
            "Remove diacritical marks from names."
            "Return ONLY a raw comma-separated list of strings. Example: 'Paris, Lyon, Marseille'. "
            "Do not number them."
        )
        response = llm.invoke(prompt)
        return [t.strip() for t in response.content.split(',')]
    except:
        return ["Paris", "London", "Berlin", "Vienna", "Madrid"]

# --- 4. STATE & NODES ---

class AgentState(TypedDict):
    target_town: str
    target_town_data: dict
    final_dataset_names: List[str]
    top_correlations: dict
    pdf_path: str # Added for interface

def check_db_node(state: AgentState):
    town = state['target_town']
    print(f"--- Node 1: Checking DB for {town} ---")
    conn = sqlite3.connect(DB_NAME)
    try:
        df = pd.read_sql(f"SELECT * FROM towns WHERE name = '{town}'", conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    
    if not df.empty:
        return {"target_town_data": df.iloc[0].to_dict()}
    return {"target_town_data": None}

def scrape_target_node(state: AgentState):
    town = state['target_town']
    print(f"--- Node 2: Scraping target {town} ---")
    data = search_city_selenium(town)
    
    if data:
        save_to_db(data)
        return {"target_town_data": data}
    else:
        return {"target_town_data": {}}

def find_neighbors_node(state: AgentState):
    target_data = state['target_town_data']
    if not target_data: 
        return {"final_dataset_names": []}

    target_index = target_data['safety_index']
    target_name = target_data['name']
    
    print(f"--- Node 3: Search for neighbors ---")
    
    similar_towns = get_similar_towns_from_llm(target_name)
    similar_towns = [t for t in list(set(similar_towns)) if t != target_name]
    
    valid_towns = [target_name]
    towns_to_scrape = []
    
    conn = sqlite3.connect(DB_NAME)
    for town in similar_towns:
        existing = pd.read_sql(f"SELECT * FROM towns WHERE name = '{town}'", conn)
        if not existing.empty:
            town_data = existing.iloc[0].to_dict()
            # RELAXED FILTER for Bucharest/Safe cities
            if town_data['safety_index'] > (target_index - 15):
                valid_towns.append(town)
        else:
            towns_to_scrape.append(town)
    conn.close()
    
    if towns_to_scrape:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_town = {executor.submit(search_city_selenium, town): town for town in towns_to_scrape}
            
            for future in as_completed(future_to_town):
                try:
                    data = future.result()
                    if data:
                        # RELAXED FILTER for new scrapes
                        if data['safety_index'] > (target_index - 15):
                            save_to_db(data)
                            valid_towns.append(data['name'])
                except Exception:
                    pass

    return {"final_dataset_names": valid_towns}

def analysis_node(state: AgentState):
    """(Kept exact logic using XGBoost)"""
    town_names = state['final_dataset_names']
    print(f"--- Node 4: Analyzing {len(town_names)} towns ---")
    
    if len(town_names) < 3:
        return {"top_correlations": {}}

    conn = sqlite3.connect(DB_NAME)
    placeholders = ','.join('?' for _ in town_names)
    try:
        df = pd.read_sql(
            f"SELECT * FROM towns WHERE name IN ({placeholders})",
            conn,
            params=town_names
        )
    except Exception:
        return {"top_correlations": {}}
    finally:
        conn.close()
    
    numeric_df = df.select_dtypes(include=[np.number])
    if 'safety_index' not in numeric_df.columns:
        return {"top_correlations": {}}

    feature_cols = [c for c in numeric_df.columns if c not in ['safety_index', 'lat', 'lon']]
    X = numeric_df[feature_cols].dropna()
    y = numeric_df.loc[X.index, 'safety_index']

    if X.empty:
        return {"top_correlations": {}}

    # XGBoost Logic
    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    importances = model.feature_importances_
    importance_series = pd.Series(importances, index=feature_cols)
    top = importance_series.sort_values(ascending=False).head(5)

    return {"top_correlations": top.to_dict()}

def generate_pdf_report_node(state: AgentState):
    """
    (REPLACED LATEX WITH FPDF FOR INTERFACE STABILITY)
    """
    town = state['target_town']
    correlations = state['top_correlations']
    
    print(f"--- Node 5: Generating PDF Report for {town} ---")
    
    if not correlations:
        return {"pdf_path": None}

    # 1. LLM for Text
    data_summary = "\n".join([f"- {feature}: {score:.4f} importance" for feature, score in correlations.items()])
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    
    prompt = textwrap.dedent(f"""
        Write a safety strategy report for {town}.
        Top Risk Factors (Feature Importance):
        {data_summary}
        
        Write clear, plain English. No Markdown (* or #).
        Structure:
        1. Executive Summary
        2. Key Risk Factors
        3. Recommendations
    """)
    text_content = llm.invoke(prompt).content

    # 2. FPDF Generation
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, f"Safety Report: {town}", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", size=11)
    safe_text = text_content.encode('latin-1', 'ignore').decode('latin-1')
    pdf.multi_cell(0, 7, safe_text)
    
    fname = f"{town.replace(' ', '_')}_report.pdf"
    pdf.output(fname)
    
    return {"pdf_path": fname}

# --- 5. BUILD GRAPH ---

def decide_route(state: AgentState):
    if state['target_town_data'] is None:
        return "scrape_target"
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