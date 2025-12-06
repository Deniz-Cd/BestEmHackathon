import os
import sqlite3
import pandas as pd
import numpy as np
import json
import random
import time
from typing import TypedDict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# --- SELENIUM IMPORTS ---
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException

# --- CONFIGURATION ---
load_dotenv()
# os.environ["OPENAI_API_KEY"] = "sk-..." 
DB_NAME = "towns_detailed.db"
MAX_WORKERS = 5  # Adjust based on your RAM (each worker = 1 Chrome window)

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
            safety_walking_night FLOAT
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
    if not text_value:
        return 0.0
    try:
        return float(text_value.strip().split()[0])
    except (ValueError, IndexError):
        return 0.0

def search_city_selenium(city, country=""):    
    # NOTE: This function is called by each thread independently.
    # It creates its OWN driver instance, which is thread-safe.
    print(f"   [Worker] Starting scrape for {city}...")
    
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument(f'user-agent={get_ua()}')
    options.page_load_strategy = 'eager'
    
    # Suppress logging
    options.add_argument("--log-level=3")

    city_url = city.replace(" ", "-").title()
    country_url = country.replace(" ", "-").title() if country else ""
    
    driver = webdriver.Chrome(options=options)
    found = False
    
    try:
        driver.get(f"https://www.numbeo.com/crime/in/{city_url}")
        time.sleep(1) 
        
        if "Cannot find city" in driver.page_source:
            if country:
                driver.get(f"https://www.numbeo.com/crime/in/{city_url}-{country_url}")
                time.sleep(1)
                if "Cannot find city" not in driver.page_source:
                    found = True
        else:
            found = True

        if not found:
            print(f"   [Worker] ❌ Data not found for {city}")
            driver.quit()
            return None

        # Scraping Logic
        scraped_data = {}
        mapping = {
            "Level of crime": "level_of_crime",
            "Crime increasing in the past 3 years": "crime_increasing_5y",
            "Crime increasing in the past 5 years": "crime_increasing_5y",
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
                db_col = None
                for map_key, map_val in mapping.items():
                    if map_key in category:
                        db_col = map_val
                        break
                
                if db_col:
                    raw_val = row.find_element(By.CSS_SELECTOR, "td.indexValueTd").text.strip()
                    scraped_data[db_col] = clean_value(raw_val)
            except Exception:
                continue

        driver.quit()

        # Fill defaults
        for col in mapping.values():
            if col not in scraped_data:
                scraped_data[col] = 0.0

        # Calculate Mean
        vals = list(scraped_data.values())
        safety_index = (100 - (sum(vals) / len(vals))) if vals else 0.0

        final_packet = {
            "name": city,
            "country": country if country else "Unknown",
            "safety_index": round(safety_index, 2),
            **scraped_data
        }
        print(f"   [Worker] ✅ Finished {city} (Index: {final_packet['safety_index']})")
        return final_packet

    except Exception as e:
        print(f"   [Worker] Error scraping {city}: {e}")
        driver.quit()
        return None

def save_to_db(data: dict):
    """Writes a single record to DB. Must be called from Main Thread."""
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
    except Exception as e:
        print(f"DB Error: {e}")
    conn.close()

# --- 3. AGENT TOOLS ---

def get_similar_towns_from_llm(town_name: str):
    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        prompt = (
            f"Find 50 similar towns to {town_name} regarding culture, religion, and laws. "
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

def check_db_node(state: AgentState):
    town = state['target_town']
    print(f"--- Node 1: Checking DB for {town} ---")
    conn = sqlite3.connect(DB_NAME)
    # Check if table exists first to avoid crash on fresh install
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
        print("Could not find data for target town. Stopping.")
        return {"target_town_data": {}}

def find_neighbors_node(state: AgentState):
    target_data = state['target_town_data']
    if not target_data: 
        return {"final_dataset_names": []}

    target_index = target_data['safety_index']
    target_name = target_data['name']
    
    print(f"--- Node 3: Parallel Search for neighbors safer than {target_index} ---")
    
    # 1. Get List of Towns
    similar_towns = get_similar_towns_from_llm(target_name)
    # Dedup and remove target
    similar_towns = [t for t in list(set(similar_towns)) if t != target_name]
    
    valid_towns = [target_name]
    towns_to_scrape = []
    
    # 2. Check DB first (Fast sequential check)
    conn = sqlite3.connect(DB_NAME)
    for town in similar_towns:
        existing = pd.read_sql(f"SELECT * FROM towns WHERE name = '{town}'", conn)
        if not existing.empty:
            town_data = existing.iloc[0].to_dict()
            if town_data['safety_index'] > target_index:
                print(f"   [Cache] {town} matched ({town_data['safety_index']})")
                valid_towns.append(town)
        else:
            towns_to_scrape.append(town)
    conn.close()
    
    # 3. Multithreaded Scraping for the rest
    if towns_to_scrape:
        print(f"   ...Spawning {len(towns_to_scrape)} threads for scraping...")
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Map scraper to towns
            future_to_town = {executor.submit(search_city_selenium, town): town for town in towns_to_scrape}
            
            for future in as_completed(future_to_town):
                town = future_to_town[future]
                try:
                    data = future.result() # This blocks until thread finishes
                    if data:
                        # Logic: Add if Safety Index is HIGHER
                        if data['safety_index'] > target_index:
                            save_to_db(data) # Write to DB in main thread (Safe)
                            valid_towns.append(town)
                except Exception as exc:
                    print(f"   [Main] Thread exception for {town}: {exc}")

    return {"final_dataset_names": valid_towns}

def analysis_node(state: AgentState):
    town_names = state['final_dataset_names']
    print(f"--- Node 4: Analyzing {len(town_names)} towns ---")
    
    if len(town_names) < 3:
        print("Not enough data for correlation.")
        return

    conn = sqlite3.connect(DB_NAME)
    placeholders = ',' .join('?' for _ in town_names)
    df = pd.read_sql(f"SELECT * FROM towns WHERE name IN ({placeholders})", conn, params=town_names)
    conn.close()
    
    numeric_df = df.select_dtypes(include=[np.number])
    if 'safety_index' not in numeric_df.columns: return

    corr = numeric_df.corr()['safety_index'].drop('safety_index')
    top_5 = corr.abs().sort_values(ascending=False).head(5)
    
    print("\n" + "="*50)
    print("📈 TOP 5 INFLUENTIAL FEATURES")
    print("="*50)
    for feature, score in top_5.items():
        print(f"{feature: <30} | Correlation: {corr[feature]:.4f}")
    print("="*50 + "\n")

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

workflow.set_entry_point("check_db")
workflow.add_conditional_edges("check_db", decide_route, {"scrape_target": "scrape_target", "find_neighbors": "find_neighbors"})
workflow.add_edge("scrape_target", "find_neighbors")
workflow.add_edge("find_neighbors", "analyze")
workflow.add_edge("analyze", END)

app = workflow.compile()

if __name__ == "__main__":
    print("Initializing Database...")
    init_db()  # Ensures tables exist
    
    town_input = input("Enter a town name: ")
    print(f"🚀 Starting Parallel Agent for: {town_input}")
    app.invoke({"target_town": town_input})