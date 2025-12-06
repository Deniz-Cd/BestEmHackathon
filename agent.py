import os
import sqlite3
import pandas as pd
import numpy as np
import json
import random
import time
import subprocess
import textwrap
import re
from typing import TypedDict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV

# --- SELENIUM IMPORTS ---
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException

# --- CONFIGURATION ---
load_dotenv()
# os.environ["OPENAI_API_KEY"] = "sk-..." 
DB_NAME = "towns_detailed.db"
MAX_WORKERS = 5  # Adjust based on your RAM

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
    print(f"   [Worker] Starting scrape for {city}...")
    
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument(f'user-agent={get_ua()}')
    options.page_load_strategy = 'eager'
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
    top_correlations: dict  # Stores the top 5 influential features

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
        print("Could not find data for target town. Stopping.")
        return {"target_town_data": {}}

def find_neighbors_node(state: AgentState):
    target_data = state['target_town_data']
    if not target_data: 
        return {"final_dataset_names": []}

    target_index = target_data['safety_index']
    target_name = target_data['name']
    
    print(f"--- Node 3: Parallel Search for neighbors safer than {target_index} ---")
    
    similar_towns = get_similar_towns_from_llm(target_name)
    similar_towns = [t for t in list(set(similar_towns)) if t != target_name]
    
    valid_towns = [target_name]
    towns_to_scrape = []
    
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
    
    if towns_to_scrape:
        print(f"   ...Spawning {len(towns_to_scrape)} threads for scraping...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_town = {executor.submit(search_city_selenium, town): town for town in towns_to_scrape}
            
            for future in as_completed(future_to_town):
                town = future_to_town[future]
                try:
                    data = future.result()
                    if data:
                        if data['safety_index'] > target_index:
                            save_to_db(data)
                            valid_towns.append(town)
                except Exception as exc:
                    print(f"   [Main] Thread exception for {town}: {exc}")

    return {"final_dataset_names": valid_towns}

from xgboost import XGBRegressor  # add this at the top of the file with other imports

def analysis_node(state: AgentState):
    """Trains an XGBoost model and uses feature importance to find the top 5 features."""
    town_names = state['final_dataset_names']
    print(f"--- Node 4: Analyzing {len(town_names)} towns with XGBoost ---")
    
    if len(town_names) < 3:
        print("Not enough data for model-based feature importance.")
        return {"top_correlations": {}}

    conn = sqlite3.connect(DB_NAME)
    placeholders = ','.join('?' for _ in town_names)
    try:
        df = pd.read_sql(
            f"SELECT * FROM towns WHERE name IN ({placeholders})",
            conn,
            params=town_names
        )
    except Exception as e:
        print(f"Error reading DB: {e}")
        return {"top_correlations": {}}
    finally:
        conn.close()
    
    numeric_df = df.select_dtypes(include=[np.number])
    if 'safety_index' not in numeric_df.columns:
        print("No safety_index column found in numeric data.")
        return {"top_correlations": {}}

    # Prepare features (X) and target (y)
    feature_cols = [c for c in numeric_df.columns if c != 'safety_index']
    if not feature_cols:
        print("No feature columns available for training.")
        return {"top_correlations": {}}

    X = numeric_df[feature_cols]
    y = numeric_df['safety_index']

    # Optional: handle missing values
    X = X.dropna()       # remove rows with any NaNs
    y = y.loc[X.index]   # keep only corresponding rows in y
    y = y.dropna()       # remove any NaNs in y
    X = X.loc[y.index]   # keep only corresponding rows in X

    # Train XGBoost model
    model = XGBRegressor(n_estimators=100, random_state=42)

    try:
        model.fit(X, y)
    except Exception as e:
        print(f"Error training XGBoost model: {e}")
        return {"top_correlations": {}}

    # Get feature importances from the trained model
    importances = model.feature_importances_
    importance_series = pd.Series(importances, index=feature_cols)

    # Take top 5 most important features
    top = importance_series.sort_values(ascending=False)

    top = top[top > 0.1]  # Filter out zero importance features

    print("\n   🌲 Top Influential Features (XGBoost importance):")
    for feature, score in top.items():
        print(f"   - {feature}: {score:.4f}")

    # Keep the key name 'top_correlations' so the rest of the pipeline still works
    # (it's now "importance", not correlation, but the downstream code doesn't care).
    return {"top_correlations": top.to_dict()}


def generate_pdf_report_node(state: AgentState):
    """Generates LaTeX code via LLM and compiles it to PDF."""
    town = state['target_town']
    correlations = state['top_correlations']
    
    print(f"--- Node 5: Generating PDF Report for {town} ---")
    
    if not correlations:
        print("No correlations found to report on.")
        return

    # 1. Prepare Data
    data_summary = "\n".join([f"- {feature}: {score:.4f} correlation" for feature, score in correlations.items()])

    # 2. Prompt LLM
    llm = ChatOpenAI(model="gpt-4.1", temperature=0)
    
    prompt = textwrap.dedent(f"""
        You will be given:
        1. The name of a town.
        2. The top features with the highest feature importance in predicting its safety index.
        3. Approximate numeric values for those features for that specific town.

        Your task is to produce a COMPLETE LaTeX document, from \\documentclass to \\end{{document}}, which can be compiled with pdflatex into a PDF report.

        GENERAL REQUIREMENTS
        - Language: English.
        - Title: "Safety Profile and Improvement Plan for {town}".
        - Use the standard article class and only standard LaTeX packages (no custom .sty files).
        - The output MUST be pure LaTeX: no Markdown fences, no backticks, no extra commentary outside the LaTeX.
        - Do NOT invent specific crime statistics or exact numeric values for the town that are not provided. Use only qualitative descriptions (e.g. "relatively high", "moderate", "low").

        DOCUMENT STRUCTURE

        Abstract:
        - Provide a concise, professional summary of the town’s current safety situation, based on the provided features and their feature importance in predicting the safety index.

        Section 1: Overview of the safety index and methodology
        - Explain what the safety index conceptually represents.
        - Describe that the analysis is based on a predictive model of the safety index and the associated feature importance scores derived from a dataset of similar or safer towns.
        - Clarify that the goal is to understand how the town compares and to identify targeted strategies to improve safety.

        Section 2: Detailed analysis of the most influential features
        - Focus on all the provided features (x in total), which are those with the highest feature importance scores with respect to the safety index.
        - For each of the provided features:
            * Provide a user-friendly, human-readable title. Do NOT simply repeat the raw feature name.
            * Explain what the feature represents in practical, everyday terms for residents and policymakers.
            * Explain what higher and lower values of this feature typically indicate in terms of urban conditions, risks, or protections.
            * Interpret the magnitude of its feature importance:
                - Explain what it means for this feature to be more or less influential in predicting the safety index compared with the other features.
                - If directional information about its relationship to safety is available in the description (e.g. higher values tend to be associated with safer or less safe conditions), incorporate that qualitatively. Otherwise, remain non-committal about direction and focus on its influence.
            * Discuss what the approximate current value for {town} suggests about its safety situation, using only qualitative descriptions (e.g. "relatively elevated compared to typical towns", "moderate level", "low level").
            * Do NOT introduce any new numerical data not provided in the input.

        Section 3: Actionable recommendations
        - For each of the provided features:
            * Propose concrete, realistic policies or interventions that local authorities or communities could implement to influence that feature in a way that improves safety.
            * Recommendations must be specific and actionable (e.g. "expand targeted street lighting in high-traffic pedestrian corridors", "increase community-based patrols around transit hubs", "implement traffic-calming measures on specific categories of roads").
            * Avoid vague or generic advice; focus on interventions that plausibly affect the underlying feature and, through it, the safety index.

        Section 4: Prioritized action plan
        - Rank all the provided features by how impactful they are likely to be if addressed, taking into account:
            * The strength of their feature importance in the predictive model of the safety index.
            * Feasibility and cost of interventions.
            * Expected speed of impact.
        - For each feature in priority order:
            * State its priority (e.g. highest, high, medium).
            * Assign a timeline classification: short-term, medium-term, or long-term.
            * Describe the main course of action: what should be done first, and why.
            * Define at least 3 concrete solutions or initiatives for that feature that can meaningfully impact the problem presented.
            * Summarize the expected outcomes in qualitative terms (e.g. "reduced perception of disorder", "lower risk of nighttime victimization", "improved traffic safety in residential areas"), without inventing specific statistical impacts.

        STYLE AND TONE
        - Use professional, formal language suitable for policymakers, urban planners, and public safety officials.
        - Ensure the document is logically structured, with clear subsections and coherent narrative flow.
        - Keep explanations accessible but not overly technical; definitions should be understandable to non-specialists.

        OUTPUT FORMAT
        - The final answer must be a complete LaTeX document, from \\documentclass to \\end{{document}}.
        - Do NOT include any Markdown syntax or backticks in your answer.
        - Output only the LaTeX code, nothing else.

        INPUT VARIABLES (for your reference):
        - Town name:
        {town}

        - Top features (with feature importance scores for safety_index and approximate values for this town):
        {data_summary}
    """)

    
    print("   ...Querying LLM for LaTeX code...")
    response = llm.invoke(prompt)
    latex_code = response.content

    # 3. Clean Output
    match = re.search(r'```latex(.*?)```', latex_code, re.DOTALL)
    if match:
        latex_code = match.group(1).strip()
    else:
        latex_code = latex_code.replace("```latex", "").replace("```", "")

    # 4. Save .tex
    filename_base = f"{town.replace(' ', '_')}_safety_report"
    tex_filename = f"{filename_base}.tex"
    pdf_filename = f"{filename_base}.pdf"

    with open(tex_filename, "w", encoding="utf-8") as f:
        f.write(latex_code)
    
    print(f"   ...Saved {tex_filename}")

    try:
        # 🛑 UPDATED PATH BELOW 🛑
        pdflatex_path = r"C:\texlive\2025\bin\windows\pdflatex.exe"
        
        print(f"   ...Compiling PDF using: {pdflatex_path}")
        
        # Run pdflatex using the direct path
        subprocess.run(
            [pdflatex_path, "-interaction=nonstopmode", tex_filename], 
            check=True, 
            stdout=subprocess.DEVNULL
        )
        print(f"   ✅ Report generated successfully: {pdf_filename}")
        
        # Open PDF
        if os.name == 'nt': 
            os.startfile(pdf_filename)
        elif os.name == 'posix': 
            subprocess.call(('open', pdf_filename))

    except FileNotFoundError:
        print(f"   ❌ Error: The system could not find the file at: {pdflatex_path}")
        print("   Please check if the version year (2025) or path is correct.")
    except subprocess.CalledProcessError:
        print("   ❌ Error during LaTeX compilation. Check the .log file created in the folder.")

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

if __name__ == "__main__":
    print("Initializing Database...")
    init_db()
    
    town_input = input("Enter a town name: ")
    print(f"🚀 Starting Parallel Agent for: {town_input}")
    app.invoke({"target_town": town_input})