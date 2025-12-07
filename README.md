# 🛡️ SafeCity Agent

**SafeCity Agent** is an AI-driven urban analytics tool designed to evaluate city safety. Unlike standard statistical tools, it uses a "Compare and Prescribe" approach: it identifies culturally similar but safer "neighbor" cities, analyzes the specific crime metrics driving the difference using Machine Learning, and generates an automated, professional PDF policy report.

## 🚀 Key Features

* **Contextual Benchmarking:** Uses GPT-4o to identify cities with similar culture, religion, and laws to ensure comparisons are realistic (e.g., comparing Bucharest to Warsaw rather than Tokyo).
* **Real-Time Data Acquisition:** Automated headless browser scraping (Selenium) gathers 15+ specific safety metrics (e.g., "worry of mugging," "corruption") from Numbeo.
* **3D Visualization:** Interactive PyDeck map rendering target cities and drawing connection arcs to safer alternatives.
* **Root Cause Analysis:** Utilizes **XGBoost Regression** to determine which specific safety factors have the highest feature importance in predicting the Safety Index.
* **Automated Reporting:** Generates a full LaTeX document containing actionable policy recommendations and compiles it into a PDF.

## ⚙️ Architecture & Workflow

The application uses **LangGraph** to orchestrate a state-based workflow:

1.  **Check DB:** Queries a local SQLite cache (`towns_detailed.db`) to see if the city exists.
2.  **Scrape Target:** If missing, scrapes the target city's data using Selenium.
3.  **Find Neighbors:** Queries an LLM for similar cities, filters for those that are safer, and scrapes them in parallel using `ThreadPoolExecutor`.
4.  **Analyze:** Trains an XGBoost model on the dataset to identify top correlations and filters features where the target city performs worse than the average.
5.  **Generate Report:** Uses an LLM to write a LaTeX report based on the analysis and compiles it via `pdflatex`.

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Orchestration:** LangGraph, LangChain
* **AI Models:** OpenAI GPT-4o (Logic) & GPT-4.1 (Report Writing)
* **Data Science:** Pandas, XGBoost, Numpy
* **Scraping:** Selenium (Chrome WebDriver)
* **Visualization:** PyDeck
* **Database:** SQLite3
* **Reporting:** LaTeX (TeX Live)

## 📝 Prerequisites

Before running the application, ensure you have the following installed:

1.  **Python 3.9+**
2.  **Google Chrome** (for Selenium scraping)
3.  **LaTeX Distribution:** You must have `pdflatex` installed.
    * *Note:* The code currently defaults to a Windows path: `C:\texlive\2025\bin\windows\pdflatex.exe`. You may need to adjust this path in `app.py` line 335 if you are on Linux/Mac or have a different installation path.

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Deniz-Cd/safecity-agent.git](https://github.com/Deniz-Cd/safecity-agent.git)
    cd safecity-agent
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up Environment Variables:**
    Create a `.env` file in the root directory and add your OpenAI API key:
    ```env
    OPENAI_API_KEY=sk-proj-...
    ```

4.  **Configure LaTeX Path:**
    Open `app.py` and ensure the `pdflatex_path` variable matches your system's location of `pdflatex`.

## ▶️ Usage

Run the Streamlit application:

```bash
streamlit run app.py