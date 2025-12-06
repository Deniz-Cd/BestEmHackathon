import pandas as pd
import json

data = "fake_towns.json"

def identify_key_safety_drivers(json_data: str):
    """
    Analyzes a dataset of towns to find which features influence the 'safety_index' the most.
    Input: A JSON string containing data for multiple towns.
    Output: A JSON string listing the top 3 factors that impact safety.
    """
    # 1. Load data into a DataFrame
    try:
        df = pd.read_json(json_data)
    except ValueError:
        return "Error: Invalid data format."

    # 2. Check if we have enough data
    if len(df) < 5:
        return "Not enough data points to run statistical significance tests."

    # 3. Preprocessing: Ensure we only look at numeric columns
    # We drop 'town_name' or other text columns for the math part
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    # Check if 'safety_index' exists
    if 'safety_index' not in numeric_df.columns:
        return "Error: 'safety_index' column missing."

    # 4. THE STATISTIC TEST: Correlation Matrix
    # We calculate how every column relates to 'safety_index'
    corr_matrix = numeric_df.corr()
    
    # We extract just the safety correlations and drop the safety_index itself (it correlates 1.0 with itself)
    safety_correlations = corr_matrix['safety_index'].drop('safety_index')

    # 5. Filter for "Relevance"
    # We want high absolute values. -0.9 (high unemployment drops safety) is just as important as 0.9 (good lighting raises safety)
    top_features = safety_correlations.abs().sort_values(ascending=False).head(3)
    
    # 6. Format the output for the Agent
    results = {}
    for feature_name, score in top_features.items():
        # Get the original signed value (+ or -) to tell the agent the direction
        raw_correlation = safety_correlations[feature_name]
        
        impact_type = "Positive" if raw_correlation > 0 else "Negative"
        
        # Translate math to English for the Agent
        explanation = (
            f"Strong {impact_type} impact. "
            f"Correlation: {raw_correlation:.2f}. "
        )
        results[feature_name] = explanation

    return json.dumps(results)

print(identify_key_safety_drivers(data))