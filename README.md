# 🧠 Natural Language Data Visualizer

A **Streamlit app** that lets you explore, clean, and visualize datasets using **natural language prompts** — powered by an LLM (like CodeLlama or LM Studio).

Upload your dataset, clean it automatically or manually, and ask questions like:

> “Show average sales by region”  
> “Plot total revenue over time”  
> “Visualize gender distribution”

The app automatically:
- Cleans and preprocesses your data
- Saves cleaned versions persistently (so refreshes don’t lose progress)
- Generates smart visualizations with Python + Seaborn + Matplotlib
- Interprets natural language queries using a local LLM

## 🚀 Features

✅ **Automatic + Manual Cleaning**
- Removes duplicates, fills missing values, converts columns to numeric  
- Saves cleaned datasets as `<dataset_name>_cleaned_1.csv`, `<dataset_name>_cleaned_2.csv`, etc.

✅ **Persistent Data**
- Reloads the last cleaned dataset automatically after refresh

✅ **Natural Language Visualization**
- Describe visuals in plain English  
- LLM generates valid pandas/seaborn code  
- Supports groupby aggregations (mean, sum, count, median)

✅ **Local + Private**
- Runs entirely on your machine using [LM Studio](https://lmstudio.ai/) or any OpenAI-compatible local endpoint  
- No cloud dependency or API keys required

---

## 🧩 Setup Instructions

### Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/nl-data-explorer.git
cd nl-data-explorer

### Activate Virtual Environment
python -m venv venv
venv\Scripts\activate       # On Windows
# source venv/bin/activate  # On macOS/Linux

### Install Dependencies
pip install -r requirements.txt

### How to Run
Make sure your LM Studio or local LLM endpoint is running on
http://localhost:1234/v1/chat/completions.

Then launch the Streamlit app:
streamlit run app.py

Open http://localhost:8501 in your browser.
