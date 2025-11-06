import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import requests
import traceback
import os
import re
from sklearn.preprocessing import StandardScaler

# -------------------------
# Configuration
# -------------------------
LM_STUDIO_CHAT_URL = "http://localhost:1234/v1/chat/completions"
LM_MODEL = "codellama-7b-instruct"
REQUEST_TIMEOUT = 120


# -------------------------
# Helper: call LM Studio
# -------------------------
def call_lm_chat(messages, max_tokens=1024, temperature=0.0):
    payload = {
        "model": LM_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    resp = requests.post(LM_STUDIO_CHAT_URL, json=payload, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    if "choices" in data and len(data["choices"]) > 0:
        choice = data["choices"][0]
        if "message" in choice and "content" in choice["message"]:
            return choice["message"]["content"]
        elif "text" in choice:
            return choice["text"]
    return json.dumps(data)


# -------------------------
# Load Dataset
# -------------------------
def load_csv():
    st.subheader("📂 Upload Your Dataset")
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.success("✅ File uploaded successfully!")
            st.dataframe(df.head())
            st.write(f"**Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")
            return df, uploaded_file.name
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
    return None, None


# -------------------------
# Automatic Cleaning
# -------------------------
def automatic_cleaning(df):
    st.subheader("🧹 Automatic Cleaning Process")
    log = []
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if before != after:
        log.append(f"Removed {before - after} duplicate rows")

    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = pd.to_numeric(df[col])
                log.append(f"Converted column '{col}' from string to numeric")
            except:
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    df[col] = df[col].fillna("Unknown")
                    log.append(f"Filled {missing_count} missing in '{col}' with 'Unknown'")
        # else numeric → optionally fill here

    if log:
        st.write("**Automatic Cleaning Summary:**")
        for step in log:
            st.write(f"- {step}")
    else:
        st.info("No automatic cleaning required.")
    return df


# -------------------------
# Manual Cleaning
# -------------------------
def manual_cleaning(df, base_name):
    st.subheader("🧭 Manual Cleaning Options")
    missing_cols = df.columns[df.isnull().any()].tolist()
    if not missing_cols:
        st.info("No missing values found.")
        return df

    st.write("Columns with missing values:", missing_cols)
    user_choices = {}

    for col in missing_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            choice = st.selectbox(
                f"Handle missing values in numeric column '{col}':",
                ["Mean", "Median", "0 (Zero)"],
                key=f"clean_{col}",
            )
        else:
            choice = "Unknown"
        user_choices[col] = choice

    if st.button("✅ Apply Manual Cleaning"):
        for col, choice in user_choices.items():
            missing_count = df[col].isnull().sum()
            if missing_count == 0:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                if choice == "Mean":
                    df[col] = df[col].fillna(df[col].mean())
                elif choice == "Median":
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna("Unknown")
            st.write(f"→ Filled {missing_count} missing values in '{col}'")

        cleaned_path = get_next_cleaned_filename(base_name)
        df.to_csv(cleaned_path, index=False)
        st.success(f"✅ Manual cleaning completed and saved as `{cleaned_path}`")
        st.dataframe(df.head(10))
    return df


# -------------------------
# Generate Next Cleaned Filename
# -------------------------
def get_next_cleaned_filename(original_filename):
    base_name = os.path.splitext(original_filename)[0]
    existing_files = [f for f in os.listdir() if f.startswith(base_name + "_cleaned_")]
    if not existing_files:
        version = 1
    else:
        # Extract version numbers and find max
        versions = [
            int(re.search(r"_cleaned_(\d+)", f).group(1))
            for f in existing_files
            if re.search(r"_cleaned_(\d+)", f)
        ]
        version = max(versions) + 1
    return f"{base_name}_cleaned_{version}.csv"


def get_latest_cleaned_file(base_name):
    files = [f for f in os.listdir() if f.startswith(base_name + "_cleaned_")]
    if not files:
        return None
    versions = [
        (f, int(re.search(r"_cleaned_(\d+)", f).group(1)))
        for f in files
        if re.search(r"_cleaned_(\d+)", f)
    ]
    latest_file = sorted(versions, key=lambda x: x[1], reverse=True)[0][0]
    return latest_file


# -------------------------
# LLM Visualization
# -------------------------
def visualize_with_llm(df, prompt):
    try:
        df_sample = df.head(10).to_csv(index=False)
        llm_prompt = f"""
        You are a data visualization expert.
        The variable 'df' already contains the dataset.
        Do NOT read or write any files.
        Do NOT call plt.show().
        Use matplotlib/seaborn to create a figure named 'fig'.

        The user asked: "{prompt}"

        Dataset sample:
        {df_sample}
        
        VERY IMPORTANT RULES:
        1. If the prompt mentions 'average', 'mean', 'median', 'sum', or 'count', first perform a proper pandas groupby aggregation.
        Example:
            df_grouped = df.groupby('column_name')['target_column'].mean().reset_index()
            or
            df_grouped = df.groupby('column_name', as_index=False)['target_column'].sum()
        Then visualize df_grouped using sns.barplot, sns.lineplot, etc.
        2. Never compute an average manually using loops; always use pandas aggregation.
        3. Do not use %matplotlib inline or %matplotlib notebook in your code.
        4. If multiple columns are mentioned, choose the most logical one as category (x) and numeric one as y.
        5. Always label axes clearly and add a descriptive title.
        6. Return ONLY valid Python code (no explanations or markdown).
        7. Always create a figure as:
                fig, ax = plt.subplots(figsize=(8,8))

        """
        text = call_lm_chat(
            [
                {"role": "system", "content": "You create clean, runnable visualization code only."},
                {"role": "user", "content": llm_prompt},
            ],
            max_tokens=512,
        )
        import numpy as np
        code_blocks = re.findall(r"```(?:python)?\s*([\s\S]*?)```", text)
        cleaned = code_blocks[0] if code_blocks else text
        banned = ["pd.read_csv", "open(", "os.", "subprocess", "plt.show(", "%"]
        cleaned = "\n".join(
            line.lstrip() for line in cleaned.splitlines()
            if line.strip() and not any(b in line for b in banned)
        )
        st.code(cleaned, language="python")
        local_vars = {"df": df, "plt": plt, "sns": sns, "np": np, "pd": pd}
        cleaned = re.sub(r"^%.*", "", cleaned, flags=re.MULTILINE)
        exec(cleaned, {}, local_vars)
        fig = local_vars.get("fig")

        # 🛡️ Safety fix: handle cases where fig is actually an Axes, not a Figure
        if fig is None and "ax" in local_vars:
            ax = local_vars["ax"]
            if hasattr(ax, "figure"):
                fig = ax.figure

        elif hasattr(fig, "figure"):  # fig might actually be an Axes
            fig = fig.figure

        # ✅ Final fallback
        if fig is None:
            st.warning("⚠️ No figure generated. Try refining your prompt.")
            return

        st.pyplot(fig)
    except Exception as e:
        st.error(f"Visualization failed: {e}")
        st.write(traceback.format_exc())

# -------------------------
# Main App
# -------------------------
def main():
    st.title("🧠 Natural Language Data Visualizer")

    # Load dataset
    df, dataset_name = load_csv()
    if df is None:
        return

    base_name = os.path.splitext(dataset_name)[0]

    # Check for latest cleaned file
    latest_cleaned = get_latest_cleaned_file(base_name)
    if latest_cleaned:
        st.success(f"Loaded latest cleaned dataset: `{latest_cleaned}` ✅")
        df = pd.read_csv(latest_cleaned)
        st.dataframe(df.head())
    else:
        # Perform cleaning steps and save as first version
        df = automatic_cleaning(df)
        df = manual_cleaning(df, dataset_name)

    # Visualization Section
    st.subheader("📊 Create Visuals from Cleaned Data")
    prompt = st.text_area("Enter your visualization prompt")
    if st.button("Generate Visualization"):
        visualize_with_llm(df, prompt)


if __name__ == "__main__":
    main()