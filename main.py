import os
import gzip
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import shap
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

# Directories and constants
DOWNLOAD_DIR = "./tmp_geo_data/"
CANCER_GEO_IDS = ["GSE15008", "GSE10072", "GSE76722", "GSE40515"]

# Ensure download directory exists
if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)

# Download GEO data
@st.cache_data
def download_geo_data(gsename):
    """
    Downloads a GEO dataset series matrix file.
    """
    try:
        series_prefix = gsename[0:5]
        url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{series_prefix}nnn/{gsename}/matrix/{gsename}_series_matrix.txt.gz"
        filepath = os.path.join(DOWNLOAD_DIR, f"{gsename}.gz")
        st.info(f"Attempting to download **{gsename}** from `{url}`...")
        urllib.request.urlretrieve(url, filepath)
        st.success(f"Successfully downloaded **{gsename}** to `{filepath}`")
        return filepath
    except urllib.error.URLError as e:
        st.error(f"Network error or invalid URL while downloading **{gsename}**: {e}. Please check your internet connection or verify the GEO ID.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while downloading GEO data for **{gsename}**: {e}")
        return None

# Preprocess data
@st.cache_data
def preprocess_geo_data(filepath):
    """
    Preprocesses the downloaded GEO data file, handling common encoding and parsing issues.
    """
    try:
        st.info(f"Processing data from **{os.path.basename(filepath)}**...")
        with gzip.open(filepath, 'rt', encoding='latin-1') as f:
            lines = []
            data_started = False
            for line in f:
                if "!series_matrix_table_begin" in line:
                    data_started = True
                    continue
                if "!series_matrix_table_end" in line:
                    break
                if data_started:
                    lines.append(line)
            if not lines:
                st.error("No data found between '!series_matrix_table_begin' and '!series_matrix_table_end'. The file format might be unexpected.")
                return None
            data = pd.read_csv(pd.io.common.StringIO("".join(lines)), delimiter="\t", index_col=0)
        data = data.dropna(axis=1)
        data = data.transpose()
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        data = data.dropna()
        if data.empty:
            st.warning("Processed data is **empty** after cleaning. This might mean the file format is highly unusual, or it contains no valid numeric expression data.")
            return None
        st.success("Data preprocessing complete!")
        st.write(f"Processed Data Shape (samples, genes): **{data.shape[0]} samples, {data.shape[1]} genes**")
        st.markdown("---")
        st.subheader("Preview of Processed Gene Expression Data (First 5 Samples):")
        st.dataframe(data.head())
        return data
    except pd.errors.EmptyDataError:
        st.error(f"The data file **{os.path.basename(filepath)}** appears empty or has no data to parse. Please check its content.")
        return None
    except ValueError as e:
        st.error(f"**Data parsing error** for **{os.path.basename(filepath)}**: {e}. This often indicates an unexpected file structure.")
        return None
    except Exception as e:
        st.error(f"An **unexpected error** occurred during data preprocessing for **{os.path.basename(filepath)}**: {e}")
        return None

# Select top genes using SHAP
@st.cache_data(show_spinner="Selecting top genes with SHAP...")
def select_top_genes(data, labels):
    """
    Selects top genes based on SHAP values from a RandomForestClassifier.
    Requires meaningful 'labels' for proper gene selection.
    """
    if data.empty:
        st.warning("Input data is empty; cannot select top genes.")
        return pd.DataFrame()
    if len(np.unique(labels)) < 2:
        st.warning("Only one class found in labels. SHAP gene selection relies on at least two distinct groups (e.g., control vs. disease). Skipping SHAP and returning original data.")
        return data
    try:
        st.info("Training RandomForestClassifier for SHAP gene selection (this may take a moment)...")
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(data, labels)
        explainer = shap.TreeExplainer(model)
        shap_values_raw = explainer.shap_values(data)
        if isinstance(shap_values_raw, list) and len(shap_values_raw) > 1:
            shap_values = np.abs(shap_values_raw[1]).mean(0)
        else:
            shap_values = np.abs(shap_values_raw).mean(0)
        if len(shap_values) != data.shape[1]:
            st.warning(f"Mismatch detected between SHAP values length ({len(shap_values)}) and data columns ({data.shape[1]}). This might indicate an internal SHAP calculation issue. Returning original data as fallback.")
            return data
        top_gene_indices = np.argsort(shap_values)[-50:]
        top_genes_df = data.iloc[:, top_gene_indices]
        st.success(f"Successfully selected **{len(top_gene_indices)} top genes** based on SHAP importance!")
        st.write("Selected top genes (first 5 shown):", top_genes_df.columns.tolist()[:5])
        st.markdown("---")
        return top_genes_df
    except Exception as e:
        st.error(f"Error selecting top genes using SHAP: {e}")
        st.info("Returning original data as a fallback due to an error in SHAP selection.")
        return data

# Generate synthetic data using autoencoder
@st.cache_data(show_spinner="Generating synthetic data with Autoencoder...")
def generate_synthetic_data(data):
    """
    Generates synthetic data using an autoencoder.
    Normalizes data before training and denormalizes after.
    """
    if data.empty:
        st.warning("Input data for synthetic generation is empty. Skipping synthetic data generation.")
        return pd.DataFrame()
    try:
        st.info("Normalizing data for autoencoder training...")
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)
        input_dim = data_scaled.shape[1]
        inputs = Input(shape=(input_dim,))
        encoded = Dense(64, activation="relu")(inputs)
        encoded = Dense(32, activation="relu")(encoded)
        decoded = Dense(64, activation="relu")(encoded)
        outputs = Dense(input_dim, activation="sigmoid")(decoded)
        autoencoder = Model(inputs, outputs)
        autoencoder.compile(optimizer="adam", loss="mse")
        st.info("Training autoencoder (this might take a few seconds)...")
        autoencoder.fit(data_scaled, data_scaled, epochs=100, batch_size=32, verbose=0)
        st.info("Generating synthetic data from the trained autoencoder...")
        synthetic_scaled_data = autoencoder.predict(data_scaled, verbose=0)
        synthetic_data = scaler.inverse_transform(synthetic_scaled_data)
        synthetic_df = pd.DataFrame(synthetic_data, columns=data.columns, index=data.index)
        st.success("Synthetic data generation complete!")
        st.write(f"Synthetic Data Shape: **{synthetic_df.shape[0]} samples, {synthetic_df.shape[1]} genes**")
        st.markdown("---")
        return synthetic_df
    except Exception as e:
        st.error(f"Error generating synthetic data with autoencoder: {e}")
        st.info("Returning original data as a fallback due to an error in synthetic data generation.")
        return data

# Volcano plot with Streamlit
def plot_volcano(data, labels=None):
    """
    Generates a Volcano Plot.
    If actual labels are provided, it's a placeholder for real differential expression.
    Otherwise, it uses dummy p-values.
    """
    if data.empty:
        st.warning("No data available to generate Volcano Plot.")
        return
    try:
        st.info("Calculating fold changes and p-values for Volcano Plot...")
        fold_changes = pd.Series([], dtype=float)
        p_values = pd.Series([], dtype=float)
        if labels is not None and len(np.unique(labels)) >= 2:
            unique_labels = np.unique(labels)
            if len(unique_labels) == 2:
                group1_data = data[labels == unique_labels[0]]
                group2_data = data[labels == unique_labels[1]]
                mean_group1 = group1_data.mean(axis=0).replace(0, np.nan)
                mean_group2 = group2_data.mean(axis=0).replace(0, np.nan)
                fold_changes = np.log2((mean_group2 + 1e-9) / (mean_group1 + 1e-9))
                p_values = pd.Series(np.random.uniform(1e-5, 0.1, len(fold_changes)), index=fold_changes.index)
                st.warning("P-values in the Volcano Plot are currently **randomly generated**. For a scientifically meaningful analysis, you would perform a proper statistical differential expression test (e.g., t-test, DESeq2, Limma) and use those calculated p-values.")
            else:
                st.warning("More than two unique labels were provided for Volcano Plot, or labels are not suitable for a simple two-group comparison. Using overall mean fold change and dummy p-values.")
                fold_changes = np.log2(data.mean(axis=0).replace(0, np.nan) + 1e-9)
                p_values = pd.Series(np.random.uniform(1e-5, 0.1, len(fold_changes)), index=fold_changes.index)
        else:
            st.warning("Volcano Plot is using simplified fold changes (overall mean) and dummy p-values because no valid labels were provided for group comparison.")
            fold_changes = np.log2(data.mean(axis=0).replace(0, np.nan) + 1e-9)
            p_values = pd.Series(np.random.uniform(1e-5, 0.1, len(fold_changes)), index=fold_changes.index)
        valid_indices = ~fold_changes.isna()
        fold_changes = fold_changes[valid_indices]
        p_values = p_values[valid_indices]
        if fold_changes.empty or p_values.empty:
            st.warning("No valid fold changes or p-values could be calculated for the Volcano Plot. This might happen if all gene expression values are zero or very similar across samples.")
            return
        p_values = p_values.replace(0, np.finfo(float).eps)
        neg_log10_p_values = -np.log10(p_values)
        plt.figure(figsize=(10, 7))
        sns.scatterplot(x=fold_changes, y=neg_log10_p_values, s=20, alpha=0.7)
        p_value_threshold = 0.05
        log2_fc_threshold = 1.0
        plt.axhline(y=-np.log10(p_value_threshold), color="red", linestyle="--", label=f"P-value = {p_value_threshold}")
        plt.axvline(x=log2_fc_threshold, color="blue", linestyle="--", label=f"Log2 FC = {log2_fc_threshold}")
        plt.axvline(x=-log2_fc_threshold, color="blue", linestyle="--")
        plt.xlabel("Log2 Fold Change")
        plt.ylabel("-Log10 P-value")
        plt.title("Volcano Plot of Gene Expression")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        st.pyplot(plt.gcf())
        plt.close()
        st.markdown("---")
    except Exception as e:
        st.error(f"Error generating Volcano Plot: {e}")

# Streamlit interface
def create_web_interface():
    st.set_page_config(layout="wide", page_title="Cancer Gene Expression Explorer")
    st.title("🧬 Cancer Gene Expression Explorer")
    st.markdown("""
    This application allows you to explore gene expression data from GEO datasets,
    select top genes using SHAP, generate synthetic data with an autoencoder,
    and visualize potential differentially expressed genes via a Volcano Plot.

    **Note:** For demonstration purposes, sample labels and p-values for the Volcano Plot
    are currently generated randomly. For real analysis, these would be derived from
    your experimental design and statistical tests.
    """)
    st.sidebar.header("Configuration")
    selected_geo = st.sidebar.selectbox("Choose GEO Dataset:", CANCER_GEO_IDS)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Sample Labels (Placeholder)")
    st.sidebar.info("For a real analysis, you would load or define actual sample groups (e.g., 'control', 'disease') here to provide meaningful labels for SHAP and Volcano Plot.")
    if st.sidebar.button("Analyze Data"):
        st.subheader(f"Analyzing Dataset: **{selected_geo}**")
        file_path = download_geo_data(selected_geo)
        if file_path:
            with st.spinner("Step 1/4: Preprocessing data..."):
                data = preprocess_geo_data(file_path)
            if data is not None and not data.empty:
                st.warning("Currently, sample labels are generated randomly for demonstration. **For meaningful biological results, replace this with your actual sample group labels (e.g., 'cancer' vs. 'normal')!**")
                labels = np.random.randint(0, 2, size=(data.shape[0],))
                with st.spinner("Step 2/4: Selecting top genes using SHAP..."):
                    filtered_data = select_top_genes(data, labels)
                if not filtered_data.empty:
                    with st.spinner("Step 3/4: Generating synthetic data with autoencoder..."):
                        synthetic_data = generate_synthetic_data(filtered_data)
                    if not synthetic_data.empty:
                        st.subheader("📊 Volcano Plot (using synthetic data)")
                        plot_volcano(synthetic_data, labels)
                        st.subheader("📈 Visualize Individual Gene Expression (from synthetic data)")
                        st.markdown("Select a gene name from the dropdown to see its expression levels across samples in the synthetic dataset.")
                        gene_options = synthetic_data.columns.tolist()
                        gene_query = st.selectbox("Select Gene Name to visualize expression:", [""] + gene_options, help="Choose a gene from the list of top genes.")
                        if gene_query:
                            if gene_query in synthetic_data.columns:
                                st.write(f"Expression of **{gene_query}** (from synthetic data):")
                                gene_expression_df = pd.DataFrame(synthetic_data[gene_query])
                                st.line_chart(gene_expression_df)
                            else:
                                st.warning(f"Gene '**{gene_query}**' not found in the selected top genes. Please choose from the dropdown.")
                    else:
                        st.error("Synthetic data could not be generated or is empty. Please check previous steps.")
                else:
                    st.error("No top genes were selected, or the filtered data is empty. Cannot proceed with synthetic data generation or plotting.")
            else:
                st.error("Data could not be preprocessed or is empty after cleaning. Analysis stopped.")
        else:
            st.error("Could not download the selected GEO dataset. Analysis stopped.")

# Run the Streamlit app
if __name__ == "__main__":
    create_web_interface()
