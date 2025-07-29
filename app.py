import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import zscore
import io
import matplotlib.pyplot as plt
import seaborn as sns



# === 1. PAGE CONFIGURATION ===
st.set_page_config(
    page_title="Automated Data Explorer",
    page_icon="📊",
    layout="wide",
)



# === 2. SESSION STATE INITIALIZATION ===
if 'app_state' not in st.session_state:
    st.session_state.app_state = "upload"
if 'data' not in st.session_state:
    st.session_state.data = None
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None



# === 3. FUNCTION DEFINITIONS FOR EACH APP STAGE ===

# STAGE 1: FILE UPLOAD
def upload_and_process_file():
    st.header("1. Upload Your Dataset")

    uploaded_file = st.file_uploader(
        "Choose a CSV, XLSX, TSV, JSON, or Parquet file (Max 50MB)",
        type=["csv", "xlsx", "tsv", "json", "parquet"]
    )

    if uploaded_file is not None:
        file_size = uploaded_file.size
        max_file_size_mb = 50
        max_file_size_bytes = max_file_size_mb * 1024 * 1024

        if file_size > max_file_size_bytes:
            st.warning(f"File size ({file_size / (1024*1024):.2f}MB) exceeds {max_file_size_mb}MB. Trimming file.")
            
            try:
                
                if uploaded_file.name.endswith(('.csv', '.tsv')):
                    bytes_data = uploaded_file.read(max_file_size_bytes)
                    string_data = bytes_data.decode('utf-8', errors='ignore')
                    last_newline = string_data.rfind('\n')
                    
                    if last_newline != -1:
                        string_data = string_data[:last_newline]
                    
                    string_io_data = io.StringIO(string_data)
                    separator = '\t' if uploaded_file.name.endswith('.tsv') else ','
                    df = pd.read_csv(string_io_data, sep=separator)
                    st.info(f"File was trimmed to approximately {len(bytes_data)/(1024*1024):.2f} MB.")
                
                else:
                    st.error("Automatic trimming is only supported for CSV/TSV. Please upload a smaller file.")
                    return
            
            except Exception as e:
                st.error(f"Error processing trimmed file: {e}")
                return
        
        else:
            
            try:
                uploaded_file.seek(0)
                file_extension = uploaded_file.name.split('.')[-1].lower()
                
                if file_extension == 'csv':
                    df = pd.read_csv(uploaded_file)
                elif file_extension == 'xlsx':
                    df = pd.read_excel(uploaded_file)
                elif file_extension == 'tsv':
                    df = pd.read_csv(uploaded_file, sep='\t')
                elif file_extension == 'json':
                    df = pd.read_json(uploaded_file)
                elif file_extension == 'parquet':
                    df = pd.read_parquet(uploaded_file)
                else:
                    st.error("Unsupported file type.")
                    return
            
            except Exception as e:
                st.error(f"Error reading file: {e}")
                return

        st.session_state.data = df
        st.session_state.cleaned_data = df.copy()
        st.session_state.app_state = "cleaning"
        st.rerun()

# STAGE 2: DATA CLEANING
def data_cleaning_and_transformation():
    """Handles the interactive data cleaning and transformation stage within the main page."""
    st.header("2. Data Transformation and Cleaning")
    
    if 'cleaned_data' not in st.session_state or st.session_state.cleaned_data is None:
        st.warning("No data found. Please upload a file first.")
        st.session_state.app_state = "upload"
        st.rerun()
        
    df = st.session_state.cleaned_data
    
    st.subheader("Data Preview")
    st.dataframe(df.head())
    
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Cleaning & Transformation Controls")

    # === Integrated Controls using Expanders ===
    with st.expander("Rename Columns"):
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            selected_col_to_rename = st.selectbox("Select column to rename", df.columns, index=None, key="rename_select")
        
        if selected_col_to_rename:
            
            with col2:
                new_col_name = st.text_input("New column name", selected_col_to_rename)
            
            with col3:
                st.markdown("##") # Vertical alignment hack
                
                if st.button("Rename Column"):
                    df.rename(columns={selected_col_to_rename: new_col_name}, inplace=True)
                    
                    st.session_state.cleaned_data = df
                    st.rerun()

    with st.expander("Remove Columns"):
        cols_to_remove = st.multiselect("Select columns to remove", df.columns, key="remove_cols")
        
        if st.button("Remove Selected Columns"):
            df.drop(columns=cols_to_remove, inplace=True)
            
            st.session_state.cleaned_data = df
            st.rerun()

    with st.expander("Handle Missing Values"):
        missing_value_strategy = st.selectbox("Select strategy", ["Remove Rows with Missing Values", "Fill with Mean (Numeric)", "Fill with Median (Numeric)", "Fill with Mode"], index=None, key="missing_strat")
        
        if st.button("Apply Missing Value Strategy"):
            
            if missing_value_strategy == "Remove Rows with Missing Values":
                df.dropna(inplace=True)
            
            elif missing_value_strategy is not None:
                
                for col in df.columns:
                    
                    if df[col].isnull().any():
                        
                        if pd.api.types.is_numeric_dtype(df[col]):
                            
                            if missing_value_strategy == "Fill with Mean (Numeric)":
                                fill_value = df[col].mean()
                            
                            elif missing_value_strategy == "Fill with Median (Numeric)":
                                fill_value = df[col].median()
                            
                            elif missing_value_strategy == "Fill with Mode":
                                fill_value = df[col].mode().iloc[0]
                            
                            else:
                                continue
                            df[col].fillna(fill_value, inplace=True)
                        
                        else:
                            
                            if missing_value_strategy == "Fill with Mode":
                                fill_value = df[col].mode().iloc[0]
                                df[col].fillna(fill_value, inplace=True)
            
            st.session_state.cleaned_data = df
            st.rerun()

    with st.expander("Handle Outliers"):
        numeric_cols_list = df.select_dtypes(include=np.number).columns.tolist()

        if numeric_cols_list:
            outlier_col = st.selectbox("Select a numeric column", numeric_cols_list, index=None, key="outlier_col")
            
            if outlier_col:
                z_thresh = st.slider("Z-score threshold for removal", 1.0, 4.0, 3.0, 0.1)
                
                if st.button("Remove Outliers"):
                    z_scores = np.abs(zscore(df[outlier_col], nan_policy='omit'))
                    df = df[z_scores < z_thresh]
                    
                    st.session_state.cleaned_data = df
                    st.rerun()
        
        else:
            st.info("No numeric columns available for outlier detection.")
            
    with st.expander("Handle Duplicates"):

        if st.button("Remove All Duplicate Rows"):
            initial_rows = len(df)
            df.drop_duplicates(inplace=True)

            st.session_state.cleaned_data = df
            st.success(f"Removed {initial_rows - len(df)} duplicate rows.")
            st.rerun()

    with st.expander("Change Text Case"):
        object_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if object_cols:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                case_col = st.selectbox("Select a text column", object_cols, index=None, key="case_col")
            
            if case_col:
                
                with col2:
                    case_type = st.selectbox("Select case type", ["lowercase", "UPPERCASE", "Title Case"], key="case_type")
                
                with col3:
                    st.markdown("##")
                    
                    if st.button("Apply Case Change"):
                    
                        if case_type == "lowercase":
                            df[case_col] = df[case_col].str.lower()
                    
                        elif case_type == "UPPERCASE":
                            df[case_col] = df[case_col].str.upper()
                    
                        else:
                            df[case_col] = df[case_col].str.title()
                    
                        st.session_state.cleaned_data = df
                        st.rerun()
        else:
            st.info("No text columns available to change case.")

    # === FINISH CLEANING ===
    st.markdown("<hr>", unsafe_allow_html=True)
    if st.button("I have finished cleaning my data.", type="primary"):
        st.session_state.app_state = "analysis"
        st.rerun()

# HELPER FUNCTION FOR REPORT GENERATION
def generate_report_text(df):
    """Generates a comprehensive text-based report from all analyses."""
    report_lines = []
    separator = "=" * 80 + "\n"
    
    report_lines.append(separator + "FEATURE IDENTIFICATION REPORT\n" + separator)
    report_lines.append("1. GENERAL DATASET INFORMATION\n" + "-"*40 + "\n")
    report_lines.append(f"Number of rows: {df.shape[0]}")
    report_lines.append(f"Number of columns: {df.shape[1]}")
    report_lines.append(f"Number of duplicate rows: {df.duplicated().sum()}")
    
    buffer = io.StringIO()
    df.info(buf=buffer, verbose=False)

    report_lines.append(f"Memory Usage: {buffer.getvalue().splitlines()[-2].strip()}\n")
    report_lines.append("\n2. COLUMN DATA TYPES\n" + "-"*40 + "\n" + df.dtypes.to_string() + "\n")
    report_lines.append("\n3. DESCRIPTIVE STATISTICS\n" + "-"*40 + "\n")
    
    numerical_cols = df.select_dtypes(include=np.number)
    categorical_cols = df.select_dtypes(include=['object', 'category'])
    
    if not numerical_cols.empty:
        report_lines.append("=== Numerical Columns ===\n")

        stats = numerical_cols.describe()
        stats.loc['range'] = numerical_cols.max() - numerical_cols.min()
        stats.loc['IQR'] = stats.loc['75%'] - stats.loc['25%']
        stats.loc['skew'] = numerical_cols.skew()
        stats.loc['kurtosis'] = numerical_cols.kurtosis()
        stats.loc['sum'] = numerical_cols.sum()
        stats.loc['CV'] = stats.loc['std'] / stats.loc['mean']
        stats.loc['zeros_%'] = (numerical_cols == 0).sum() / len(numerical_cols) * 100

        report_lines.append(stats.to_string() + "\n")
    
    if not categorical_cols.empty:
        report_lines.append("=== Categorical Columns ===\n"); report_lines.append(categorical_cols.describe().to_string() + "\n")
    
    report_lines.append("\n4. MISSING VALUES ANALYSIS\n" + "-"*40 + "\n")
    missing_values = df.isnull().sum()
    
    if missing_values.sum() > 0:
        missing_df = pd.DataFrame({'Missing Count': missing_values, 'Percentage (%)': (missing_values / len(df)) * 100 if len(df) > 0 else 0})
        report_lines.append(missing_df[missing_df['Missing Count'] > 0].sort_values(by='Percentage (%)', ascending=False).to_string())
    
    else:
        report_lines.append("No missing values found.")
    
    report_lines.append("\n")
    report_lines.append("\n5. CORRELATION MATRIX (NUMERICAL COLUMNS)\n" + "-"*40 + "\n")
    
    if not numerical_cols.empty and len(numerical_cols.columns) > 1:
        report_lines.append(numerical_cols.corr().to_string())
    
    else:
        report_lines.append("Not enough numerical columns to calculate correlation.")
    
    report_lines.append("\n\n" + separator + "6. DEEP-DIVE COLUMN ANALYSIS\n" + separator)
    
    for col in df.columns:
        report_lines.append(f"\nCOLUMN: '{col}' ({df[col].dtype})\n" + "."*30)
        
        if pd.api.types.is_numeric_dtype(df[col]):
            
            if not df[col].empty and df[col].notna().sum() > 0:
                z_scores = np.abs(zscore(df[col], nan_policy='omit')); outliers = z_scores > 3
                report_lines.append(f"- Outliers (Z-score > 3): {outliers.sum()} ({outliers.sum()/len(z_scores)*100:.2f}%)")
            
            else:
                report_lines.append("- Outliers: Column is empty or all NaN.")
            
            s = df[col].skew(); k = df[col].kurtosis()
            
            skew_desc = "Highly positive" if s > 1 else "Moderately positive" if s > 0.5 else "Approx. symmetric" if s > -0.5 else "Moderately negative" if s > -1 else "Highly negative"
            kurt_desc = "Heavy-tailed (Leptokurtic)" if k > 0 else "Light-tailed (Platykurtic)" if k < 0 else "Normal-like (Mesokurtic)"
            
            report_lines.append(f"- Skewness: {s:.2f} ({skew_desc})")
            report_lines.append(f"- Kurtosis: {k:.2f} ({kurt_desc})")
        
        else:
            report_lines.append(f"- Cardinality (Unique Values): {df[col].nunique()}")
            
            if df[col].nunique() > 1:
                report_lines.append("- Top 5 Most Frequent Values:\n" + df[col].value_counts(normalize=True, dropna=False).nlargest(5).to_string())
            
            if df[col].dtype == 'object' and df[col].astype(str).str.strip().ne(df[col].astype(str)).any():
                report_lines.append("- WARNING: Contains values with leading/trailing whitespace.")
    
    report_lines.append("\n\n" + separator + "END OF REPORT\n" + separator)
    return "\n".join(report_lines)

# STAGE 3: FEATURE IDENTIFICATION
def feature_identification_page():
    st.header("3. Feature Identification")

    df = st.session_state.cleaned_data
    st.subheader("General Dataset Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Dataset Shape"); st.metric("Rows", df.shape[0]); st.metric("Columns", df.shape[1])
        st.markdown("#### Duplicate Rows"); st.metric("Count", df.duplicated().sum())
    
    with col2:
        st.markdown("#### Memory Usage"); buffer = io.StringIO(); df.info(buf=buffer, verbose=False); st.text(buffer.getvalue().split('\n')[-2])
        st.markdown("#### Column Data Types"); st.dataframe(df.dtypes.astype(str).rename("Data Type"))

    st.markdown("<hr>", unsafe_allow_html=True); st.subheader("Descriptive Statistics")
    numerical_cols = df.select_dtypes(include=np.number); categorical_cols = df.select_dtypes(include=['object', 'category'])
    
    if not numerical_cols.empty:
        st.markdown("#### Numerical Columns")

        stats = numerical_cols.describe()
        stats.loc['range'] = numerical_cols.max() - numerical_cols.min()
        stats.loc['IQR'] = stats.loc['75%'] - stats.loc['25%']; stats.loc['skew'] = numerical_cols.skew()
        stats.loc['kurtosis'] = numerical_cols.kurtosis()
        stats.loc['sum'] = numerical_cols.sum()
        stats.loc['CV'] = stats.loc['std'] / stats.loc['mean']
        stats.loc['zeros_%'] = (numerical_cols == 0).sum() / len(numerical_cols) * 100

        st.dataframe(stats)
    
    if not categorical_cols.empty:
        st.markdown("#### Categorical Columns")
        st.dataframe(categorical_cols.describe())

    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Missing Values Analysis")
    missing_values = df.isnull().sum()

    if missing_values.sum() > 0:
        missing_df = pd.DataFrame({'Missing Count': missing_values, 'Percentage (%)': (missing_values / len(df)) * 100 if len(df) > 0 else 0})
        st.dataframe(missing_df[missing_df['Missing Count'] > 0].sort_values(by='Percentage (%)', ascending=False))
    
    else:
        st.success("✅ No missing values found.")

    st.markdown("<hr>", unsafe_allow_html=True); st.subheader("Correlation Matrix Heatmap")
    
    if not numerical_cols.empty and len(numerical_cols.columns) > 1:
        corr = numerical_cols.corr(); fig, ax = plt.subplots(figsize=(12, 9)); sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax); ax.set_title("Correlation Heatmap of Numerical Features"); st.pyplot(fig)
    
    else:
        st.info("A correlation heatmap requires at least two numerical columns.")

    st.markdown("<hr>", unsafe_allow_html=True); st.subheader("Deep-Dive Column Analysis")
    

    for col in df.columns:
        
        with st.expander(f"Analysis for column: '{col}' ({df[col].dtype})"):
            
            if pd.api.types.is_numeric_dtype(df[col]):
                
                st.markdown("##### Distribution"); fig, ax = plt.subplots(figsize=(6,3)); sns.histplot(df[col], kde=True, ax=ax); st.pyplot(fig)
                st.markdown("##### Outlier Analysis (using Z-score)")
                
                if not df[col].empty and df[col].notna().sum() > 0:
                    z_scores = np.abs(zscore(df[col], nan_policy='omit')); outliers = z_scores > 3
                    st.metric(label="Potential Outliers (Z-score > 3)", value=f"{outliers.sum()} ({outliers.sum()/len(z_scores)*100:.2f}%)")
                
                else:
                    st.info("Column is empty or contains only NaN values.")
                
                s = df[col].skew(); k = df[col].kurtosis()
                
                skew_desc = "Highly positive" if s > 1 else "Moderately positive" if s > 0.5 else "Approx. symmetric" if s > -0.5 else "Moderately negative" if s > -1 else "Highly negative"
                
                kurt_desc = "Heavy-tailed" if k > 0 else "Light-tailed" if k < 0 else "Normal-like tails"
                
                st.markdown(f"**Skewness:** `{s:.2f}` ({skew_desc})")
                st.markdown(f"**Kurtosis:** `{k:.2f}` ({kurt_desc})")
                
                if pd.api.types.is_integer_dtype(df[col].dtype) and df[col].max() < np.iinfo(np.int32).max and df[col].min() > np.iinfo(np.int32).min:
                    st.info("💡 Memory Saving Tip: This column could potentially be downcast to `int32`.")
            
            else:
                st.metric("Cardinality (Unique Values)", df[col].nunique())
                
                if df[col].nunique() > 1:
                    st.markdown("##### Top 5 Most Frequent Values")

                    freq_df = df[col].value_counts(normalize=True).nlargest(5).reset_index()
                    freq_df.columns = ['Value', 'Percentage']
                    freq_df['Percentage'] = freq_df['Percentage'].apply(lambda x: f"{x:.2%}")
                    
                    st.dataframe(freq_df)
                
                if df[col].dtype == 'object' and df[col].astype(str).str.strip().ne(df[col].astype(str)).any():
                    st.warning("⚠️ Contains values with leading/trailing whitespace.")
                
                if df[col].dtype == 'object' and df[col].nunique() / len(df) < 0.5:
                    st.info("💡 Memory Saving Tip: Consider converting this column to the 'category' dtype.")

    st.markdown("<hr>", unsafe_allow_html=True); report_text = generate_report_text(df)
    
    nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 1])
    
    with nav_col1:
        if st.button("Start Over"):
            for key in list(st.session_state.keys()): del st.session_state[key]
            st.session_state.app_state = "upload"
            st.rerun()
    
    with nav_col2:
        st.download_button("📄 Download Full Report (.txt)", report_text, "feature_identification_report.txt", "text/plain")
    
    with nav_col3:
        if st.button("Proceed to Data Visualization ➡️", type="primary"):
            st.session_state.app_state = "visualization"
            st.rerun()

# STAGE 4: DATA VISUALIZATION
def data_visualization_page():
    st.header("4. Data Visualization")
    
    df = st.session_state.cleaned_data
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    st.markdown("### Univariate Analysis (Single Column)")
    
    if numerical_cols:
        st.subheader("Histogram / Distribution Plot")
        hist_col = st.selectbox("Select a numerical column", numerical_cols, key="hist_col")
        
        if hist_col:
            fig, ax = plt.subplots(); sns.histplot(df[hist_col], kde=True, ax=ax)
            ax.set_title(f"Distribution of {hist_col}"); st.pyplot(fig)
    
    if categorical_cols:
        st.subheader("Bar Chart")
        bar_col = st.selectbox("Select a categorical column", categorical_cols, key="bar_col")
        
        if bar_col:
            st.bar_chart(df[bar_col].value_counts())
    
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### Bivariate Analysis (Two or More Columns)")
    st.subheader("Scatter Plot")
    
    if len(numerical_cols) >= 2:
        sc_col1, sc_col2, sc_col3 = st.columns(3)
        
        with sc_col1:
            x_ax = st.selectbox("X-axis", numerical_cols, index=0, key="x_scatter")
        
        with sc_col2:
            y_ax = st.selectbox("Y-axis", numerical_cols, index=1, key="y_scatter")
        
        with sc_col3:
            hue_ax = st.selectbox("Color by (optional)", [None] + categorical_cols, key="hue_scatter")
        
        fig, ax = plt.subplots(); sns.scatterplot(data=df, x=x_ax, y=y_ax, hue=hue_ax, ax=ax)
        ax.set_title(f"Scatter Plot of {x_ax} vs {y_ax}"); st.pyplot(fig)
    
    else:
        st.info("A scatter plot requires at least two numerical columns.")
    
    st.subheader("Box Plot")
    
    if numerical_cols and categorical_cols:
        box_col1, box_col2 = st.columns(2)
        with box_col1: num_box = st.selectbox("Numerical column", numerical_cols, key="num_box")
        with box_col2: cat_box = st.selectbox("Group by categorical column", categorical_cols, key="cat_box")
        
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x=cat_box, y=num_box, ax=ax)
        ax.set_title(f"Box Plot of {num_box} by {cat_box}")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    else:
        st.info("A box plot requires at least one numerical and one categorical column.")
    
    st.markdown("<hr>", unsafe_allow_html=True); nav_col1, nav_col2 = st.columns([1, 1])
    
    with nav_col1:
        
        if st.button("⬅️ Go Back to Feature Identification"):
            st.session_state.app_state = "analysis"
            st.rerun()
    
    with nav_col2:
        
        if st.button("Start Over", key="viz_restart"):
            
            for key in list(st.session_state.keys()): del st.session_state[key]
            st.session_state.app_state = "upload"
            st.rerun()



# === 4. APP LOGIC ===
def main():
    st.title("Automated Data Explorer")
    
    if st.session_state.app_state == "upload":
        upload_and_process_file()
    
    elif st.session_state.app_state == "cleaning":
        data_cleaning_and_transformation()
    
    elif st.session_state.app_state == "analysis":
        feature_identification_page()
    
    elif st.session_state.app_state == "visualization":
        data_visualization_page()

if __name__ == "__main__":
    main()