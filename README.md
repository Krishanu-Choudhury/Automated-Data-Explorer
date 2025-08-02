# 📊 Automated Data Explorer

> A collaborative and intuitive, no-code web application built using Streamlit that empowers users to explore, clean, and understand datasets quickly and efficiently.
Whether you're a data analyst, student, researcher, or business professional, this tool eliminates the need to write code for routine data preparation and analysis tasks. With just a few clicks, you can upload your data, clean missing values, handle outliers, inspect column distributions, detect correlations, and generate ready-to-download profiling reports.

---

## 👥 Collaboration

This project was developed in collaboration between:

- **Krishanu Bikram Choudhury** – *University of Auckland, New Zealand*  
  *Role:* Data analysis and back-end logic, including data cleaning, profiling, and visualization functionalities.
- **Priyanuj Boruah** – *Indian Institute of Technology (IIT) Madras, India*  
  *Role:* UI development and deployment, including building the Streamlit interface and setting up hosting.

We combined our expertise to deliver a seamless, no-code platform for exploratory data analysis.

---

## 🚀 Overview

**Automated Data Explorer** is an interactive, no-code platform designed to help users rapidly:
- Upload datasets from various formats
- Clean and transform data
- Automatically generate profiling reports
- Perform visual data analysis

This tool is perfect for data analysts, students, and researchers looking to understand their data before diving into modeling or reporting.

---

## 🔧 Features

### ✅ Data Upload
- Supports `.csv`, `.tsv`, `.xlsx`, `.json`, and `.parquet` files
- Handles files up to **50MB**
- Auto-trims large CSV/TSV files to avoid crashes

### 🧹 Data Cleaning
- Rename or remove columns
- Handle missing values (drop/fill with mean/median/mode)
- Detect and remove outliers using Z-score
- Change text case
- Remove duplicates

### 📈 Feature Identification (EDA)
- Summary statistics (mean, std, skewness, kurtosis, IQR, etc.)
- Memory usage and data types
- Missing values analysis
- Correlation heatmap
- Deep-dive column analysis
- Auto-generated downloadable `.txt` report

### 📊 Visualization
- Histogram and bar plots
- Scatter and box plots with grouping
- Easy column selection via dropdowns

---

## 🖥️ Technologies Used

- [Streamlit](https://streamlit.io/) – UI framework
- [Pandas](https://pandas.pydata.org/) – Data manipulation
- [NumPy](https://numpy.org/) – Numerical computing
- [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/) – Data visualization
- [SciPy](https://scipy.org/) – Outlier detection

---


---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/PriyanujBoruah/Automated-Data-Explorer.git
```
### 2. Create and Activate a Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```
(Make sure you have a `requirements.txt` file with the necessary packages, or create one using `pip freeze > requirements.txt`)

### 4. Run the App

```bash
streamlit run app.py
```
(Make sure app.py is the name of your main file.)









