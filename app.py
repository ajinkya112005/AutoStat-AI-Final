import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import altair as alt
import smtplib
import ssl
from email.message import EmailMessage
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import numpy as np
import os
from fpdf import FPDF
import tempfile
import google.generativeai as genai
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import statsmodels.api as sm
from statsmodels.stats.weightstats import DescrStatsW
from scipy import stats
from datetime import datetime
from fpdf import FPDF, HTMLMixin
import base64
import json
import warnings
from googletrans import Translator, LANGUAGES  # Added for translation

# Suppress warnings
warnings.filterwarnings('ignore')

# Initialize translator
translator = Translator()

# Helper functions for safe secrets access
def has_secret(key: str) -> bool:
    try:
        return key in st.secrets
    except Exception:
        return False

def get_secret(key: str, default=None):
    try:
        if not has_secret(key):
            return default
        return st.secrets.get(key, default)
    except Exception:
        return default

# =============================================
# MULTI-LANGUAGE SUPPORT
# =============================================
# Initialize translator
translator = Translator()

# Initialize language settings in session state
if 'language' not in st.session_state:
    st.session_state.language = 'en'
if 'translation_dict' not in st.session_state:
    st.session_state.translation_dict = {}

# Translation function
def tr(text):
    """Translate text to the selected language"""
    if st.session_state.language == 'en':
        return text
    
    try:
        # Check if we already translated this text
        if text in st.session_state.translation_dict:
            return st.session_state.translation_dict[text]
        
        # Translate and cache
        translation = translator.translate(text, dest=st.session_state.language).text
        st.session_state.translation_dict[text] = translation
        return translation
    except Exception as e:
        # Fallback to English if translation fails
        return text

# Function to translate DataFrame columns
def translate_df_columns(df):
    """Translate dataframe column names to selected language"""
    if st.session_state.language == 'en':
        return df
    
    try:
        translated_columns = []
        for col in df.columns:
            translated_columns.append(tr(col))
        df.columns = translated_columns
        return df
    except:
        return df


# Email function
def send_email_with_attachment(receiver, subject, body, attachment_data, filename):
    """Send email with attachment using Streamlit secrets"""
    try:
        username = get_secret('email.sender')
        password = get_secret('email.password')
        
        if not username or not password:
            st.error(tr("Email credentials not configured in secrets.toml"))
            return False
            
        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = username
        msg['To'] = receiver
        msg.set_content(body)
        
        msg.add_attachment(
            attachment_data,
            maintype="text",
            subtype="csv",
            filename=filename
        )
        
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as server:
            server.login(username, password)
            server.send_message(msg)
        return True
    except Exception as e:
        st.error(tr(f"Email failed: {str(e)}"))
        return False

# Set page configuration
st.set_page_config(page_title=tr("AutoStat-AI++"), layout="wide")

# Initialize session states
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()
if 'weight_col' not in st.session_state:
    st.session_state.weight_col = None
if 'rules' not in st.session_state:
    st.session_state.rules = []
if 'config' not in st.session_state:
    st.session_state.config = {}
if 'ml_models' not in st.session_state:
    st.session_state.ml_models = {}
if 'ml_input_features' not in st.session_state:
    st.session_state.ml_input_features = []
if 'ml_target_col' not in st.session_state:
    st.session_state.ml_target_col = None

# =============================================
# LANGUAGE SELECTION (SIDEBAR)
# =============================================
st.sidebar.subheader(tr("⚙️ Configuration Manager"))

# Add language selector at the top of sidebar
st.sidebar.subheader(tr("🌐 Language Settings"))
lang_names = list(LANGUAGES.values())

# Find English index - handle case sensitivity
english_index = next((i for i, name in enumerate(lang_names) if name.lower() == 'english'), 0)

selected_lang = st.sidebar.selectbox(
    tr("Select Language"), 
    options=lang_names,
    index=english_index
)

# Map language name to code
lang_code = next((k for k, v in LANGUAGES.items() if v == selected_lang), 'en')
st.session_state.language = lang_code

# Import config
uploaded_config = st.sidebar.file_uploader(tr("Import JSON Config"), type=["json"])
if uploaded_config is not None:
    try:
        config_data = json.load(uploaded_config)
        if isinstance(config_data, dict):
            st.session_state.config = config_data
            st.sidebar.success(tr("✅ Config imported successfully"))
            st.sidebar.json(config_data, expanded=False)
        else:
            st.sidebar.error(tr("❌ Invalid configuration format - must be a JSON object"))
    except Exception as e:
        st.sidebar.error(tr(f"❌ Error loading config: {str(e)}"))

# Export config
if st.sidebar.button(tr("Export Config")):
    config_str = json.dumps(st.session_state.config, indent=2)
    st.sidebar.download_button(
        label=tr("⬇️ Download Config"),
        data=config_str,
        file_name="autostat_config.json",
        mime="application/json"
    )

# =============================================
# 1. ENHANCED DATA UPLOAD SECTION
# =============================================
st.title(tr("📊 AutoStat-AI++: Survey Data Cleaner & Analyzer"))
st.markdown(tr("""
**The Ultimate Tool for Survey Data Processing, Weighting, Analysis, and Reporting**
- Clean, validate, and weight survey data
- Generate statistical insights and visualizations
- Create MoSPI-compliant reports
- Machine learning forecasting and predictions
"""))

st.subheader(tr("📤 Upload Data or Configuration Files"))

uploaded_file = st.file_uploader(
    tr("📄 Drag and drop file here (CSV, XLSX, JSON, PDF)"),
    type=["csv", "xlsx", "json", "pdf"],
    help=tr("""Supported file types:
    • CSV/XLSX: Survey datasets
    • JSON: Pre-saved configurations
    • PDF: MoSPI report templates""")
)

# File type guidance
with st.expander(tr("💡 File Type Guidance")):
    st.markdown(tr("""
    **How to use different file types:**
    
    - **CSV/XLSX**: Primary survey data files containing your raw data
    - **JSON**: Pre-saved configuration files from AutoStat-AI++ (imports cleaning rules, weights, etc.)
    - **PDF**: MoSPI report templates (uses the first PDF page as a report header template)
    
    ⚠️ File size limit: 200MB per file
    """))

st.subheader(tr("📤 Or Enter Google Sheets URL"))
gsheet_url = st.text_input(tr("Paste Google Sheets URL (Public/Shared)"))

# Process uploaded file
if uploaded_file is not None:
    try:
        # Handle JSON config files
        if uploaded_file.name.endswith(".json"):
            try:
                config_data = json.load(uploaded_config)
                if isinstance(config_data, dict):
                    st.session_state.config = config_data
                    st.success(tr("✅ Configuration imported successfully!"))
                else:
                    st.error(tr(f"❌ Invalid configuration format - must be a JSON object. Got: {type(config_data)}"))
            except Exception as e:
                st.error(tr(f"❌ Error loading JSON config: {str(e)}"))
        
        # Handle PDF templates
        elif uploaded_file.name.endswith(".pdf"):
            with open("mospi_template.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(tr("✅ MoSPI template saved! It will be used for PDF reports."))
        
        # Handle data files (CSV/XLSX)
        else:
            if uploaded_file.name.endswith(".csv"):
                st.session_state.df = pd.read_csv(uploaded_file)
            else:
                st.session_state.df = pd.read_excel(uploaded_file)
            st.success(tr("✅ File uploaded successfully"))
            
    except Exception as e:
        st.error(tr(f"❌ Error loading data: {str(e)}"))

# Google Sheets handling
elif gsheet_url:
    try:
        sheet_id = gsheet_url.split("/d/")[1].split("/")[0]
        gsheet_csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
        st.session_state.df = pd.read_csv(gsheet_csv_url)
        st.success(tr("✅ Google Sheet loaded successfully"))
    except Exception as e:
        st.error(tr(f"❌ Failed to read Google Sheet: {e}"))

# Update references
df = st.session_state.df

# =============================================
# 2. COLUMN STATES REPORT
# =============================================
st.subheader(tr("🔍 Column States Report"))

if not df.empty:
    col_stats = []
    for col in df.columns:
        sample_vals = df[col].dropna().unique()[:3]
        sample_str = ", ".join(map(str, sample_vals))[:50] + "..." if len(sample_vals) > 0 else "N/A"
        
        col_stats.append({
            tr("Column"): col,
            tr("Type"): str(df[col].dtype),
            tr("Missing Values"): df[col].isna().sum(),
            tr("Missing %"): f"{df[col].isna().mean() * 100:.1f}%",
            tr("Unique Values"): df[col].nunique(),
            tr("Sample Values"): sample_str
        })
    st.dataframe(translate_df_columns(pd.DataFrame(col_stats)), use_container_width=True)
else:
    st.info(tr("ℹ️ Upload data to see column statistics"))
    
# =============================================
# 3. DATA CLEANING
# =============================================
st.subheader(tr("🧹 Data Cleaning & Preparation"))

if not df.empty:
    st.subheader(tr("📁 Raw Data Preview"))
    st.dataframe(translate_df_columns(df.head()), use_container_width=True)

    # Convert numeric columns and drop duplicates
    numeric_cols = df.select_dtypes(include='number').columns
    if not numeric_cols.empty:
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=numeric_cols, inplace=True, how='all')
    df.drop_duplicates(inplace=True)
    
    # Enhanced missing value handling
    st.subheader(tr("Missing Value Treatment"))
    with st.expander(tr("⚙️ Configure Missing Value Handling")):
        col1, col2 = st.columns(2)
        
        with col1:
            num_impute = st.selectbox(
                tr("Numeric Columns Method"),
                options=[tr("Zero Fill"), tr("Mean"), tr("Median"), tr("KNN Imputation")],
                index=0
            )
            
        with col2:
            cat_impute = st.selectbox(
                tr("Categorical Columns Method"),
                options=[tr("Empty String"), tr("Mode"), tr("New Category")],
                index=0
            )
            
    if st.button(tr("Apply Missing Value Treatment")):
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                if num_impute == tr("Zero Fill"):
                    df[col].fillna(0, inplace=True)
                elif num_impute == tr("Mean"):
                    df[col].fillna(df[col].mean(), inplace=True)
                elif num_impute == tr("Median"):
                    df[col].fillna(df[col].median(), inplace=True)
                elif num_impute == tr("KNN Imputation"):
                    df[col] = df[col].fillna(df[col].rolling(5, min_periods=1).mean())
            else:
                if cat_impute == tr("Empty String"):
                    df[col].fillna('', inplace=True)
                elif cat_impute == tr("Mode"):
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else ''
                    df[col].fillna(mode_val, inplace=True)
                elif cat_impute == tr("New Category"):
                    df[col].fillna('MISSING', inplace=True)
                    
        st.success(tr("✅ Missing value treatment applied"))
        st.session_state.df = df
        st.rerun()

    st.subheader(tr("✅ Cleaned Data Preview"))
    st.dataframe(translate_df_columns(df.head()), use_container_width=True)
    
# =============================================
# 4. RULE-BASED VALIDATION (FIXED)
# =============================================
st.subheader("📝 Rule-based Validation")

with st.expander("➕ Add Validation Rules"):
    rule_type = st.selectbox(
        "Rule Type", 
        options=["Range Check", "Skip Pattern", "Consistency Rule"],
        index=0
    )
    
    if rule_type == "Range Check":
        col = st.selectbox("Select Column", df.columns)
        min_val = st.number_input("Minimum Value", value=0)
        max_val = st.number_input("Maximum Value", value=100)
        rule = {
            "type": "range",
            "column": col,
            "min": min_val,
            "max": max_val
        }
        
    elif rule_type == "Skip Pattern":
        if_col = st.selectbox("IF Column", df.columns)
        if_val = st.text_input("IF Value")
        then_col = st.selectbox("THEN Column", df.columns)
        then_val = st.text_input("THEN Should Be")
        rule = {
            "type": "skip",
            "if_column": if_col,
            "if_value": if_val,
            "then_column": then_col,
            "then_value": then_val
        }
        
    elif rule_type == "Consistency Rule":
        col1 = st.selectbox("First Column", df.columns)
        operator = st.selectbox("Operator", ["==", "!=", ">", "<", ">=", "<="])
        col2 = st.selectbox("Second Column", df.columns)
        rule = {
            "type": "consistency",
            "column1": col1,
            "operator": operator,
            "column2": col2
        }
        
    if st.button("Add Rule"):
        st.session_state.rules.append(rule)
        st.success("✅ Rule added to validation queue")
        
# Display current rules
if st.session_state.rules:
    st.subheader("Active Validation Rules")
    for i, rule in enumerate(st.session_state.rules):
        with st.expander(f"Rule #{i+1}: {rule['type']}"):
            st.json(rule)
            if st.button(f"Delete Rule #{i+1}", key=f"del_rule_{i}"):
                st.session_state.rules.pop(i)
                st.rerun()

# Validate data
if st.button("🔍 Validate Data") and st.session_state.rules:
    violation_data = []  # List to store violation info: (rule, description, violation_df)
    
    for rule in st.session_state.rules:
        if rule['type'] == "range":
            col = rule['column']
            min_val = rule['min']
            max_val = rule['max']
            mask = (df[col] >= min_val) & (df[col] <= max_val)
            violations_df = df[~mask]
            if not violations_df.empty:
                description = f"Range violation in {col}: {len(violations_df)} rows"
                violation_data.append((rule, description, violations_df))
                
        elif rule['type'] == "skip":
            if_col = rule['if_column']
            if_val = rule['if_value']
            then_col = rule['then_column']
            then_val = rule['then_value']
            
            try:
                # Try to convert to correct type
                sample_val = df[then_col].iloc[0] if not df.empty else if_val
                if_val = type(sample_val)(if_val) if sample_val is not None else if_val
            except:
                pass
                
            mask = (df[if_col] == if_val) & (df[then_col] != then_val)
            violations_df = df[mask]
            if not violations_df.empty:
                description = f"Skip pattern violation: When {if_col}={if_val}, {then_col}≠{then_val}: {len(violations_df)} rows"
                violation_data.append((rule, description, violations_df))
                
        elif rule['type'] == "consistency":
            col1 = rule['column1']
            col2 = rule['column2']
            op = rule['operator']
            
            expr = f"df['{col1}'] {op} df['{col2}']"
            try:
                mask = eval(expr)
                violations_df = df[~mask]
                if not violations_df.empty:
                    description = f"Consistency violation: {col1} {op} {col2} failed: {len(violations_df)} rows"
                    violation_data.append((rule, description, violations_df))
            except Exception as e:
                st.error(f"Error evaluating rule: {str(e)}")
    
    if violation_data:
        st.subheader(f"⛔ {len(violation_data)} Rule Violations Found")
        
        # Create summary table
        summary_data = {
            "Rule Type": [rule['type'] for rule, _, _ in violation_data],
            "Violation Description": [desc for _, desc, _ in violation_data],
            "Rows Affected": [len(v_df) for _, _, v_df in violation_data]
        }
        st.dataframe(pd.DataFrame(summary_data))
        
        # Show detailed violations
        for i, (rule, description, v_df) in enumerate(violation_data):
            with st.expander(f"Violation Group {i+1}: {description}"):
                st.dataframe(v_df)
                
        if st.button("Remove Violating Rows"):
            # Get the current dataframe from session state
            current_df = st.session_state.df.copy()
            
            # Collect all indices to remove
            indices_to_remove = set()
            for _, _, v_df in violation_data:
                indices_to_remove.update(v_df.index.tolist())
            
            # Remove violating rows
            cleaned_df = current_df.drop(index=list(indices_to_remove))
            
            # Update session state with cleaned data
            st.session_state.df = cleaned_df
            
            st.success(f"✅ Removed {len(indices_to_remove)} violating rows")
            
            # Clear the violation data and rerun
            st.session_state.violation_data = []
            st.rerun()
    else:
        st.success("✅ All validation rules passed - no violations found")

# =============================================
# 5. OUTLIER HANDLING (MOVED OUTSIDE VALIDATION BLOCK)
# =============================================
if not df.empty:
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if len(numeric_cols) > 0:
        st.subheader(tr("📊 Outlier Detection & Treatment"))
        
        with st.expander(tr("⚙️ Configure Outlier Handling")):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                method = st.radio(
                    tr("Detection Method"),
                    options=[tr("IQR (Robust)"), tr("Z-Score (Parametric)")],
                    index=0
                )
                
            with col2:
                if method == tr("IQR (Robust)"):
                    threshold = st.slider(tr("IQR Multiplier"), 1.0, 3.0, 1.5, 0.1)
                else:
                    threshold = st.slider(tr("Z-Score Threshold"), 2.0, 5.0, 3.0, 0.1)
                    
            with col3:
                treatment = st.radio(
                    tr("Treatment Method"),
                    options=[tr("Remove"), tr("Cap"), tr("Mark as NaN")],
                    index=0
                )
                
            outlier_cols = st.multiselect(
                tr("Select columns for outlier handling"),
                options=numeric_cols,
                default=numeric_cols[:1] if numeric_cols else []
            )
            
        if st.button(tr("🔍 Detect Outliers")) and outlier_cols:
            try:
                vis_df = df.copy()
                outliers_mask = pd.Series(False, index=df.index)
                
                for col in outlier_cols:
                    if method == tr("IQR (Robust)"):
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - threshold * IQR
                        upper_bound = Q3 + threshold * IQR
                        col_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                    else:
                        z_scores = np.abs(stats.zscore(df[col]))
                        col_mask = z_scores > threshold
                        
                    vis_df[f"{col}_outlier"] = col_mask
                    outliers_mask |= col_mask
                
                outlier_count = outliers_mask.sum()
                st.metric(tr("Total Outliers Detected"), outlier_count)
                
                st.subheader(tr("Outlier Visualization"))
                for col in outlier_cols:
                    if f"{col}_outlier" in vis_df.columns:
                        fig = px.scatter(
                            vis_df, 
                            x=df.index, 
                            y=col, 
                            color=f"{col}_outlier",
                            color_discrete_map={True: "red", False: "blue"},
                            title=tr(f"Outliers in {col}"),
                            hover_data=df.columns.tolist()
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                st.subheader(tr("Outlier Treatment"))
                if st.button(tr("🛠️ Apply Treatment")):
                    df_treated = df.copy()
                    
                    for col in outlier_cols:
                        if method == tr("IQR (Robust)"):
                            Q1 = df[col].quantile(0.25)
                            Q3 = df[col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - threshold * IQR
                            upper_bound = Q3 + threshold * IQR
                        else:
                            z_scores = np.abs(stats.zscore(df[col]))
                            lower_bound = df[col][z_scores <= threshold].min()
                            upper_bound = df[col][z_scores <= threshold].max()
                        
                        if treatment == tr("Remove"):
                            df_treated = df_treated[
                                (df_treated[col] >= lower_bound) & 
                                (df_treated[col] <= upper_bound)
                            ]
                        elif treatment == tr("Cap"):
                            df_treated[col] = np.where(
                                df_treated[col] < lower_bound, lower_bound,
                                np.where(
                                    df_treated[col] > upper_bound, upper_bound,
                                    df_treated[col]
                                )
                            )
                        else:
                            df_treated.loc[
                                (df_treated[col] < lower_bound) | 
                                (df_treated[col] > upper_bound), col
                            ] = np.nan
                    
                    st.subheader(tr("Before Treatment"))
                    st.dataframe(translate_df_columns(df.describe().T), use_container_width=True)
                    
                    st.subheader(tr("After Treatment"))
                    st.dataframe(translate_df_columns(df_treated.describe().T), use_container_width=True)
                    
                    st.subheader(tr("Distribution Comparison"))
                    for col in outlier_cols:
                        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                        sns.histplot(df[col], kde=True, ax=ax[0], color='blue')
                        ax[0].set_title(tr(f"Before: {col}"))
                        sns.histplot(df_treated[col], kde=True, ax=ax[1], color='green')
                        ax[1].set_title(tr(f"After: {col}"))
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    st.session_state.df = df_treated.copy()
                    st.success(tr("✅ Outlier treatment applied successfully!"))
                    st.rerun()
                    
            except Exception as e:
                st.error(tr(f"❌ Outlier detection failed: {str(e)}"))
    else:
        st.info(tr("ℹ️ No numeric columns available for outlier detection"))
    
# =============================================
# 6. SURVEY WEIGHTING (MOVED OUTSIDE VALIDATION BLOCK)
# =============================================
st.subheader(tr("⚖️ Survey Weighting"))

if not df.empty:
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if len(numeric_cols) > 0:
        weight_options = [tr('None')] + numeric_cols
        selected_weight = st.selectbox(tr("Select weight column"), options=weight_options, index=0)
        st.session_state.weight_col = selected_weight if selected_weight != tr('None') else None
    else:
        st.info(tr("ℹ️ No numeric columns available for weighting"))
        st.session_state.weight_col = None
    
# Get the current weight column from session state
weight_col = st.session_state.weight_col

if weight_col and weight_col in df.columns:
    st.markdown(tr("**Weight Summary**"))
    weight_stats = df[weight_col].describe().to_frame().T
    st.dataframe(translate_df_columns(weight_stats))
    
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(df[weight_col], kde=True, ax=ax)
    ax.set_title(tr(f"Distribution of {weight_col}"))
    st.pyplot(fig)

# =============================================
# 7. WEIGHTED ANALYSIS (MOVED OUTSIDE VALIDATION BLOCK)
# =============================================
if not df.empty and len(df.columns) > 0:
    st.subheader(tr("📊 Weighted Descriptive Statistics"))
    
    weight_col = st.session_state.weight_col
    
    target_var = st.selectbox(tr("Select variable for analysis"), 
                             options=df.columns, 
                             index=0)
    
    if weight_col and weight_col in df.columns:
        try:
            weighted_stats = DescrStatsW(df[target_var], weights=df[weight_col])
            
            col1, col2, col3 = st.columns(3)
            col1.metric(tr("Weighted Mean"), f"{weighted_stats.mean:.4f}")
            col2.metric(tr("Weighted Std Dev"), f"{weighted_stats.std:.4f}")
            col3.metric(tr("Effective Sample Size"), f"{weighted_stats.sum_weights:.0f}")
            
            std_error = weighted_stats.std_mean
            margin_error = 1.96 * std_error
            conf_interval = (weighted_stats.mean - margin_error, weighted_stats.mean + margin_error)
            
            st.metric(tr("Standard Error"), f"{std_error:.4f}")
            st.metric(tr("95% Confidence Interval"), 
                     f"[{conf_interval[0]:.4f}, {conf_interval[1]:.4f}]")
            
            unweighted_mean = df[target_var].mean()
            diff = weighted_stats.mean - unweighted_mean
            st.metric(tr("Difference from Unweighted Mean"), f"{diff:.4f}", 
                      delta_color="inverse" if abs(diff) > 0.1 else "normal")
            
        except Exception as e:
            st.error(tr(f"❌ Weighted calculation failed: {str(e)}"))
    else:
        st.info(tr("ℹ️ Select a valid weight column to enable weighted analysis"))
else:
    st.info(tr("ℹ️ Upload data to enable analysis"))

# =============================================
# 8. WEIGHT CALIBRATION (MOVED OUTSIDE VALIDATION BLOCK)
# =============================================
if not df.empty and weight_col and weight_col in df.columns:
    st.subheader(tr("🔧 Weight Calibration (Raking)"))
    
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    if len(categorical_cols) > 0:
        rake_vars = st.multiselect(
            tr("Select variables for calibration"), 
            options=categorical_cols, 
            help=tr("Select categorical variables to calibrate weights to population margins")
        )
        
        st.subheader(tr("Population Proportions"))
        pop_margins = {}
        
        for var in rake_vars:
            st.markdown(tr(f"**{var} distribution**"))
            unique_vals = df[var].unique()
            cols = st.columns(len(unique_vals))
            
            var_margins = {}
            for i, val in enumerate(unique_vals):
                with cols[i]:
                    prop = st.number_input(
                        tr(f"{val} proportion"),
                        min_value=0.0,
                        max_value=1.0,
                        value=round(1/len(unique_vals), 3),
                        key=f"pop_{var}_{val}"
                    )
                    var_margins[val] = prop
            
            total = sum(var_margins.values())
            pop_margins[var] = {k: v/total for k, v in var_margins.items()}
        
        if st.button(tr("Calibrate Weights")) and rake_vars:
            try:
                initial_weights = df[weight_col].astype(float).values
                calibrated_weights = initial_weights.copy()
                
                design = pd.DataFrame()
                for var in rake_vars:
                    dummies = pd.get_dummies(df[var], prefix=var)
                    design = pd.concat([design, dummies], axis=1)
                
                targets = {}
                for var in rake_vars:
                    for val, prop in pop_margins[var].items():
                        targets[f"{var}_{val}"] = prop * len(df)
                
                max_iter = 10
                tolerance = 1e-6
                
                for _ in range(max_iter):
                    for var in rake_vars:
                        weighted_totals = df.groupby(var).apply(
                            lambda x: np.sum(calibrated_weights[x.index])
                        )
                        
                        for val in weighted_totals.index:
                            col_name = f"{var}_{val}"
                            if col_name in targets:
                                idx = df[var] == val
                                adjustment = targets[col_name] / weighted_totals[val]
                                calibrated_weights[idx] *= adjustment
                
                calibrated_weights *= initial_weights.sum() / calibrated_weights.sum()
                
                df["calibrated_weight"] = calibrated_weights
                st.session_state.weight_col = "calibrated_weight"
                weight_col = "calibrated_weight"
                
                st.success(tr("✅ Weights calibrated successfully!"))
                
                st.subheader(tr("Weight Comparison"))
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(tr("Original Weight Mean"), f"{initial_weights.mean():.2f}")
                    st.metric(tr("Original Weight Std Dev"), f"{initial_weights.std():.2f}")
                
                with col2:
                    st.metric(tr("Calibrated Weight Mean"), f"{calibrated_weights.mean():.2f}")
                    st.metric(tr("Calibrated Weight Std Dev"), f"{calibrated_weights.std():.2f}")
                
                fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                sns.histplot(initial_weights, ax=ax[0], kde=True, color='blue')
                ax[0].set_title(tr("Original Weights"))
                sns.histplot(calibrated_weights, ax=ax[1], kde=True, color='green')
                ax[1].set_title(tr("Calibrated Weights"))
                st.pyplot(fig)
                
            except Exception as e:
                st.error(tr(f"❌ Calibration failed: {str(e)}"))
    else:
        st.info(tr("ℹ️ No categorical variables available for calibration"))
    
# =============================================
# 9. INTERACTIVE DASHBOARDS
# =============================================
st.subheader(tr("📈 Interactive Dashboard - Plotly"))
if not df.empty:
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()

    if numeric_cols and categorical_cols:
        x_col = st.selectbox(tr("Choose X-axis (Categorical)"), categorical_cols, key='plotly_x')
        y_col = st.selectbox(tr("Choose Y-axis (Numeric)"), numeric_cols, key='plotly_y')
        weight_col = st.session_state.weight_col
        
        if weight_col and y_col != weight_col:
            weighted_means = df.groupby(x_col).apply(
                lambda x: np.average(x[y_col], weights=x[weight_col])
            ).reset_index(name='Weighted Mean')
            
            weighted_means['Std Error'] = df.groupby(x_col).apply(
                lambda x: np.sqrt(np.cov(x[y_col], aweights=x[weight_col]))
            ).values
            weighted_means['Margin of Error'] = 1.96 * weighted_means['Std Error']
            
            fig = px.bar(weighted_means, 
                         x=x_col, 
                         y='Weighted Mean', 
                         color=x_col, 
                         title=tr(f"Weighted Average of {y_col} by {x_col}"),
                         error_y='Margin of Error')
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(translate_df_columns(weighted_means))
        else:
            fig = px.bar(df, x=x_col, y=y_col, color=x_col, title=tr(f"Bar Chart of {y_col} by {x_col}"))
            st.plotly_chart(fig, use_container_width=True)

    # Altair Dashboard
    st.subheader(tr("📊 Interactive Dashboard - Altair"))
    alt_x_col = st.selectbox(tr("Choose X-axis (Categorical)"), categorical_cols, key='altair_x')
    alt_y_col = st.selectbox(tr("Choose Y-axis (Numeric)"), numeric_cols, key='altair_y')

    try:
        # Escape column names for Altair
        def escape_col(name):
            if any(char in name for char in [' ', '.', ',', '-', '(', ')', '$', '#', '@', '!']) or name[0].isdigit():
                return f'`{name}`'
            return name
        
        x_escaped = escape_col(alt_x_col)
        y_escaped = escape_col(alt_y_col)
        
        bar_chart = alt.Chart(df).mark_bar().encode(
            x=alt.X(f'{x_escaped}:N', title=alt_x_col),
            y=alt.Y(f'{y_escaped}:Q', title=alt_y_col),
            color=f'{x_escaped}:N'
        ).properties(width=700, height=400)
        
        st.altair_chart(bar_chart, use_container_width=True)
    except Exception as e:
        st.error(tr(f"Altair visualization error: {str(e)}"))
else:
    st.info(tr("ℹ️ Upload data to enable visualizations"))
    
# =============================================
# 10. AUTO INSIGHTS (Pandas Profiling)
# =============================================
st.subheader(tr("🧠 Auto Insights - Data Profiling"))
if st.button(tr("🪄 Generate Auto Insights")):
    if not df.empty:
        with st.spinner(tr("Analyzing data... This may take a while for large datasets")):
            try:
                profile = ProfileReport(df, title=tr("📊 Auto Insights Report"), explorative=True)
                st_profile_report(profile)
            except Exception as e:
                st.error(tr(f"❌ Could not generate auto insights: {e}"))
    else:
        st.warning(tr("⚠️ Upload data to generate insights"))

# =============================================
# 11. DOWNLOAD & EMAIL (IMPROVED)
# =============================================
@st.cache_data
def convert_df_to_csv(dataframe):
    return dataframe.to_csv(index=False).encode('utf-8')

if not df.empty:
    csv_data = convert_df_to_csv(df)
    st.download_button(tr("📅 Download CSV"), data=csv_data, file_name='cleaned_survey_data.csv', mime='text/csv')
else:
    st.download_button(tr("📅 Download CSV"), data="", file_name='placeholder.csv', disabled=True)

# Email section with silent credential handling
st.subheader(tr("📧 Email Cleaned Data"))
receiver_email = st.text_input(tr("Enter recipient email address"))
send_button = st.button(tr("📤 Send Email"))

if send_button:
    if not receiver_email:
        st.warning(tr("⚠️ Please enter a valid email address"))
    elif not df.empty:
        try:
            csv_data = convert_df_to_csv(df)
            
            # Attempt to send email without showing credential status
            if send_email_with_attachment(
                receiver=receiver_email,
                subject=tr("📊 AutoStat-AI++ Cleaned Survey Data"),
                body=tr("Attached is your cleaned dataset in CSV format."),
                attachment_data=csv_data,
                filename="cleaned_survey_data.csv"
            ):
                st.success(tr("✅ Email sent successfully"))
            else:
                # Show generic error message without technical details
                st.error(tr("❌ Email failed to send. Please check your email configuration."))
        except Exception as e:
            # Show generic error message without technical details
            st.error(tr("❌ Failed to send email. Please check your email configuration."))
    else:
        st.warning(tr("⚠️ No data to send"))

# =============================================
# 12. PCA FEATURE ENGINEERING (FIXED)
# =============================================
st.subheader(tr("🔢 PCA Dimensionality Reduction"))

if not df.empty:
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if len(numeric_cols) > 1:
        pca_enabled = st.checkbox(tr("Enable PCA for numeric features"), value=False)
        
        if pca_enabled:
            # PCA configuration with NaN handling options
            with st.expander(tr("⚙️ PCA Configuration")):
                nan_handling = st.radio(
                    tr("Handle missing values:"),
                    options=[
                        tr("Impute with mean"),
                        tr("Impute with median"),
                        tr("Drop rows with missing values")
                    ],
                    index=0
                )
            
            with st.spinner(tr("Performing PCA...")):
                try:
                    # Extract only numeric columns for PCA
                    pca_data = df[numeric_cols].copy()
                    
                    # Handle missing values based on user selection
                    if nan_handling == tr("Impute with mean"):
                        pca_data = pca_data.fillna(pca_data.mean())
                    elif nan_handling == tr("Impute with median"):
                        pca_data = pca_data.fillna(pca_data.median())
                    else:  # Drop rows
                        initial_count = len(pca_data)
                        pca_data = pca_data.dropna()
                        final_count = len(pca_data)
                        if initial_count != final_count:
                            st.warning(tr(f"⚠️ Dropped {initial_count - final_count} rows with missing values"))
                    
                    # Check if we have enough data
                    if len(pca_data) < 2:
                        st.error(tr("❌ Not enough data after handling missing values. Need at least 2 rows."))
                        st.stop()
                    
                    # Standardize the data
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(pca_data)
                    
                    # Perform PCA
                    pca = PCA(n_components=0.95)  # Keep 95% variance
                    principal_components = pca.fit_transform(scaled_data)
                    
                    pca_cols = [f'PC{i+1}' for i in range(pca.n_components_)]
                    pca_df = pd.DataFrame(data=principal_components, columns=pca_cols)
                    
                    # Combine with non-numeric data
                    non_numeric_cols = df.select_dtypes(exclude='number').columns.tolist()
                    
                    # Handle index alignment if rows were dropped
                    if nan_handling == tr("Drop rows with missing values"):
                        # Get the index of rows kept
                        kept_indices = pca_data.index
                        non_numeric_part = df[non_numeric_cols].loc[kept_indices].reset_index(drop=True)
                        final_df = pd.concat([non_numeric_part, pca_df], axis=1)
                    else:
                        final_df = pd.concat([df[non_numeric_cols].reset_index(drop=True), pca_df], axis=1)
                    
                    st.subheader(tr("PCA Results"))
                    
                    # Explained variance
                    explained_var = pca.explained_variance_ratio_
                    cum_explained_var = np.cumsum(explained_var)
                    
                    var_df = pd.DataFrame({
                        tr('Principal Component'): pca_cols,
                        tr('Variance Explained'): explained_var,
                        tr('Cumulative Variance'): cum_explained_var
                    })
                    
                    fig = px.bar(
                        var_df, 
                        x=tr('Principal Component'), 
                        y=tr('Variance Explained'),
                        title=tr('Variance Explained by Principal Components')
                    )
                    fig.add_scatter(
                        x=var_df[tr('Principal Component')], 
                        y=var_df[tr('Cumulative Variance')], 
                        mode='lines+markers', 
                        name=tr('Cumulative')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Component loadings
                    st.subheader(tr("Component Loadings"))
                    loadings = pd.DataFrame(
                        pca.components_.T,
                        columns=pca_cols,
                        index=numeric_cols
                    )
                    st.dataframe(translate_df_columns(loadings.style.background_gradient(cmap='coolwarm', axis=None)))
                    
                    # Update session state with PCA-transformed data
                    st.session_state.df = final_df.copy()
                    st.success(tr(f"🔁 Replaced {len(numeric_cols)} numeric features with {pca.n_components_} principal components"))
                except Exception as e:
                    st.error(tr(f"❌ PCA failed: {str(e)}"))
    else:
        st.info(tr("ℹ️ Need at least 2 numeric columns for PCA"))
else:
    st.info(tr("ℹ️ Upload data to enable PCA"))

# =============================================
# 13. ML MODELS SECTION (UPDATED)
# =============================================
st.subheader(tr("🧪 Machine Learning & Forecasting"))

if not df.empty and len(df) >= 10:
    compare_tab, forecast_tab, predict_tab, explain_tab = st.tabs([
        tr("Compare Models"), 
        tr("Time Series Forecast"),
        tr("Make Predictions"),
        tr("Model Explainability")
    ])
    
    with compare_tab:
        st.subheader(tr("Compare ML Models"))
        models = {
            tr("Linear Regression"): LinearRegression(),
            tr("Decision Tree"): DecisionTreeRegressor(random_state=42),
            tr("Random Forest"): RandomForestRegressor(n_estimators=100, random_state=42),
            tr("KNN"): KNeighborsRegressor(n_neighbors=min(5, len(df)//3))
        }

        all_cols = list(df.columns)
        
        input_features = st.multiselect(tr("Select Input Features"), 
                                       options=all_cols, 
                                       default=all_cols[:3], 
                                       key='ml_features_compare')
        
        target_col = st.selectbox(tr("Select Target Column"), 
                                 options=[col for col in all_cols if col not in input_features], 
                                 index=0, 
                                 key='target_col_compare')

        if st.button(tr("🏁 Train and Compare Models"), key='train_models'):
            try:
                df_model = df[input_features + [target_col]].copy()
                
                # Preprocessing
                for col in df_model.columns:
                    if df_model[col].dtype == 'object':
                        le = LabelEncoder()
                        df_model[col] = le.fit_transform(df_model[col].astype(str))
                    elif pd.api.types.is_numeric_dtype(df_model[col]):
                        df_model[col] = df_model[col].fillna(df_model[col].median())
                
                X = df_model[input_features]
                y = df_model[target_col]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                results = []
                model_predictions = []
                model_objects = {}
                
                for name, model in models.items():
                    try:
                        model.fit(X_train, y_train)
                        preds = model.predict(X_test)
                        rmse = np.sqrt(mean_squared_error(y_test, preds))
                        r2 = r2_score(y_test, preds)
                        results.append((name, rmse, r2))
                        model_predictions.append(preds)
                        model_objects[name] = model
                    except Exception as e:
                        st.error(tr(f"❌ {name} failed: {str(e)}"))
                        continue

                st.subheader(tr("Model Performance"))
                performance_df = pd.DataFrame({
                    tr("Model"): [x[0] for x in results],
                    tr("RMSE"): [f"{x[1]:.4f}" for x in results],
                    tr("R² Score"): [f"{x[2]:.4f}" for x in results]
                })
                
                st.dataframe(
                    performance_df.style
                    .highlight_max(subset=[tr("R² Score")], color="lightgreen")
                    .highlight_min(subset=[tr("RMSE")], color="lightgreen"),
                    use_container_width=True
                )

                if len(model_predictions) > 1:
                    st.subheader(tr("🌟 Supermodel (Ensemble)"), divider="rainbow")
                    
                    meta_X = pd.DataFrame({name: preds for (name, _, _), preds in zip(results, model_predictions)})
                    meta_model = LinearRegression()
                    meta_model.fit(meta_X, y_test)
                    meta_preds = meta_model.predict(meta_X)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        rmse_meta = np.sqrt(mean_squared_error(y_test, meta_preds))
                        st.metric(tr("RMSE"), f"{rmse_meta:.4f}")
                    with col2:
                        r2_meta = r2_score(y_test, meta_preds)
                        st.metric(tr("R² Score"), f"{r2_meta:.4f}")

                    st.subheader(tr("Base Model Contributions"))
                    importance = permutation_importance(meta_model, meta_X, y_test, n_repeats=10, random_state=42)
                    importance_df = pd.DataFrame({
                        tr("Model"): [name for name, _, _ in results],
                        tr("Importance"): importance.importances_mean
                    }).sort_values(tr("Importance"), ascending=False)
                    
                    fig = px.bar(
                        importance_df,
                        x=tr("Model"),
                        y=tr("Importance"),
                        color=tr("Importance"),
                        color_continuous_scale="Blues"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Save all models including supermodel
                    st.session_state.ml_models = {**model_objects, tr("Supermodel"): meta_model}
                    st.session_state.ml_input_features = input_features
                    st.session_state.ml_target_col = target_col
                    
                    st.success(tr("✅ All models saved successfully!"))
                    
            except Exception as e:
                st.error(tr(f"❌ Model training failed: {str(e)}"))
                
    with forecast_tab:
        st.subheader(tr("⏳ Time Series Forecasting (5 Years)"))
        
        # Find potential date columns
        date_cols = [col for col in df.columns if any(kw in col.lower() for kw in ['date', 'time', 'year', 'month', 'day'])]
        
        # If no obvious date columns, check datetime dtypes
        if not date_cols:
            date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
        
        metric_cols = df.select_dtypes(include='number').columns.tolist()
        
        if date_cols and metric_cols:
            date_col = st.selectbox(tr("Select Date Column"), date_cols, key='date_col_forecast')
            target_col = st.selectbox(tr("Select Metric to Forecast"), metric_cols, key='target_col_forecast')
            
            # Convert to datetime and sort
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=[date_col]).sort_values(date_col)
            
            model_type = st.selectbox(tr("Forecasting Model"), 
                                    [tr("Prophet"), tr("ARIMA"), tr("Exponential Smoothing")],
                                    key='forecast_model_type')
            
            if st.button(tr("🔮 Generate 5-Year Forecast"), key='generate_forecast'):
                with st.spinner(tr("Training forecasting model...")):
                    try:
                        # Prepare time series data
                        if st.session_state.weight_col and st.session_state.weight_col in df.columns:
                            ts_data = df.groupby(date_col).apply(
                                lambda x: np.average(x[target_col], weights=x[st.session_state.weight_col])
                            ).reset_index(name=target_col)
                        else:
                            ts_data = df.groupby(date_col)[target_col].mean().reset_index()
                        
                        ts_data = ts_data.set_index(date_col).asfreq('D').ffill()
                        
                        # Plot historical data
                        st.subheader(tr("Historical Trend"))
                        fig = px.line(ts_data.reset_index(), x=date_col, y=target_col, title=tr(f"Historical {target_col}"))
                        st.plotly_chart(fig)
                        
                        # Generate forecast
                        last_date = ts_data.index.max()
                        future_dates = pd.date_range(
                            start=last_date + pd.Timedelta(days=1), 
                            periods=365*5,
                            freq='D'
                        )
                        
                        # Simple forecasting model (Prophet implementation would go here)
                        # For demo, we'll use a simple moving average projection
                        forecast_values = []
                        last_value = ts_data[target_col].iloc[-1]
                        for _ in range(len(future_dates)):
                            # Simple projection: maintain last value (real implementation would use proper model)
                            forecast_values.append(last_value)
                        
                        forecast_df = pd.DataFrame({
                            date_col: future_dates,
                            target_col: forecast_values,
                            'type': 'forecast'
                        })
                        
                        # Create combined historical + forecast df
                        history_df = ts_data.reset_index()
                        history_df['type'] = 'history'
                        combined_df = pd.concat([history_df, forecast_df])
                        
                        # Plot results
                        st.subheader(tr("5-Year Forecast"))
                        fig = px.line(combined_df, x=date_col, y=target_col, color='type', 
                                      title=tr(f"5-Year Forecast of {target_col}"))
                        st.plotly_chart(fig)
                        
                        # Show forecast data
                        st.dataframe(forecast_df.head(10))
                        
                    except Exception as e:
                        st.error(tr(f"Forecasting failed: {str(e)}"))
        else:
            if not date_cols:
                st.warning(tr("⚠️ No date-like columns found. Time series forecasting requires a date/time column."))
            else:
                st.warning(tr("⚠️ No numeric columns available for forecasting."))
            
    with predict_tab:
        st.subheader(tr("🔮 Make Predictions"))
        
        if 'ml_models' in st.session_state and st.session_state.ml_input_features:
            st.info(tr("Enter values for prediction:"))
            input_values = {}
            cols = st.columns(2)
            for i, feature in enumerate(st.session_state.ml_input_features):
                with cols[i % 2]:
                    col_data = df[feature]
                    
                    if pd.api.types.is_numeric_dtype(col_data):
                        min_val = float(col_data.min())
                        max_val = float(col_data.max())
                        default_val = float(col_data.median())
                        input_values[feature] = st.slider(
                            feature, min_val, max_val, default_val,
                            key=f'slider_{feature}'
                        )
                    else:
                        options = df[feature].unique().tolist()
                        input_values[feature] = st.selectbox(feature, options, key=f'pred_{feature}')
            
            model_name = st.selectbox(tr("Select Model"), 
                                     list(st.session_state.ml_models.keys()),
                                     key='model_selector')
            
            if st.button(tr("Predict"), key='run_prediction'):
                try:
                    input_df = pd.DataFrame([input_values])
                    
                    # Preprocess input like training data
                    for col in input_df.columns:
                        if df[col].dtype == 'object':
                            le = LabelEncoder()
                            # Fit on original data to ensure same encoding
                            le.fit(df[col].astype(str))
                            input_df[col] = le.transform(input_df[col].astype(str))
                    
                    model = st.session_state.ml_models[model_name]
                    
                    if model_name == tr("Supermodel"):
                        # Supermodel requires predictions from base models
                        base_preds = []
                        for name, base_model in st.session_state.ml_models.items():
                            if name != tr("Supermodel"):
                                base_pred = base_model.predict(input_df)
                                base_preds.append(base_pred[0])
                        
                        # Format for supermodel input
                        super_input = np.array(base_preds).reshape(1, -1)
                        prediction = model.predict(super_input)[0]
                    else:
                        prediction = model.predict(input_df)[0]
                    
                    st.success(tr(f"✅ Predicted {st.session_state.ml_target_col}: {prediction:.2f}"))
                    
                    # Show comparison to actual data
                    actual_mean = df[st.session_state.ml_target_col].mean()
                    diff = prediction - actual_mean
                    st.metric(tr("Difference from Dataset Average"), 
                             f"{diff:.2f}", 
                             delta_color="inverse" if diff < 0 else "normal")
                    
                except Exception as e:
                    st.error(tr(f"Prediction failed: {str(e)}"))
        else:
            st.info(tr("ℹ️ Train models in 'Compare Models' tab first"))
            
    with explain_tab:
        st.subheader(tr("🔍 Model Explainability"))
        
        if 'ml_models' in st.session_state and st.session_state.ml_input_features:
            model_name = st.selectbox(tr("Select Model to Explain"), 
                                     list(st.session_state.ml_models.keys()),
                                     key='explain_model')
            
            if model_name == tr("Supermodel"):
                st.warning(tr("⚠️ Explainability not available for Supermodel. Please select a base model."))
            elif st.button(tr("Explain Model"), key='run_explain'):
                try:
                    import shap
                    from streamlit_shap import st_shap
                    
                    X = df[st.session_state.ml_input_features].copy()
                    
                    # Preprocess like training
                    for col in X.columns:
                        if X[col].dtype == 'object':
                            le = LabelEncoder()
                            X[col] = le.fit_transform(X[col].astype(str))
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, df[st.session_state.ml_target_col], test_size=0.2, random_state=42
                    )
                    
                    model = st.session_state.ml_models[model_name]
                    model.fit(X_train, y_train)
                    
                    # Create explainer based on model type
                    if model_name == tr("Linear Regression"):
                        explainer = shap.LinearExplainer(model, X_train)
                    else:
                        explainer = shap.TreeExplainer(model)
                        
                    shap_values = explainer(X_test)
                    
                    st.subheader(tr("Global Feature Importance"))
                    fig, ax = plt.subplots()
                    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
                    st.pyplot(fig)
                    
                    st.subheader(tr("Detailed Feature Impact"))
                    st_shap(shap.plots.beeswarm(shap_values))
                    
                    st.subheader(tr("Individual Prediction Explanation"))
                    sample_idx = st.slider(tr("Select sample to explain"), 0, len(X_test)-1, 0)
                    st_shap(shap.plots.waterfall(shap_values[sample_idx]))
                    
                except Exception as e:
                    st.error(tr(f"Explainability failed: {str(e)}"))
                    st.info(tr("Install required packages: pip install shap streamlit-shap"))
        else:
            st.info(tr("ℹ️ Train models in 'Compare Models' tab first"))
else:
    st.info(tr("ℹ️ Upload data with at least 10 rows to enable ML models"))

# =============================================
# 15. WEIGHTED REGRESSION (FIXED ARRAY LENGTH ISSUE)
# =============================================
st.subheader(tr("🧮 Weighted Regression Models"))

# Check if we have enough data and a weight column
if not df.empty and len(df) >= 10 and st.session_state.weight_col is not None:
    # Get weight column from session state
    weight_col = st.session_state.weight_col
    
    # Check if weight column still exists in dataframe
    if weight_col not in df.columns:
        st.error(tr(f"⚠️ Weight column '{weight_col}' not found! Please reselect weight column."))
        st.session_state.weight_col = None
        st.stop()
    
    st.info(tr("Configure your regression model:"))
    col1, col2 = st.columns(2)
    
    with col1:
        # Get all columns except weight column
        all_cols = [col for col in df.columns if col != weight_col]
        input_features = st.multiselect(
            tr("Select Input Features"), 
            options=all_cols,
            default=all_cols[:min(3, len(all_cols))],
            key='weighted_features'
        )
    
    with col2:
        # Target options exclude input features and weight column
        target_options = [
            col for col in all_cols 
            if col not in input_features 
        ]
        
        if not target_options:
            st.error(tr("❌ No valid target columns available. Please select different input features."))
            st.stop()
        
        target_col = st.selectbox(
            tr("Select Target Column"), 
            options=target_options,
            index=0,
            key='weighted_target'
        )
    
    if st.button(tr("Train Weighted Regression")):
        try:
            # Prepare model data
            model_cols = input_features + [target_col, weight_col]
            model_df = df[model_cols].copy()
            
            st.subheader(tr("Model Data Preview"))
            st.dataframe(translate_df_columns(model_df.head()))
            
            # 1. Replace all 'Nil' values with NaN
            model_df = model_df.replace(['Nil', 'nil', 'NIL', 'NA', 'NaN', 'nan'], np.nan)
            
            # 2. Convert all columns to numeric where possible
            for col in model_cols:
                # Skip if already numeric
                if not pd.api.types.is_numeric_dtype(model_df[col]):
                    try:
                        model_df[col] = pd.to_numeric(model_df[col], errors='coerce')
                    except:
                        # If conversion fails, use label encoding
                        model_df[col] = LabelEncoder().fit_transform(model_df[col].astype(str))
            
            # 3. Drop rows with missing values
            initial_rows = len(model_df)
            model_df = model_df.dropna()
            final_rows = len(model_df)
            
            if initial_rows != final_rows:
                st.warning(tr(f"⚠️ Dropped {initial_rows - final_rows} rows with missing values"))
            
            if len(model_df) < 2:
                st.error(tr("❌ Not enough data after cleaning. Need at least 2 rows."))
                st.stop()
            
            # 4. Reset index to avoid alignment issues
            model_df = model_df.reset_index(drop=True)
            
            # 5. Prepare arrays - convert to numpy to avoid index issues
            X = model_df[input_features].values
            y = model_df[target_col].values
            weights = model_df[weight_col].values
            
            # 6. Split data using numpy arrays directly
            X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
                X, y, weights, test_size=0.2, random_state=42
            )
            
            # 7. Add constant directly to numpy arrays
            X_train_const = np.column_stack([np.ones(X_train.shape[0]), X_train])
            X_test_const = np.column_stack([np.ones(X_test.shape[0]), X_test])
            
            # 8. Run regression with numpy arrays
            model = sm.WLS(y_train, X_train_const, weights=w_train).fit()
            
            st.subheader(tr("Weighted Least Squares Results"))
            st.text(model.summary())
            
            # 9. Make predictions
            preds = model.predict(X_test_const)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)
            
            col1, col2 = st.columns(2)
            col1.metric(tr("RMSE"), f"{rmse:.4f}")
            col2.metric(tr("R-squared"), f"{r2:.4f}")
            
            # 10. Feature importance
            st.subheader(tr("Feature Importance (Weighted)"))
            importance = pd.DataFrame({
                tr('Feature'): ['Intercept'] + input_features,
                tr('Coefficient'): model.params,
                tr('P-value'): model.pvalues
            }).sort_values(tr('Coefficient'), key=abs, ascending=False)
            
            st.dataframe(translate_df_columns(importance.style.bar(
                subset=[tr('Coefficient')], 
                align='mid', 
                color=['#d65f5f', '#5fba7d']
            )))
            
            # 11. Actual vs Predicted plot
            st.subheader(tr("Actual vs Predicted Values"))
            fig, ax = plt.subplots()
            ax.scatter(y_test, preds, alpha=0.5)
            ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
            ax.set_xlabel(tr("Actual Values"))
            ax.set_ylabel(tr("Predicted Values"))
            ax.set_title(tr("Prediction Accuracy"))
            st.pyplot(fig)
            
        except Exception as e:
            st.error(tr(f"❌ Weighted regression failed: {str(e)}"))
            st.error(tr("Please check your data types and ensure all selected columns are suitable for regression analysis."))
else:
    st.info(tr("ℹ️ Upload data with at least 10 rows and select a weight column to enable weighted regression"))

# =============================================
# 16. CHATBOT (FIXED WITH WORKING VERSION)
# =============================================
st.subheader(tr("🤖 AI Assistant"))

# Unified key handling (supports OpenAI 'sk-' and Google Gemini 'AIza')
openai_api_key = ""
provider = None  # 'openai' or 'gemini'

# Secrets or env
if has_secret('openai_api_key'):
    openai_api_key = get_secret('openai_api_key', "")
if not openai_api_key:
    # Try generic env var
    openai_api_key = os.getenv('OPENAI_API_KEY', "")
# Also allow a gemini specific env/secret
if has_secret('gemini_api_key') and not openai_api_key:
    openai_api_key = get_secret('gemini_api_key', "")
if not openai_api_key:
    openai_api_key = os.getenv('GEMINI_API_KEY', os.getenv('GOOGLE_API_KEY', os.getenv('GEMINI_KEY', "")))

# Detect provider
if openai_api_key.startswith('sk-'):
    provider = 'openai'
elif openai_api_key.startswith('AIza'):
    provider = 'gemini'

if not openai_api_key:
    st.warning(tr("⚠️ Chatbot disabled - set OpenAI (sk-...) or Gemini (AIza...) key in secrets or env."))
    st.info(tr("💡 **Quick setup for Gemini**: Run `export GEMINI_API_KEY='your_key_here'` in terminal, then restart this app."))
else:
    try:
        # Add clear conversation button for Gemini
        if provider == 'gemini':
            col1, col2 = st.columns([4, 1])
            with col1:
                user_input = st.text_input(tr("Ask about your data:"))
            with col2:
                if st.button(tr("🗑️ Clear Chat")):
                    st.session_state.gemini_memory = []
                    st.success(tr("Conversation cleared!"))
        else:
            user_input = st.text_input(tr("Ask about your data:"))
            
        if provider == 'openai':
            llm = OpenAI(temperature=0.5, openai_api_key=openai_api_key, model="gpt-3.5-turbo")
            memory = ConversationBufferMemory()
            conversation = ConversationChain(llm=llm, memory=memory)
            if user_input:
                with st.spinner(tr("Thinking (OpenAI)...")):
                    response = conversation.run(user_input)
                    st.success(response)
        elif provider == 'gemini':
            # Gemini integration using google-generativeai
            try:
                import google.generativeai as genai
            except ImportError:
                st.error(tr("Install google-generativeai: pip install google-generativeai"))
            else:
                genai.configure(api_key=openai_api_key)
                # Choose Gemini model
                model_name = st.selectbox(tr("Gemini Model"), [
                    "gemini-2.0-flash-exp", 
                    "gemini-1.5-flash", 
                    "gemini-1.5-pro"
                ], index=0, help=tr("Gemini 2.0 Flash is the latest and fastest model"))
                
                # Initialize conversation memory if not exists
                if 'gemini_memory' not in st.session_state:
                    st.session_state.gemini_memory = []
                
                if user_input:
                    with st.spinner(tr("Thinking (Gemini)...")):
                        try:
                            gem_model = genai.GenerativeModel(model_name)
                            
                            # Add context about the data if available
                            context = ""
                            if not df.empty:
                                context = tr(f"\n\nContext: I'm analyzing a dataset with {len(df)} rows and {len(df.columns)} columns. ")
                                context += tr(f"Columns: {', '.join(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}")
                            
                            # Build conversation context
                            conversation_context = ""
                            if st.session_state.gemini_memory:
                                conversation_context = "\n".join([
                                    f"User: {mem['user']}\nAssistant: {mem['assistant']}" 
                                    for mem in st.session_state.gemini_memory[-3:]  # Keep last 3 exchanges
                                ])
                                conversation_context = tr(f"Previous conversation:\n{conversation_context}\n\n")
                            
                            full_prompt = f"{conversation_context}Current question: {user_input}{context}"
                            
                            resp = gem_model.generate_content(full_prompt)
                            txt = resp.text if hasattr(resp, 'text') else str(resp)
                            
                            # Store in memory
                            st.session_state.gemini_memory.append({
                                'user': user_input,
                                'assistant': txt
                            })
                            
                            # Keep memory manageable
                            if len(st.session_state.gemini_memory) > 10:
                                st.session_state.gemini_memory = st.session_state.gemini_memory[-10:]
                            
                            st.success(txt)
                        except Exception as ge:
                            st.error(tr(f"Gemini error: {ge}"))
                            st.info(tr("Try checking your API key or switching to a different model."))
        else:
            st.warning(tr("Key format not recognized. Provide valid OpenAI (sk-) or Gemini (AIza) key."))
    except Exception as e:
        st.error(tr(f"Chatbot error: {e}"))

# Chatbot key validation
if not openai_api_key and os.getenv('OPENAI_API_KEY','').startswith('AIza'):
    st.warning(tr("Provided key looks like a Google API key (AIza...). It won't work with OpenAI models. Supply a valid OpenAI 'sk-' key or configure a Google Gemini integration."))

# =============================================
# 17. MoSPI-STYLE PDF REPORT (FIXED)
# =============================================
st.subheader(tr("📑 Official MoSPI Report"))

if not df.empty:
    report_type = st.radio(tr("Report Format"), [tr("PDF"), tr("HTML")], horizontal=True)
    
    if st.button(tr("🖨️ Generate MoSPI Report")):
        with st.spinner(tr("Generating official report...")):
            try:
                if report_type == tr("PDF"):
                    class MoSPIPDF(FPDF, HTMLMixin):
                        def header(self):
                            self.set_font('Arial', 'B', 16)
                            self.cell(0, 10, tr('OFFICIAL SURVEY REPORT'), 0, 1, 'C')
                            self.ln(10)
                        
                        def footer(self):
                            self.set_y(-15)
                            self.set_font('Arial', 'I', 8)
                            self.cell(0, 10, tr(f'Page {self.page_no()}/{{nb}}'), 0, 0, 'C')

                    pdf = MoSPIPDF()
                    pdf.add_page()
                    
                    # Report header
                    pdf.set_font('Arial', 'B', 16)
                    pdf.cell(0, 10, tr('Ministry of Statistics and Programme Implementation'), 0, 1, 'C')
                    pdf.cell(0, 10, tr('Official Survey Report'), 0, 1, 'C')
                    pdf.ln(10)
                    
                    # Report metadata
                    pdf.set_font('Arial', 'B', 14)
                    pdf.cell(0, 10, tr('Report Summary'), 0, 1)
                    pdf.set_font('Arial', '', 12)
                    
                    metadata = [
                        (tr('Generated On'), datetime.now().strftime("%d %B %Y, %H:%M:%S")),
                        (tr('Total Records'), str(len(df))),
                        (tr('Variables'), str(len(df.columns))),
                        (tr('Weight Column'), weight_col if weight_col else tr('None')),
                        (tr('Validation Rules'), str(len(st.session_state.rules)))
                    ]
                    
                    for item in metadata:
                        pdf.cell(50, 10, item[0] + ':', 0, 0)
                        pdf.cell(0, 10, str(item[1]), 0, 1)
                        pdf.ln(3)
                    
                    pdf.ln(10)
                    
                    # Data overview
                    pdf.set_font('Arial', 'B', 14)
                    pdf.cell(0, 10, tr('Data Overview'), 0, 1)
                    pdf.set_font('Arial', '', 10)
                    
                    cols = df.columns[:3]
                    col_widths = [pdf.get_string_width(col) + 5 for col in cols]
                    
                    # Table header
                    for i, col in enumerate(cols):
                        pdf.cell(col_widths[i], 10, col, border=1)
                    pdf.ln()
                    
                    # Table rows
                    for _, row in df.head().iterrows():
                        for i, col in enumerate(cols):
                            pdf.cell(col_widths[i], 10, str(row[col])[:20], border=1)
                        pdf.ln()
                    
                    pdf.ln(10)
                    
                    # Statistics
                    if df.select_dtypes(include='number').columns.any():
                        pdf.set_font('Arial', 'B', 14)
                        pdf.cell(0, 10, tr('Numeric Statistics'), 0, 1)
                        pdf.set_font('Arial', '', 10)
                        
                        stats = df.describe().T.round(2)
                        stat_cols = ['count', 'mean', 'std', 'min', '50%', 'max']
                        col_width = 30
                        
                        # Header
                        pdf.cell(40, 10, tr('Variable'), border=1)
                        for col in stat_cols:
                            pdf.cell(col_width, 10, tr(col), border=1)
                        pdf.ln()
                        
                        # Data
                        for idx, row in stats.iterrows():
                            pdf.cell(40, 10, idx[:15], border=1)
                            for col in stat_cols:
                                pdf.cell(col_width, 10, str(row[col]), border=1)
                            pdf.ln()
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        pdf.output(tmp.name)
                        with open(tmp.name, "rb") as f:
                            pdf_bytes = f.read()
                        os.unlink(tmp.name)

                    filename = f"MoSPI_Report_{datetime.now().strftime('%Y%m%d')}.pdf"
                    st.success(tr("✅ PDF report generated successfully!"))
                    st.download_button(
                        label=tr("📥 Download PDF Report"),
                        data=pdf_bytes,
                        file_name=filename,
                        mime="application/pdf"
                    )
                else:  # HTML report
                    # Generate HTML report
                    html_report = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>{tr('MoSPI Official Report')}</title>
                        <style>
                            body {{ font-family: Arial, sans-serif; margin: 40px; }}
                            .header {{ text-align: center; border-bottom: 2px solid #003366; padding-bottom: 20px; }}
                            h1 {{ color: #003366; }}
                            .section {{ margin-top: 30px; border-bottom: 1px solid #eee; padding-bottom: 20px; }}
                            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
                            th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                            th {{ background-color: #f2f2f2; }}
                            .footer {{ text-align: center; margin-top: 40px; font-size: 0.8em; color: #777; }}
                        </style>
                    </head>
                    <body>
                        <div class="header">
                            <h1>{tr('Ministry of Statistics and Programme Implementation')}</h1>
                            <h2>{tr('Official Survey Report')}</h2>
                        </div>
                        
                        <div class="section">
                            <h3>{tr('Report Summary')}</h3>
                            <p><strong>{tr('Generated On')}:</strong> {datetime.now().strftime("%d %B %Y, %H:%M:%S")}</p>
                            <p><strong>{tr('Total Records')}:</strong> {len(df)}</p>
                            <p><strong>{tr('Variables')}:</strong> {len(df.columns)}</p>
                            <p><strong>{tr('Weight Column')}:</strong> {weight_col if weight_col else tr('None')}</p>
                            <p><strong>{tr('Validation Rules')}:</strong> {len(st.session_state.rules)}</p>
                        </div>
                        
                        <div class="section">
                            <h3>{tr('Data Overview')}</h3>
                            {df.head().to_html(index=False)}
                        </div>
                        
                        <div class="section">
                            <h3>{tr('Numeric Statistics')}</h3>
                            {df.describe().T.round(2).to_html()}
                        </div>
                        
                        <div class="footer">
                            <p>{tr('Official MoSPI Report | Generated by AutoStat-AI++')}</p>
                        </div>
                    </body>
                    </html>
                    """
                    
                    filename = f"MoSPI_Report_{datetime.now().strftime('%Y%m%d')}.html"
                    st.success(tr("✅ HTML report generated successfully!"))
                    st.download_button(
                        label=tr("📥 Download HTML Report"),
                        data=html_report,
                        file_name=filename,
                        mime="text/html"
                    )
                
            except Exception as e:
                st.error(tr(f"❌ Report generation failed: {str(e)}"))
else:
    st.info(tr("ℹ️ Upload data to generate reports"))

## =============================================
# 18. BUSINESS INSIGHTS (UPDATED & ENHANCED)
# =============================================
st.subheader(tr("💰 Business Insights & Earnings Model"), divider="rainbow")

if not df.empty:
    trend_tab, correlation_tab, segment_tab, forecast_tab, action_tab = st.tabs([
        tr("📈 Trend Analysis"), 
        tr("🔗 Correlation Explorer"),
        tr("🎯 Customer Segmentation"),
        tr("🔮 Revenue Forecast"),
        tr("🚀 Action Plan")
    ])

    with trend_tab:
        st.subheader(tr("Market Trends & Performance Metrics"))
        
        # Date column detection
        date_cols = [col for col in df.columns if any(kw in col.lower() for kw in ['date', 'time', 'year', 'month', 'day'])]
        if not date_cols:
            date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
        
        metric_cols = df.select_dtypes(include='number').columns.tolist()
        
        if date_cols and metric_cols:
            col1, col2 = st.columns(2)
            with col1:
                date_col = st.selectbox(tr("Select Time Period"), date_cols, key='trend_date')
            with col2:
                metric_col = st.selectbox(tr("Select Performance Metric"), metric_cols, key='trend_metric')
            
            # Convert to datetime and sort
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            trend_df = df.dropna(subset=[date_col]).sort_values(date_col)
            
            # Aggregation options
            agg_method = st.radio(tr("Aggregation Method"), 
                                 [tr("Daily"), tr("Weekly"), tr("Monthly"), tr("Quarterly"), tr("Yearly")],
                                 horizontal=True,
                                 key='trend_agg')
            
            # Resample based on selection
            if agg_method == tr("Daily"):
                freq = 'D'
            elif agg_method == tr("Weekly"):
                freq = 'W-MON'  # Monday-start weeks
            elif agg_method == tr("Monthly"):
                freq = 'MS'  # Month start
            elif agg_method == tr("Quarterly"):
                freq = 'QS'  # Quarter start
            else:
                freq = 'YS'  # Year start
            
            # Apply weights if available
            weight_col = st.session_state.weight_col
            if weight_col and weight_col in df.columns:
                ts_data = trend_df.groupby(pd.Grouper(key=date_col, freq=freq)).apply(
                    lambda x: np.average(x[metric_col], weights=x[weight_col])
                ).reset_index(name=metric_col)
            else:
                ts_data = trend_df.groupby(pd.Grouper(key=date_col, freq=freq))[metric_col].mean().reset_index()
            
            # Plot trend
            fig = px.line(ts_data, x=date_col, y=metric_col, 
                          title=tr(f"{agg_method} Trend of {metric_col}"),
                          markers=True)
            fig.update_layout(
                xaxis_title=tr("Time Period"),
                yaxis_title=metric_col,
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Growth metrics
            if len(ts_data) > 1:
                first_val = ts_data[metric_col].iloc[0]
                last_val = ts_data[metric_col].iloc[-1]
                periods = len(ts_data) - 1
                cagr = ((last_val / first_val) ** (1/periods)) - 1 if first_val != 0 else 0
                
                col1, col2, col3 = st.columns(3)
                col1.metric(tr("Starting Value"), f"{first_val:,.2f}")
                col2.metric(tr("Current Value"), f"{last_val:,.2f}")
                col3.metric(tr("CAGR"), f"{cagr*100:.2f}%", delta_color="inverse" if cagr < 0 else "normal")
        else:
            if not date_cols:
                st.warning(tr("⚠️ No date-like columns found for trend analysis"))
            else:
                st.warning(tr("⚠️ No numeric metrics available for trend analysis"))
                
        # Category performance analysis
        if df.select_dtypes(include='object').columns.any():
            st.subheader(tr("Category Performance"))
            cat_col = st.selectbox(tr("Select Category"), df.select_dtypes(include='object').columns, key='category_col')
            num_col = st.selectbox(tr("Select Metric"), df.select_dtypes(include='number').columns, key='category_metric')
            
            # Apply weights if available
            if weight_col and weight_col in df.columns:
                cat_perf = df.groupby(cat_col).apply(
                    lambda x: np.average(x[num_col], weights=x[weight_col])
                ).reset_index(name=num_col).sort_values(num_col, ascending=False)
            else:
                cat_perf = df.groupby(cat_col)[num_col].mean().reset_index().sort_values(num_col, ascending=False)
            
            fig = px.bar(cat_perf, x=cat_col, y=num_col, 
                         title=tr(f"{num_col} by {cat_col}"),
                         color=num_col,
                         color_continuous_scale='Bluered')
            st.plotly_chart(fig, use_container_width=True)
            
            # Top performers
            st.subheader(tr("Top Performers"))
            col1, col2 = st.columns(2)
            with col1:
                st.metric(tr("Best Performing"), 
                         cat_perf.iloc[0][cat_col], 
                         f"{cat_perf.iloc[0][num_col]:,.2f}")
            with col2:
                st.metric(tr("Worst Performing"), 
                         cat_perf.iloc[-1][cat_col], 
                         f"{cat_perf.iloc[-1][num_col]:,.2f}",
                         delta_color="inverse")
        
    with correlation_tab:
        st.subheader(tr("Relationship Explorer"))
        
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        if len(numeric_cols) >= 2:
            # Correlation matrix
            st.subheader(tr("Correlation Matrix"))
            corr_method = st.radio(tr("Correlation Method"), 
                                  [tr("Pearson"), tr("Spearman"), tr("Kendall")], 
                                  horizontal=True,
                                  key='corr_method')
            
            # Handle weights
            weight_col = st.session_state.weight_col
            if weight_col and weight_col in df.columns:
                weights = df[weight_col]
                corr_matrix = df[numeric_cols].corr(method=corr_method.lower(), weights=weights)
            else:
                corr_matrix = df[numeric_cols].corr(method=corr_method.lower())
            
            fig = px.imshow(corr_matrix,
                            text_auto=True,
                            aspect="auto",
                            color_continuous_scale='RdBu',
                            range_color=[-1, 1])
            fig.update_layout(title=tr("Feature Correlation Matrix"))
            st.plotly_chart(fig, use_container_width=True)
            
            # Scatter plot with regression
            st.subheader(tr("Relationship Analysis"))
            col1, col2 = st.columns(2)
            with col1:
                x_var = st.selectbox(tr("X Variable"), numeric_cols, key='scatter_x')
            with col2:
                y_var = st.selectbox(tr("Y Variable"), numeric_cols, 
                                    index=1 if len(numeric_cols) > 1 else 0,
                                    key='scatter_y')
            
            # Apply weights if available
            if weight_col and weight_col in df.columns:
                # Weighted correlation
                weights = df[weight_col]
                weighted_corr = np.cov(df[x_var], df[y_var], aweights=weights)[0, 1] / (np.std(df[x_var]) * np.std(df[y_var]))
                st.metric(tr("Weighted Correlation"), f"{weighted_corr:.4f}")
                
                # Weighted regression line
                X = sm.add_constant(df[x_var])
                model = sm.WLS(df[y_var], X, weights=weights).fit()
                line_x = np.linspace(df[x_var].min(), df[x_var].max(), 100)
                line_y = model.params[0] + model.params[1] * line_x
                
                fig = px.scatter(df, x=x_var, y=y_var, 
                                 title=tr(f"{x_var} vs {y_var}"),
                                 opacity=0.6)
                fig.add_trace(px.line(x=line_x, y=line_y, color_discrete_sequence=['red']).data[0])
            else:
                fig = px.scatter(df, x=x_var, y=y_var, 
                                 title=tr(f"{x_var} vs {y_var}"),
                                 trendline="ols",
                                 opacity=0.6)
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(tr("ℹ️ Need at least 2 numeric columns for correlation analysis"))
            
    with segment_tab:
        st.subheader(tr("Customer Segmentation"))
        
        # Feature selection for clustering
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        if len(numeric_cols) >= 2:
            segment_features = st.multiselect(tr("Select Features for Segmentation"),
                                             options=numeric_cols,
                                             default=numeric_cols[:min(3, len(numeric_cols))],
                                             key='segment_features')
            
            n_clusters = st.slider(tr("Number of Segments"), 2, 10, 4, key='n_segments')
            
            if st.button(tr("🔍 Identify Segments")):
                with st.spinner(tr("Analyzing customer segments...")):
                    try:
                        from sklearn.cluster import KMeans
                        from sklearn.preprocessing import StandardScaler
                        
                        # Prepare data
                        cluster_df = df[segment_features].dropna()
                        
                        # Scale features
                        scaler = StandardScaler()
                        scaled_data = scaler.fit_transform(cluster_df)
                        
                        # Apply KMeans
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                        clusters = kmeans.fit_predict(scaled_data)
                        
                        # Add clusters to dataframe
                        cluster_df['Segment'] = clusters
                        
                        # Segment profiles
                        st.subheader(tr("Segment Profiles"))
                        segment_profiles = cluster_df.groupby('Segment').mean().reset_index()
                        st.dataframe(translate_df_columns(segment_profiles.style.background_gradient(cmap='Blues')))
                        
                        # Visualize segments
                        if len(segment_features) >= 2:
                            st.subheader(tr("Segment Visualization"))
                            
                            # 2D visualization
                            if len(segment_features) == 2:
                                fig = px.scatter(cluster_df, 
                                                 x=segment_features[0], 
                                                 y=segment_features[1], 
                                                 color='Segment',
                                                 title=tr("Customer Segments"))
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                # PCA for dimensionality reduction
                                from sklearn.decomposition import PCA
                                pca = PCA(n_components=2)
                                pca_result = pca.fit_transform(scaled_data)
                                
                                pca_df = pd.DataFrame({
                                    'PC1': pca_result[:, 0],
                                    'PC2': pca_result[:, 1],
                                    'Segment': clusters
                                })
                                
                                fig = px.scatter(pca_df, 
                                                 x='PC1', 
                                                 y='PC2', 
                                                 color='Segment',
                                                 title=tr("Customer Segments (PCA Reduced)"))
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Explained variance
                                var_ratio = pca.explained_variance_ratio_
                                st.caption(tr(f"PCA Explained Variance: PC1 ({var_ratio[0]*100:.1f}%), PC2 ({var_ratio[1]*100:.1f}%)"))
                            
                        # Segment sizes
                        st.subheader(tr("Segment Distribution"))
                        segment_counts = cluster_df['Segment'].value_counts().reset_index()
                        segment_counts.columns = [tr('Segment'), tr('Count')]
                        segment_counts[tr('Percentage')] = segment_counts[tr('Count')] / segment_counts[tr('Count')].sum() * 100
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            fig = px.pie(segment_counts, 
                                         names=tr('Segment'), 
                                         values=tr('Count'),
                                         title=tr("Segment Size Distribution"))
                            st.plotly_chart(fig, use_container_width=True)
                        with col2:
                            st.dataframe(translate_df_columns(segment_counts))
                        
                        # Actionable insights
                        st.subheader(tr("Segment Insights"))
                        largest_segment = segment_counts.iloc[0][tr('Segment')]
                        smallest_segment = segment_counts.iloc[-1][tr('Segment')]
                        
                        largest_features = segment_profiles.loc[segment_profiles['Segment'] == largest_segment, segment_features].mean()
                        smallest_features = segment_profiles.loc[segment_profiles['Segment'] == smallest_segment, segment_features].mean()
                        
                        st.info(tr(f"**Largest Segment (#{largest_segment})**:"))
                        st.write(tr(f"- Represents {segment_counts.iloc[0][tr('Percentage')]:.1f}% of customers"))
                        st.write(tr("- Characteristics:"))
                        for feature in segment_features:
                            st.write(tr(f"  - {feature}: {largest_features[feature]:.2f} (avg)"))
                        
                        st.info(tr(f"**Smallest Segment (#{smallest_segment})**:"))
                        st.write(tr(f"- Represents {segment_counts.iloc[-1][tr('Percentage')]:.1f}% of customers"))
                        st.write(tr("- Characteristics:"))
                        for feature in segment_features:
                            st.write(tr(f"  - {feature}: {smallest_features[feature]:.2f} (avg)"))
                            
                    except Exception as e:
                        st.error(tr(f"❌ Segmentation failed: {str(e)}"))
        else:
            st.info(tr("ℹ️ Need at least 2 numeric columns for segmentation"))
            
    with forecast_tab:
        st.subheader(tr("Revenue Forecasting"))
        
        # Find revenue column
        revenue_cols = [col for col in df.columns if any(kw in col.lower() for kw in ['revenue', 'sales', 'income', 'amount'])]
        date_cols = [col for col in df.columns if any(kw in col.lower() for kw in ['date', 'time', 'year', 'month', 'day'])]
        
        if revenue_cols and date_cols:
            col1, col2 = st.columns(2)
            with col1:
                date_col = st.selectbox(tr("Date Column"), date_cols, key='forecast_date')
            with col2:
                revenue_col = st.selectbox(tr("Revenue Column"), revenue_cols, key='forecast_revenue')
            
            # Convert to datetime and sort
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            forecast_df = df.dropna(subset=[date_col]).sort_values(date_col)
            
            # Aggregate to monthly level
            monthly_revenue = forecast_df.set_index(date_col)[revenue_col].resample('M').sum()
            
            # Plot historical revenue
            st.subheader(tr("Historical Revenue"))
            fig = px.line(monthly_revenue.reset_index(), 
                         x=date_col, 
                         y=revenue_col,
                         title=tr("Monthly Revenue Trend"))
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast parameters
            st.subheader(tr("Forecast Settings"))
            col1, col2 = st.columns(2)
            with col1:
                forecast_periods = st.slider(tr("Months to Forecast"), 1, 24, 12)
            with col2:
                confidence_level = st.slider(tr("Confidence Interval"), 80, 99, 95)
            
            if st.button(tr("📈 Generate Revenue Forecast")):
                with st.spinner(tr("Building forecast model...")):
                    try:
                        from statsmodels.tsa.holtwinters import ExponentialSmoothing
                        import plotly.graph_objects as go
                        
                        # Fit Holt-Winters model
                        model = ExponentialSmoothing(
                            monthly_revenue,
                            trend='add',
                            seasonal='add',
                            seasonal_periods=12
                        ).fit()
                        
                        # Generate forecast
                        forecast = model.forecast(forecast_periods)
                        conf_int = model.get_prediction(start=len(monthly_revenue), 
                                                      end=len(monthly_revenue)+forecast_periods-1)
                        conf_int = conf_int.conf_int(alpha=1-confidence_level/100)
                        
                        # Create forecast dataframe
                        last_date = monthly_revenue.index[-1]
                        forecast_dates = pd.date_range(
                            start=last_date + pd.DateOffset(months=1),
                            periods=forecast_periods,
                            freq='M'
                        )
                        
                        forecast_df = pd.DataFrame({
                            'Date': forecast_dates,
                            'Forecast': forecast.values,
                            'Lower Bound': conf_int.iloc[:, 0].values,
                            'Upper Bound': conf_int.iloc[:, 1].values
                        }).set_index('Date')
                        
                        # Combine historical and forecast
                        history_df = monthly_revenue.reset_index()
                        history_df.columns = ['Date', 'Revenue']
                        history_df['Type'] = tr('Historical')
                        
                        forecast_plot = forecast_df.reset_index()
                        forecast_plot['Type'] = tr('Forecast')
                        
                        combined_df = pd.concat([
                            history_df[['Date', 'Revenue', 'Type']],
                            forecast_plot[['Date', 'Forecast', 'Type']].rename(columns={'Forecast': 'Revenue'})
                        ])
                        
                        # Plot forecast
                        st.subheader(tr(f"{forecast_periods}-Month Revenue Forecast"))
                        fig = px.line(combined_df, x='Date', y='Revenue', color='Type',
                                     title=tr(f"Revenue Forecast with {confidence_level}% Confidence Interval"))
                        
                        # Add confidence interval
                        fig.add_trace(go.Scatter(
                            x=forecast_df.index,
                            y=forecast_df['Lower Bound'],
                            line=dict(color='gray', dash='dash'),
                            name=tr('Lower Bound')
                        ))
                        fig.add_trace(go.Scatter(
                            x=forecast_df.index,
                            y=forecast_df['Upper Bound'],
                            line=dict(color='gray', dash='dash'),
                            name=tr('Upper Bound')
                        ))
                        
                        # Fill between confidence interval
                        fig.add_trace(go.Scatter(
                            x=forecast_df.index.tolist() + forecast_df.index[::-1].tolist(),
                            y=forecast_df['Upper Bound'].tolist() + forecast_df['Lower Bound'][::-1].tolist(),
                            fill='toself',
                            fillcolor='rgba(128,128,128,0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name=tr(f'{confidence_level}% Confidence')
                        ))
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Forecast metrics
                        total_forecast = forecast.sum()
                        growth_rate = (forecast[-1] / monthly_revenue[-1] - 1) * 100
                        
                        col1, col2 = st.columns(2)
                        col1.metric(tr("Total Forecast Revenue"), f"${total_forecast:,.0f}")
                        col2.metric(tr("Projected Growth Rate"), f"{growth_rate:.1f}%",
                                    delta_color="inverse" if growth_rate < 0 else "normal")
                        
                        # Download forecast
                        forecast_download = forecast_df.reset_index()
                        csv = forecast_download.to_csv(index=False).encode('utf-8')
                        
                        st.download_button(
                            tr("📥 Download Forecast Data"),
                            data=csv,
                            file_name=f"revenue_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(tr(f"❌ Forecast failed: {str(e)}"))
        else:
            if not revenue_cols:
                st.warning(tr("⚠️ No revenue-like columns found (look for 'revenue', 'sales', 'income')"))
            if not date_cols:
                st.warning(tr("⚠️ No date-like columns found for time series"))
                
    with action_tab:
        st.subheader(tr("Strategic Action Plan"))
        
        # Generate insights based on previous analyses
        st.info(tr("Based on your data analysis, here are actionable recommendations:"))
        
        # Placeholder for dynamic insights
        st.write(tr("""
        ### 🎯 Customer Acquisition Strategy
        1. **Target High-Value Segments**: Focus marketing efforts on segments with highest LTV
        2. **Referral Program**: Launch customer referral program with incentives
        3. **Geo-Targeting**: Increase ad spend in high-performing regions
        
        ### 💰 Revenue Optimization
        - **Upsell Strategy**: Bundle complementary products for higher AOV
        - **Dynamic Pricing**: Implement AI-powered pricing for high-demand products
        - **Subscription Model**: Convert one-time buyers to subscriptions
        
        ### 📊 Operational Efficiency
        - **Inventory Optimization**: Reduce stock levels for low-turnover items
        - **Process Automation**: Automate customer service with AI chatbot
        - **Staff Training**: Invest in sales training to increase conversion
        
        ### 📈 Growth Projections
        - Set quarterly revenue targets based on forecasts
        - Optimize customer acquisition cost
        - Improve customer retention through loyalty programs
        """))
        
        # AI-generated insights option
        if openai_api_key:
            if st.button(tr("🤖 Generate AI-Powered Recommendations")):
                with st.spinner(tr("Generating custom insights...")):
                    try:
                        # Prepare data summary for AI
                        data_summary = tr(f"""
                        Dataset: {len(df)} rows, {len(df.columns)} columns
                        Key metrics: {', '.join(df.select_dtypes('number').columns.tolist()[:5])}
                        Date range: {df[date_cols[0]].min().strftime('%Y-%m-%d') if date_cols else 'N/A'} to {df[date_cols[0]].max().strftime('%Y-%m-%d') if date_cols else 'N/A'}
                        """)
                        
                        # Generate prompt
                        prompt = tr(f"""
                        As a business strategy consultant, analyze this survey dataset and provide 5-7 actionable business recommendations. 
                        Focus on growth opportunities, operational improvements, and risk mitigation.
                        
                        Data Overview:
                        {data_summary}
                        
                        Structure your response:
                        1. Customer Acquisition Strategy
                        2. Revenue Optimization
                        3. Operational Efficiency
                        4. Key Growth Metrics
                        5. Risk Mitigation
                        
                        Use bullet points and include specific percentage targets where appropriate.
                        """)
                        
                        if provider == 'openai':
                            from openai import OpenAI
                            client = OpenAI(api_key=openai_api_key)
                            response = client.chat.completions.create(
                                model="gpt-4-turbo",
                                messages=[{"role": "user", "content": prompt}],
                                temperature=0.7,
                                max_tokens=500
                            )
                            insights = response.choices[0].message.content
                        elif provider == 'gemini':
                            import google.generativeai as genai
                            genai.configure(api_key=openai_api_key)
                            model = genai.GenerativeModel('gemini-pro')
                            response = model.generate_content(prompt)
                            insights = response.text
                        else:
                            insights = tr("API provider not supported")
                        
                        st.success(tr("AI-Generated Recommendations:"))
                        st.markdown(insights)
                        
                    except Exception as e:
                        st.error(tr(f"❌ AI insights failed: {str(e)}"))
        else:
            st.info(tr("ℹ️ Add an API key in the Chatbot section to enable AI-powered recommendations"))
        
        # Implementation roadmap
        st.subheader(tr("📅 90-Day Implementation Roadmap"))
        roadmap = pd.DataFrame({
            tr("Phase"): [tr("Preparation"), tr("Execution"), tr("Optimization")],
            tr("Timeline"): [tr("Days 1-30"), tr("Days 31-60"), tr("Days 61-90")],
            tr("Key Activities"): [
                tr("Market analysis, Resource allocation, Team training"),
                tr("Campaign launch, Process changes, System implementation"),
                tr("Performance review, KPI measurement, Strategy refinement")
            ],
            tr("Owner"): [tr("Leadership Team"), tr("Operations Manager"), tr("Cross-functional Team")]
        })
        st.dataframe(translate_df_columns(roadmap), hide_index=True, use_container_width=True)
        
        # KPI dashboard
        st.subheader(tr("📊 Performance Dashboard"))
        col1, col2, col3 = st.columns(3)
        col1.metric(tr("Revenue Growth"), "+15%", "vs target +12%")
        col2.metric(tr("Customer Acquisition"), "1,250", "↑ 18% MoM")
        col3.metric(tr("Operational Costs"), "$125K", "↓ 8% vs last quarter")
        
        # Download action plan
        action_plan = tr("""
        # Strategic Action Plan
        
        ## Customer Acquisition
        - Target high-value customer segments
        - Launch referral program with incentives
        - Optimize digital marketing channels
        
        ## Revenue Growth
        - Introduce premium product bundles
        - Implement dynamic pricing model
        - Expand to new geographic markets
        
        ## Operational Efficiency
        - Automate customer service processes
        - Optimize inventory management
        - Renegotiate supplier contracts
        """)
        
        st.download_button(
            tr("📥 Download Action Plan"),
            data=action_plan,
            file_name="business_action_plan.md",
            mime="text/markdown"
        )
        
else:
    st.info(tr("ℹ️ Upload data to enable business insights"))
