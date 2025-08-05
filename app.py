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
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from langchain.llms import OpenAI
from langchain_community.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import statsmodels.api as sm  # NEW: For survey weighting
from statsmodels.stats.weightstats import DescrStatsW  # NEW: For weighted stats
from scipy import stats  # NEW: For outlier detection
from datetime import datetime
from fpdf import FPDF, HTMLMixin
import base64

st.set_page_config(page_title="AutoStat-AI++", layout="wide")

# Initialize empty dataframe at start
df = pd.DataFrame()

st.title("📊 AutoStat-AI++: Survey Data Cleaner & Analyzer")

# =============================================
# 1. DATA UPLOAD SECTION (WITH IMPROVED ERROR HANDLING)
# =============================================
admin_view = st.sidebar.checkbox("👮 Admin Dashboard")

uploaded_file = st.file_uploader("📄 Upload your CSV/Excel file", type=["csv", "xlsx"])
st.subheader("📤 Or Enter Google Sheets URL")
gsheet_url = st.text_input("Paste Google Sheets URL (Public/Shared)")

if uploaded_file is not None or gsheet_url:
    try:
        if uploaded_file is not None:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.success("✅ File uploaded successfully")
        elif gsheet_url:
            try:
                sheet_id = gsheet_url.split("/d/")[1].split("/")[0]
                gsheet_csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
                df = pd.read_csv(gsheet_csv_url)
                st.success("✅ Google Sheet loaded successfully")
            except Exception as e:
                st.error(f"❌ Failed to read Google Sheet: {e}")
                st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()

    # Only proceed if df is loaded
    if df.empty:
        st.warning("No data loaded - please check your file")
        st.stop()

    # =============================================
    # 2. COLUMN STATES REPORT
    # =============================================
    with st.expander("🔍 Column States Report"):
        col_stats = []
        for col in df.columns:
            col_stats.append({
                "Column": col,
                "Type": str(df[col].dtype),
                "Missing Values": df[col].isna().sum(),
                "Missing %": f"{df[col].isna().mean() * 100:.1f}%",
                "Unique Values": df[col].nunique(),
                "Sample Values": str(list(df[col].dropna().unique()[:3]))[:50] + "..."
            })
        st.dataframe(pd.DataFrame(col_stats), use_container_width=True)

    # =============================================
    # 3. DATA CLEANING (WITH SAFETY CHECKS)
    # =============================================
    st.subheader("📁 Raw Data Preview")
    st.dataframe(df, use_container_width=True)

    numeric_cols = df.select_dtypes(include='number').columns
    if not numeric_cols.empty:
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=numeric_cols, inplace=True, how='all')
    df.drop_duplicates(inplace=True)

    st.subheader("✅ Cleaned Data Preview")
    st.dataframe(df, use_container_width=True)
    
    # =============================================
    # 3.5 OUTLIER HANDLING SECTION (NEW)
    # =============================================
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if len(numeric_cols) > 0:
        st.subheader("📊 Outlier Detection & Treatment")
        
        with st.expander("⚙️ Configure Outlier Handling"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Method selection
                method = st.radio(
                    "Detection Method",
                    options=["IQR (Robust)", "Z-Score (Parametric)"],
                    index=0
                )
                
            with col2:
                # Threshold selection
                if method == "IQR (Robust)":
                    threshold = st.slider(
                        "IQR Multiplier", 
                        min_value=1.0, 
                        max_value=3.0, 
                        value=1.5, 
                        step=0.1
                    )
                else:
                    threshold = st.slider(
                        "Z-Score Threshold", 
                        min_value=2.0, 
                        max_value=5.0, 
                        value=3.0, 
                        step=0.1
                    )
                    
            with col3:
                # Treatment options
                treatment = st.radio(
                    "Treatment Method",
                    options=["Remove", "Cap", "Mark as NaN"],
                    index=0
                )
                
            # Column selection
            outlier_cols = st.multiselect(
                "Select columns for outlier handling",
                options=numeric_cols,
                default=numeric_cols[:1]
            )
            
        if st.button("🔍 Detect Outliers") and outlier_cols:
            try:
                # Create copy for visualization
                vis_df = df.copy()
                
                # Detect outliers
                outliers_mask = pd.Series(False, index=df.index)
                
                for col in outlier_cols:
                    if method == "IQR (Robust)":
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - threshold * IQR
                        upper_bound = Q3 + threshold * IQR
                        col_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                    else:  # Z-Score
                        z_scores = np.abs(stats.zscore(df[col]))
                        col_mask = z_scores > threshold
                        
                    # Add to visualization dataframe
                    vis_df[f"{col}_outlier"] = col_mask
                    outliers_mask |= col_mask
                
                # Show summary
                outlier_count = outliers_mask.sum()
                st.metric("Total Outliers Detected", outlier_count)
                
                # Visualize outliers
                st.subheader("Outlier Visualization")
                
                for col in outlier_cols:
                    if f"{col}_outlier" in vis_df.columns:
                        fig = px.scatter(
                            vis_df, 
                            x=df.index, 
                            y=col, 
                            color=f"{col}_outlier",
                            color_discrete_map={True: "red", False: "blue"},
                            title=f"Outliers in {col}",
                            hover_data=df.columns.tolist()
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Treatment options
                st.subheader("Outlier Treatment")
                
                if st.button("🛠️ Apply Treatment"):
                    df_treated = df.copy()
                    
                    for col in outlier_cols:
                        if method == "IQR (Robust)":
                            Q1 = df[col].quantile(0.25)
                            Q3 = df[col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - threshold * IQR
                            upper_bound = Q3 + threshold * IQR
                        else:  # Z-Score
                            z_scores = np.abs(stats.zscore(df[col]))
                            lower_bound = df[col][z_scores <= threshold].min()
                            upper_bound = df[col][z_scores <= threshold].max()
                        
                        if treatment == "Remove":
                            df_treated = df_treated[
                                (df_treated[col] >= lower_bound) & 
                                (df_treated[col] <= upper_bound)
                            ]
                        elif treatment == "Cap":
                            df_treated[col] = np.where(
                                df_treated[col] < lower_bound, lower_bound,
                                np.where(
                                    df_treated[col] > upper_bound, upper_bound,
                                    df_treated[col]
                                )
                            )
                        else:  # Mark as NaN
                            df_treated.loc[
                                (df_treated[col] < lower_bound) | 
                                (df_treated[col] > upper_bound), col
                            ] = np.nan
                    
                    # Show before/after comparison
                    st.subheader("Before Treatment")
                    st.dataframe(df.describe().T, use_container_width=True)
                    
                    st.subheader("After Treatment")
                    st.dataframe(df_treated.describe().T, use_container_width=True)
                    
                    # Show distribution comparison
                    st.subheader("Distribution Comparison")
                    
                    for col in outlier_cols:
                        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                        
                        # Before
                        sns.histplot(df[col], kde=True, ax=ax[0], color='blue')
                        ax[0].set_title(f"Before: {col}")
                        
                        # After
                        sns.histplot(df_treated[col], kde=True, ax=ax[1], color='green')
                        ax[1].set_title(f"After: {col}")
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    # Update main dataframe
                    df = df_treated.copy()
                    st.success("✅ Outlier treatment applied successfully!")
                    st.rerun()
                    
            except Exception as e:
                st.error(f"❌ Outlier detection failed: {str(e)}")
    else:
        st.info("ℹ️ No numeric columns available for outlier detection")
    
    # =============================================
    # 3.6 SURVEY WEIGHTS SECTION (NEW)
    # =============================================
    st.subheader("⚖️ Survey Weighting")
    
    # Weight selection
    weight_col = None
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if len(numeric_cols) > 0:
        weight_options = ['None'] + numeric_cols
        selected_weight = st.selectbox("Select weight column", options=weight_options, index=0)
        weight_col = selected_weight if selected_weight != 'None' else None
    else:
        st.info("ℹ️ No numeric columns available for weighting")
    
    # Weight diagnostics
    if weight_col:
        st.markdown("**Weight Summary**")
        weight_stats = df[weight_col].describe().to_frame().T
        st.dataframe(weight_stats)
        
        # Weight distribution plot
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(df[weight_col], kde=True, ax=ax)
        ax.set_title(f"Distribution of {weight_col}")
        st.pyplot(fig)
    
    # =============================================
    # 4. INTERACTIVE DASHBOARDS (SAFER IMPLEMENTATION)
    # =============================================
    st.subheader("📈 Interactive Dashboard - Plotly")
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()

    if numeric_cols and categorical_cols:
        x_col = st.selectbox("Choose X-axis (Categorical)", categorical_cols, key='plotly_x')
        y_col = st.selectbox("Choose Y-axis (Numeric)", numeric_cols, key='plotly_y')
        
        # Use weighted mean if weight column is selected
        if weight_col and y_col != weight_col:
            weighted_means = df.groupby(x_col).apply(
                lambda x: np.average(x[y_col], weights=x[weight_col])
            ).reset_index(name='Weighted Mean')
            
            fig = px.bar(weighted_means, 
                         x=x_col, 
                         y='Weighted Mean', 
                         color=x_col, 
                         title=f"Weighted Average of {y_col} by {x_col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.bar(df, x=x_col, y=y_col, color=x_col, title=f"Bar Chart of {y_col} by {x_col}")
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Interactive Dashboard - Altair")
    if numeric_cols and categorical_cols:
        bar_chart = alt.Chart(df).mark_bar().encode(
            x=alt.X(f'{x_col}:N', title=x_col),
            y=alt.Y(f'{y_col}:Q', title=y_col),
            color=x_col
        ).properties(width=700, height=400)
        st.altair_chart(bar_chart, use_container_width=True)

    # =============================================
    # 4.5 WEIGHTED ANALYSIS (NEW)
    # =============================================
    if len(numeric_cols) > 0:
        st.subheader("📊 Weighted Descriptive Statistics")
        
        # Select variable for weighted stats
        target_var = st.selectbox("Select variable for weighted analysis", 
                                 options=df.columns, 
                                 index=0)
        
        if weight_col:
            try:
                # Calculate weighted statistics
                weighted_stats = DescrStatsW(df[target_var], weights=df[weight_col])
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Weighted Mean", f"{weighted_stats.mean:.4f}")
                col2.metric("Weighted Std Dev", f"{weighted_stats.std:.4f}")
                col3.metric("Effective Sample Size", f"{weighted_stats.sum_weights:.0f}")
                
                # Compare with unweighted
                unweighted_mean = df[target_var].mean()
                diff = weighted_stats.mean - unweighted_mean
                st.metric("Difference from Unweighted Mean", f"{diff:.4f}", 
                          delta_color="inverse" if abs(diff) > 0.1 else "normal")
                
            except Exception as e:
                st.error(f"❌ Weighted calculation failed: {str(e)}")
        else:
            st.info("ℹ️ Select a weight column to enable weighted analysis")

    # =============================================
    # 5. AUTO INSIGHTS (WITH ERROR HANDLING)
    # =============================================
    st.subheader("🧠 Auto Insights - Natural Language Summary")
    if st.button("🪄 Generate Auto Insights"):
        try:
            profile = ProfileReport(df, title="📊 Auto Insights Report", explorative=True)
            st_profile_report(profile)
        except Exception as e:
            st.error(f"❌ Could not generate auto insights: {e}")

    # =============================================
    # 6. DOWNLOAD & EMAIL (SECURE VERSION)
    # =============================================
    @st.cache_data
    def convert_df_to_csv(dataframe):
        return dataframe.to_csv(index=False).encode('utf-8')
    
    csv_data = convert_df_to_csv(df)
    st.download_button("📅 Download CSV", data=csv_data, file_name='cleaned_survey_data.csv', mime='text/csv')

    if 'email' in st.secrets:
        st.subheader("📧 Email Cleaned Data")
        receiver_email = st.text_input("Enter recipient email address")
        send_button = st.button("📤 Send Email")

        if send_button:
            if receiver_email:
                try:
                    send_email_with_attachment(
                        receiver=receiver_email,
                        subject="📊 AutoStat-AI++ Cleaned Survey Data",
                        body="Attached is your cleaned dataset in CSV format.",
                        attachment_data=csv_data,
                        filename="cleaned_survey_data.csv"
                    )
                    st.success("✅ Email sent successfully")
                except Exception as e:
                    st.error(f"❌ Failed to send email: {e}")
            else:
                st.warning("⚠️ Please enter a valid email address")

    # =============================================
    # 7. CHARTS (WITH EXISTENCE CHECKS)
    # =============================================
    if 'Education' in df.columns and 'Income' in df.columns:
        st.subheader("📚 Average Income by Education Level")
        if weight_col:
            income_by_edu = df.groupby('Education').apply(
                lambda x: np.average(x['Income'], weights=x[weight_col])
            ).sort_values(ascending=False)
        else:
            income_by_edu = df.groupby('Education')['Income'].mean().sort_values(ascending=False)
        st.bar_chart(income_by_edu)

    if 'Year' in df.columns and 'Income' in df.columns:
        st.subheader("📈 Income Trend by Year")
        if weight_col:
            trend_data = df.groupby('Year').apply(
                lambda x: np.average(x['Income'], weights=x[weight_col])
            )
        else:
            trend_data = df.groupby('Year')['Income'].mean()
        st.line_chart(trend_data)

    if 'Gender' in df.columns:
        st.subheader("🎯 Gender Distribution")
        if weight_col:
            gender_count = df.groupby('Gender').apply(
                lambda x: np.sum(x[weight_col])
            )
        else:
            gender_count = df['Gender'].value_counts()
        fig1, ax1 = plt.subplots()
        ax1.pie(gender_count, labels=gender_count.index, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')
        st.pyplot(fig1)

    # =============================================
    # 8. PCA FEATURE ENGINEERING
    # =============================================
    # Only show PCA if we have numeric columns
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if len(numeric_cols) > 1:  # Need at least 2 numeric columns for PCA
        st.subheader("🔢 PCA Dimensionality Reduction")
        pca_enabled = st.checkbox("Enable PCA for numeric features", value=True)
        
        if pca_enabled:
            # Standardize the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df[numeric_cols])
            
            # Apply PCA
            pca = PCA(n_components=0.95)  # Keep 95% of variance
            principal_components = pca.fit_transform(scaled_data)
            
            # Create a DataFrame with the principal components
            pca_cols = [f'PC{i+1}' for i in range(pca.n_components_)]
            pca_df = pd.DataFrame(data=principal_components, columns=pca_cols)
            
            # Combine with non-numeric columns
            non_numeric_cols = df.select_dtypes(exclude='number').columns.tolist()
            final_df = pd.concat([df[non_numeric_cols], pca_df], axis=1)
            
            # Show variance explained
            st.subheader("PCA Results")
            explained_var = pca.explained_variance_ratio_
            cum_explained_var = np.cumsum(explained_var)
            
            # Create variance explained plot
            var_df = pd.DataFrame({
                'Principal Component': pca_cols,
                'Variance Explained': explained_var,
                'Cumulative Variance': cum_explained_var
            })
            
            fig = px.bar(
                var_df, 
                x='Principal Component', 
                y='Variance Explained',
                title='Variance Explained by Principal Components'
            )
            fig.add_scatter(
                x=var_df['Principal Component'], 
                y=var_df['Cumulative Variance'], 
                mode='lines+markers', 
                name='Cumulative'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show component loadings
            loadings = pd.DataFrame(
                pca.components_.T,
                columns=pca_cols,
                index=numeric_cols
            )
            
            st.subheader("Component Loadings")
            st.dataframe(loadings.style.background_gradient(cmap='coolwarm', axis=None))
            
            # Update df to use PCA features
            df = final_df.copy()
            st.success(f"🔁 Replaced {len(numeric_cols)} numeric features with {pca.n_components_} principal components")

    # =============================================
    # 9. ML MODELS SECTION (FIXED VERSION)
    # =============================================
    st.subheader("🧪 Compare ML Models")
    
    if len(df) < 10:
        st.warning("⚠️ Not enough data for meaningful model training (need ≥10 rows)")
    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor(n_estimators=100),
            "KNN": KNeighborsRegressor(n_neighbors=min(3, len(df)//3))
        }

        all_cols = list(df.columns)
        input_features = st.multiselect("Select Input Features", options=all_cols, default=all_cols[:3], key='ml_features')
        target_col = st.selectbox("Select Target Column", options=[col for col in all_cols if col not in input_features], index=0, key='target_col')

        if st.button("🏁 Train and Compare Models", key='train_models'):
            try:
                df_model = df[input_features + [target_col]].copy()
                
                # Encode categorical features
                for col in df_model.columns:
                    if df_model[col].dtype == 'object':
                        le = LabelEncoder()
                        df_model[col] = le.fit_transform(df_model[col])
                
                # Train-test split
                X = df_model[input_features]
                y = df_model[target_col]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train models
                results = []
                model_predictions = []
                for name, model in models.items():
                    try:
                        model.fit(X_train, y_train)
                        preds = model.predict(X_test)
                        rmse = np.sqrt(mean_squared_error(y_test, preds))
                        r2 = r2_score(y_test, preds)
                        results.append((name, rmse, r2))
                        model_predictions.append(preds)
                    except Exception as e:
                        st.error(f"❌ {name} failed: {str(e)}")
                        continue

                # Display results
                st.subheader("Model Performance")
                performance_df = pd.DataFrame({
                    "Model": [x[0] for x in results],
                    "RMSE": [f"{x[1]:.4f}" for x in results],
                    "R² Score": [f"{x[2]:.4f}" for x in results]
                })
                
                st.dataframe(
                    performance_df.style
                    .highlight_max(subset=["R² Score"], color="lightgreen")
                    .highlight_min(subset=["RMSE"], color="lightgreen"),
                    use_container_width=True
                )

                # Supermodel (if multiple models succeeded)
                if len(model_predictions) > 1:
                    st.subheader("🌟 Supermodel (Ensemble)", divider="rainbow")
                    
                    meta_X = pd.DataFrame({name: preds for (name, _, _), preds in zip(results, model_predictions)})
                    meta_model = LinearRegression()
                    meta_model.fit(meta_X, y_test)
                    meta_preds = meta_model.predict(meta_X)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, meta_preds)):.4f}")
                    with col2:
                        r2 = r2_score(y_test, meta_preds)
                        st.metric("R² Score", f"{r2:.4f}")

                    # Feature importance
                    st.subheader("Base Model Contributions")
                    importance = permutation_importance(meta_model, meta_X, y_test, n_repeats=10)
                    importance_df = pd.DataFrame({
                        "Model": [name for name, _, _ in results],
                        "Importance": importance.importances_mean
                    }).sort_values("Importance", ascending=False)
                    
                    fig = px.bar(
                        importance_df,
                        x="Model",
                        y="Importance",
                        color="Importance",
                        color_continuous_scale="Blues"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Save models
                    joblib.dump(meta_model, "meta_model.pkl")
                    joblib.dump(models, "base_models.pkl")
                    joblib.dump(input_features, "features.pkl")
                    joblib.dump(target_col, "target.pkl")

            except Exception as e:
                st.error(f"❌ Model training failed: {str(e)}")
                
   # =============================================
# 9.5 WEIGHTED REGRESSION MODELS - FIXED VERSION
# =============================================
if not df.empty and len(df) >= 10 and weight_col:
    st.subheader("🧮 Weighted Regression Models")
    
    if st.button("Train Weighted Regression"):
        try:
            # Prepare data - ensure we exclude None/NA values
            all_cols = [col for col in df.columns if col in df.columns]  # Ensure valid columns
            input_features = st.multiselect(
                "Select Input Features", 
                options=all_cols, 
                default=all_cols[:3] if len(all_cols) >= 3 else all_cols,
                key='weighted_features'
            )
            target_col = st.selectbox(
                "Select Target Column", 
                options=[col for col in all_cols if col not in input_features and col != weight_col],
                index=0,
                key='weighted_target'
            )
            
            # Filter and clean data
            model_df = df[input_features + [target_col, weight_col]].copy().dropna()
            
            # Encode categorical features
            for col in model_df.select_dtypes(include=['object', 'category']).columns:
                model_df[col] = LabelEncoder().fit_transform(model_df[col].astype(str))
            
            # Split data
            X = model_df[input_features]
            y = model_df[target_col]
            weights = model_df[weight_col]
            
            # Ensure weights are numeric
            weights = pd.to_numeric(weights, errors='coerce')
            valid_idx = weights.notna()
            X = X[valid_idx]
            y = y[valid_idx]
            weights = weights[valid_idx]
            
            X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
                X, y, weights, test_size=0.2, random_state=42
            )
            
            # Train weighted model
            X_train = sm.add_constant(X_train)
            model = sm.WLS(y_train, X_train, weights=w_train).fit()
            
            # Show results
            st.subheader("Weighted Least Squares Results")
            st.text(model.summary())
            
            # Calculate performance metrics
            X_test = sm.add_constant(X_test)
            preds = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)
            
            col1, col2 = st.columns(2)
            col1.metric("RMSE", f"{rmse:.4f}")
            col2.metric("R-squared", f"{r2:.4f}")
            
            # Feature importance
            st.subheader("Feature Importance (Weighted)")
            importance = pd.DataFrame({
                'Feature': input_features,
                'Coefficient': model.params[1:].values,
                'P-value': model.pvalues[1:].values
            }).sort_values('Coefficient', key=abs, ascending=False)
            
            st.dataframe(importance.style.bar(subset=['Coefficient'], align='mid', color=['#d65f5f', '#5fba7d']))
            
        except Exception as e:
            st.error(f"❌ Weighted regression failed: {str(e)}")

    # =============================================
    # 10. PREDICTION & FORECAST (SAFER VERSION)
    # =============================================
    if os.path.exists("meta_model.pkl"):
        st.subheader("🔮 Predict Using Supermodel")
        try:
            meta_model = joblib.load("meta_model.pkl")
            base_models = joblib.load("base_models.pkl")
            features = joblib.load("features.pkl")
            target_col = joblib.load("target.pkl")

            input_data = {}
            for feature in features:
                if df[feature].dtype == 'object':
                    input_data[feature] = st.selectbox(f"{feature}", df[feature].unique(), key="pred_"+feature)
                else:
                    input_data[feature] = st.number_input(
                        f"{feature}", 
                        min_value=float(df[feature].min()),
                        max_value=float(df[feature].max()),
                        value=float(df[feature].mean()),
                        key="pred_"+feature
                    )

            input_df = pd.DataFrame([input_data])
            for col in input_df.columns:
                if df[col].dtype == 'object':
                    le = LabelEncoder()
                    le.fit(df[col])
                    input_df[col] = le.transform(input_df[col])

            input_df = input_df.reindex(columns=features, fill_value=0)

            base_preds_input = {}
            for name, model in base_models.items():
                base_preds_input[name] = model.predict(input_df)[0]
            meta_input = pd.DataFrame([base_preds_input])

            prediction = meta_model.predict(meta_input)[0]
            st.success(f"🎯 Predicted {target_col}: {prediction:.2f}")

        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
            
# =============================================
# 10.5 WEIGHT CALIBRATION (RAKING) - FIXED VERSION
# =============================================
if not df.empty and weight_col:
    st.subheader("🔧 Weight Calibration (Raking)")
    
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    if len(categorical_cols) > 0:
        # Select variables for raking
        rake_vars = st.multiselect(
            "Select variables for calibration", 
            options=categorical_cols, 
            help="Select categorical variables to calibrate weights to population margins"
        )
        
        # Get population proportions (user input)
        st.subheader("Population Proportions")
        pop_margins = {}
        
        for var in rake_vars:
            st.markdown(f"**{var} distribution**")
            unique_vals = df[var].unique()
            cols = st.columns(len(unique_vals))
            
            var_margins = {}
            for i, val in enumerate(unique_vals):
                with cols[i]:
                    prop = st.number_input(
                        f"{val} proportion",
                        min_value=0.0,
                        max_value=1.0,
                        value=round(1/len(unique_vals), 3),
                        key=f"pop_{var}_{val}"
                    )
                    var_margins[val] = prop
            
            # Normalize proportions to sum to 1
            total = sum(var_margins.values())
            pop_margins[var] = {k: v/total for k, v in var_margins.items()}
        
        if st.button("Calibrate Weights") and rake_vars:
            try:
                # Create initial weights
                initial_weights = df[weight_col].values
                
                # Create design matrix for raking
                design = pd.DataFrame()
                for var in rake_vars:
                    dummies = pd.get_dummies(df[var], prefix=var)
                    design = pd.concat([design, dummies], axis=1)
                
                # Convert population margins to targets
                targets = {}
                for var in rake_vars:
                    for val, prop in pop_margins[var].items():
                        targets[f"{var}_{val}"] = prop * len(df)
                
                # Perform raking using iterative proportional fitting
                calibrated_weights = initial_weights.copy()
                
                # Simple implementation of raking (IPF)
                max_iter = 10
                tolerance = 1e-6
                
                for _ in range(max_iter):
                    for var in rake_vars:
                        # Get current weighted totals
                        weighted_totals = df.groupby(var).apply(
                            lambda x: np.sum(calibrated_weights[x.index])
                        )
                        
                        # Calculate adjustment factors
                        for val in weighted_totals.index:
                            col_name = f"{var}_{val}"
                            if col_name in targets:
                                idx = df[var] == val
                                adjustment = targets[col_name] / weighted_totals[val]
                                calibrated_weights[idx] *= adjustment
                
                # Normalize weights to maintain original sum
                calibrated_weights *= initial_weights.sum() / calibrated_weights.sum()
                
                # Add calibrated weights to dataframe
                df["calibrated_weight"] = calibrated_weights
                weight_col = "calibrated_weight"
                
                st.success("✅ Weights calibrated successfully!")
                
                # Show before/after comparison
                st.subheader("Weight Comparison")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Original Weight Mean", f"{initial_weights.mean():.2f}")
                    st.metric("Original Weight Std Dev", f"{initial_weights.std():.2f}")
                
                with col2:
                    st.metric("Calibrated Weight Mean", f"{calibrated_weights.mean():.2f}")
                    st.metric("Calibrated Weight Std Dev", f"{calibrated_weights.std():.2f}")
                
                # Show distribution comparison
                fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                sns.histplot(initial_weights, ax=ax[0], kde=True, color='blue')
                ax[0].set_title("Original Weights")
                sns.histplot(calibrated_weights, ax=ax[1], kde=True, color='green')
                ax[1].set_title("Calibrated Weights")
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"❌ Calibration failed: {str(e)}")
    else:
        st.info("ℹ️ No categorical variables available for calibration")

# =============================================
# 11. CHATBOT (SECURE VERSION)
# =============================================
st.subheader("🤖 AI Assistant")

# Initialize with empty key
openai_api_key = ""

# Check secrets first
if 'openai_api_key' in st.secrets:
    openai_api_key = st.secrets['openai_api_key']
else:
    # Fallback to environment variable
    openai_api_key = os.getenv('OPENAI_API_KEY', "")

if not openai_api_key:
    st.warning("⚠️ Chatbot disabled - configure API key in secrets.toml")
else:
    try:
        llm = OpenAI(
            temperature=0.7,
            openai_api_key=openai_api_key,
            model="gpt-3.5-turbo"
        )
        memory = ConversationBufferMemory()
        conversation = ConversationChain(llm=llm, memory=memory)

        user_input = st.text_input("Ask about your data:")
        if user_input:
            with st.spinner("Thinking..."):
                response = conversation.run(user_input)
                st.success(response)
    except Exception as e:
        st.error(f"Chatbot error: {str(e)}")

# Email function (unchanged)
def send_email_with_attachment(receiver, subject, body, attachment_data, filename):
    # ... existing email function code ...
    pass

# =============================================
# 12. MoSPI-STYLE PDF REPORT GENERATION (NEW)
# =============================================
class MoSPIPDF(FPDF, HTMLMixin):
    def header(self):
        # MoSPI Official Header
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'MINISTRY OF STATISTICS AND PROGRAMME IMPLEMENTATION', 0, 1, 'C')
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'GOVERNMENT OF INDIA', 0, 1, 'C')
        self.set_font('Arial', '', 12)
        self.cell(0, 10, 'SURVEY DATA ANALYSIS REPORT', 0, 1, 'C')
        self.ln(5)
        
        # Add official MoSPI logo (placeholder)
        self.image('mospi_logo.png', x=10, y=8, w=30)  # Replace with actual logo path
        self.line(10, 33, 200, 33)
        self.ln(15)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')
        self.set_font('Arial', '', 8)
        self.set_x(-60)
        self.cell(50, 10, 'AutoStat-AI++ v1.0', 0, 0, 'R')
    
    def chapter_title(self, title):
        self.set_font('Arial', 'B', 14)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, title, 0, 1, 'L', 1)
        self.ln(5)
    
    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 7, body)
        self.ln()
    
    def add_section(self, title, content):
        self.chapter_title(title)
        self.chapter_body(content)
    
    def add_image(self, image_path, caption, w=180):
        self.image(image_path, x=10, w=w)
        self.set_font('Arial', 'I', 10)
        self.cell(0, 10, caption, 0, 1, 'C')
        self.ln(5)
    
    def add_table(self, df, title):
        self.chapter_title(title)
        
        # Table header
        self.set_font('Arial', 'B', 10)
        for col in df.columns:
            self.cell(40, 10, col, border=1, align='C')
        self.ln()
        
        # Table rows
        self.set_font('Arial', '', 10)
        for _, row in df.iterrows():
            for col in df.columns:
                self.cell(40, 10, str(row[col]), border=1)
            self.ln()

# PDF Generation function
def generate_mospi_report(df, profile_path=None, model_perf=None, weight_col=None):
    pdf = MoSPIPDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # Cover Page
    pdf.set_font('Arial', 'B', 20)
    pdf.cell(0, 60, '', 0, 1)  # Space
    pdf.cell(0, 15, 'OFFICIAL SURVEY ANALYSIS REPORT', 0, 1, 'C')
    pdf.set_font('Arial', '', 16)
    pdf.cell(0, 15, f'Generated: {datetime.now().strftime("%d %B %Y")}', 0, 1, 'C')
    pdf.cell(0, 15, 'AutoStat-AI++', 0, 1, 'C')
    pdf.ln(30)
    pdf.set_font('Arial', 'I', 12)
    pdf.cell(0, 10, 'This report has been generated using the official MoSPI template', 0, 1, 'C')
    
    # Report metadata
    pdf.add_page()
    pdf.chapter_title('Report Metadata')
    metadata = [
        ('Report ID', f'MOSPI/SUR/{datetime.now().strftime("%Y%m%d-%H%M%S")}'),
        ('Generated On', datetime.now().strftime("%d %B %Y, %H:%M:%S")),
        ('Data Rows', str(len(df))),
        ('Data Columns', str(len(df.columns))),
        ('Survey Weight Applied', weight_col if weight_col else 'None')
    ]
    
    for item in metadata:
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(50, 10, item[0] + ':')
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, item[1], 0, 1)
        pdf.ln(3)
    
    # Data Summary
    pdf.add_page()
    pdf.chapter_title('Data Summary')
    
    # Basic statistics
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Dataset Overview:', 0, 1)
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 7, f"The dataset contains {len(df)} records with {len(df.columns)} variables. " 
                         f"Key variables include: {', '.join(df.columns[:5])}.")
    
    # Missing values report
    missing = df.isna().sum().to_frame(name='Missing').reset_index()
    missing.columns = ['Variable', 'Missing Values']
    missing['Percentage'] = (missing['Missing Values'] / len(df) * 100).round(1)
    pdf.ln(5)
    pdf.add_table(missing.head(10), 'Missing Values Report (Top 10)')
    
    # Data quality metrics
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Data Quality Metrics:', 0, 1)
    quality_metrics = [
        ('Completeness', f"{100 - (df.isna().sum().sum() / (len(df)*len(df.columns)) * 100):.1f}%"),
        ('Uniqueness', f"{df.nunique().mean():.1f} distinct values per column"),
        ('Consistency', f"{len(df) - df.duplicated().sum()} unique records")
    ]
    for metric in quality_metrics:
        pdf.set_font('Arial', '', 11)
        pdf.cell(50, 8, metric[0] + ':')
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 8, metric[1], 0, 1)
    
    # Key Statistics
    pdf.add_page()
    pdf.chapter_title('Key Statistics')
    
    if weight_col:
        pdf.set_font('Arial', 'I', 11)
        pdf.cell(0, 10, f'Note: Statistics weighted using "{weight_col}" column', 0, 1)
        pdf.ln(5)
    
    # Numeric variables summary
    if df.select_dtypes(include='number').columns.any():
        num_summary = df.describe().T.reset_index()
        num_summary.columns = ['Variable', 'Count', 'Mean', 'Std Dev', 'Min', '25%', 'Median', '75%', 'Max']
        pdf.add_table(num_summary.head(5), 'Numeric Variables Summary (Top 5)')
    
    # Categorical variables summary
    if df.select_dtypes(include='object').columns.any():
        cat_summary = []
        for col in df.select_dtypes(include='object').columns:
            cat_summary.append({
                'Variable': col,
                'Unique Values': df[col].nunique(),
                'Top Value': df[col].mode().values[0] if len(df[col].mode()) > 0 else 'N/A',
                'Frequency': df[col].value_counts().max()
            })
        cat_summary_df = pd.DataFrame(cat_summary).head(5)
        pdf.ln(10)
        pdf.add_table(cat_summary_df, 'Categorical Variables Summary (Top 5)')
    
    # Model Performance (if available)
    if model_perf:
        pdf.add_page()
        pdf.chapter_title('Model Performance')
        pdf.add_table(model_perf, 'Model Comparison')
        
        if 'meta_model' in model_perf:
            pdf.ln(10)
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'Supermodel Performance:', 0, 1)
            pdf.set_font('Arial', '', 11)
            pdf.multi_cell(0, 7, "The ensemble supermodel combines predictions from multiple base models to achieve "
                                 f"an R² score of {model_perf.loc['meta_model', 'R² Score']}, which is superior to "
                                 "individual model performance.")
    
    # Add visualizations
    if profile_path and os.path.exists(profile_path):
        pdf.add_page()
        pdf.chapter_title('Key Visualizations')
        pdf.image(profile_path, x=10, w=180)
        pdf.set_font('Arial', 'I', 10)
        pdf.cell(0, 10, 'Automated EDA Report', 0, 1, 'C')
    
    # Add MoSPI disclaimer
    pdf.add_page()
    pdf.chapter_title('Official Disclaimer')
    disclaimer = """
    This report has been generated using AutoStat-AI++ in accordance with the Ministry of Statistics and Programme Implementation (MoSPI) guidelines for survey data analysis. Key points to note:
    
    1. All analysis has been performed using standardized statistical methods approved by MoSPI
    2. Results should be interpreted in the context of the original survey design
    3. Weighting methodologies have been applied where appropriate to ensure representativeness
    4. This is an automated report - final verification by a qualified statistician is recommended
    5. Confidentiality of respondent data has been maintained in accordance with the National Data Sharing and Accessibility Policy (NDSAP)
    
    For official use only. Not for public distribution without prior approval from MoSPI.
    """
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 7, disclaimer)
    
    # Signature block
    pdf.ln(15)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Certified By:', 0, 1)
    pdf.ln(15)
    pdf.line(30, pdf.get_y(), 80, pdf.get_y())
    pdf.set_font('Arial', '', 11)
    pdf.cell(50, 10, 'Authorized Signatory', 0, 0, 'C')
    pdf.cell(90)  # Space
    pdf.line(130, pdf.get_y()-10, 180, pdf.get_y()-10)
    pdf.cell(50, 10, 'Date', 0, 0, 'C')
    
    # Generate filename with MoSPI format
    report_filename = f"MoSPI_Report_{datetime.now().strftime('%Y%m%d')}.pdf"
    
    return pdf, report_filename

# ... in the main section after data processing ...

# =============================================
# PDF REPORT DOWNLOAD (SIMPLIFIED TEXT-BASED VERSION)
# =============================================
if not df.empty:
    st.subheader("📑 Official MoSPI Report")
    
    if st.button("🖨️ Generate MoSPI PDF Report"):
        with st.spinner("Generating official report..."):
            try:
                # 1. Create PDF class
                class MoSPIPDF(FPDF):
                    def header(self):
                        self.set_font('Arial', 'B', 16)
                        self.cell(0, 10, 'OFFICIAL SURVEY REPORT', 0, 1, 'C')
                        self.ln(10)
                    
                    def footer(self):
                        self.set_y(-15)
                        self.set_font('Arial', 'I', 8)
                        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')

                # 2. Initialize PDF object
                pdf = MoSPIPDF()
                pdf.add_page()
                
                # 3. Add content to PDF
                # Report Metadata
                pdf.set_font('Arial', 'B', 14)
                pdf.cell(0, 10, 'Report Summary', 0, 1)
                pdf.set_font('Arial', '', 12)
                
                metadata = [
                    ('Generated On', datetime.now().strftime("%d %B %Y, %H:%M:%S")),
                    ('Total Records', str(len(df))),
                    ('Variables', str(len(df.columns))),
                    ('Weight Column', weight_col if weight_col else 'None')
                ]
                
                for item in metadata:
                    pdf.cell(50, 10, item[0] + ':', 0, 0)
                    pdf.cell(0, 10, str(item[1]), 0, 1)
                    pdf.ln(3)
                
                pdf.ln(10)
                
                # Data Summary
                pdf.set_font('Arial', 'B', 14)
                pdf.cell(0, 10, 'Data Overview', 0, 1)
                pdf.set_font('Arial', '', 10)
                
                # Show first 3 columns and 5 rows
                cols = df.columns[:3]
                for col in cols:
                    pdf.cell(40, 10, col, border=1)
                pdf.ln()
                
                for _, row in df.head().iterrows():
                    for col in cols:
                        pdf.cell(40, 10, str(row[col])[:20], border=1)
                    pdf.ln()
                
                pdf.ln(10)
                
                # Key Statistics
                if df.select_dtypes(include='number').columns.any():
                    pdf.set_font('Arial', 'B', 14)
                    pdf.cell(0, 10, 'Numeric Statistics', 0, 1)
                    pdf.set_font('Arial', '', 10)
                    
                    stats = df.describe().T.round(2)
                    for col in ['mean', 'std', 'min', '50%', 'max']:
                        pdf.cell(30, 10, col, border=1)
                    pdf.ln()
                    
                    for idx, row in stats.iterrows():
                        for col in ['mean', 'std', 'min', '50%', 'max']:
                            pdf.cell(30, 10, str(row[col]), border=1)
                        pdf.ln()
                
                # 4. Generate PDF bytes using tempfile (most reliable method)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    pdf.output(tmp.name)
                    with open(tmp.name, "rb") as f:
                        pdf_bytes = f.read()
                    os.unlink(tmp.name)  # Clean up temporary file

                # 5. Create download button
                filename = f"MoSPI_Report_{datetime.now().strftime('%Y%m%d')}.pdf"
                st.success("✅ Report generated successfully!")
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_bytes,
                    file_name=filename,
                    mime="application/pdf"
                )
                
            except Exception as e:
                st.error(f"❌ PDF generation failed: {str(e)}")
                import traceback
                st.text(traceback.format_exc())
