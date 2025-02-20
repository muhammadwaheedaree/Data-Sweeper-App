import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import os
import json
from typing import List, Dict, Any
import base64

# Configure the Streamlit app's appearance and layout
# 'page_title' sets the browser tab title
# 'layout="wide"' allows more horizontal space, improving the display for tables and graphs
st.set_page_config(page_title="Data Sweeper", layout="wide")

# Custom CSS for styling the app with dark mode aesthetics
# This enhances the UI by setting background colors, button styles, and text formatting
st.markdown(
    """
    <style>
        .main {
            background-color: #121212;  /* Overall dark background for the main page */
        }
        .block-container {
            padding: 3rem 2rem;  /* Padding around main container for spacing */
            border-radius: 12px;  /* Rounds the corners of the container */
            background-color: #1e1e1e;  /* Slightly lighter shade for contrast */
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);  /* Adds subtle shadow for depth */
        }
        h1, h2, h3, h4, h5, h6 {
            color: #66c2ff;  /* Light blue color for headings to stand out */
        }
        .stButton>button {
            border: none;
            border-radius: 8px;  /* Rounds button edges */
            background-color: #0078D7;  /* Primary blue for buttons */
            color: white;  /* White text for contrast */
            padding: 0.75rem 1.5rem;  /* Enlarges button for better interaction */
            font-size: 1rem;  /* Readable button text */
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.4);  /* Shadow for button depth */
        }
        .stButton>button:hover {
            background-color: #005a9e;  /* Darker blue on hover for visual feedback */
            cursor: pointer;
        }
        .stDataFrame, .stTable {
            border-radius: 10px;  /* Smooth edges for data tables and frames */
            overflow: hidden;  /* Prevents data from overflowing the container */
        }
        .css-1aumxhk, .css-18e3th9 {
            text-align: left;
            color: white;  /* Ensures all standard text is white for readability */
        }
        .stRadio>label {
            font-weight: bold;
            color: white;
        }
        .stCheckbox>label {
            color: white;
        }
        .stDownloadButton>button {
            background-color: #28a745;  /* Green color for download buttons */
            color: white;
        }
        .stDownloadButton>button:hover {
            background-color: #218838;  /* Darker green on hover for download buttons */
        }
    </style>
    """,
    unsafe_allow_html=True  # 'unsafe_allow_html' permits raw HTML/CSS embedding in the Streamlit app
)

# Utility functions for data processing
def generate_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Generate descriptive statistics for numerical columns."""
    return df.describe()

def generate_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Generate correlation matrix for numerical columns."""
    numeric_cols = df.select_dtypes(include=['number']).columns
    return df[numeric_cols].corr()

def get_value_counts(df: pd.DataFrame, column: str) -> pd.Series:
    """Get value counts for a specific column."""
    return df[column].value_counts()

def clean_data(df: pd.DataFrame, cleaning_options: Dict[str, Any]) -> pd.DataFrame:
    """Apply various cleaning operations to the dataframe."""
    if cleaning_options.get('drop_null_rows'):
        df = df.dropna(how='all')
    
    if cleaning_options.get('drop_null_cols'):
        df = df.dropna(axis=1, how='all')
    
    if cleaning_options.get('custom_fill_value') is not None:
        df = df.fillna(cleaning_options['custom_fill_value'])
    
    if cleaning_options.get('columns_to_drop'):
        df = df.drop(columns=cleaning_options['columns_to_drop'])
    
    return df

def convert_column_types(df: pd.DataFrame, column_types: Dict[str, str]) -> pd.DataFrame:
    """Convert column types based on user selection."""
    for column, dtype in column_types.items():
        try:
            if dtype == 'integer':
                df[column] = pd.to_numeric(df[column], errors='coerce').astype('Int64')
            elif dtype == 'float':
                df[column] = pd.to_numeric(df[column], errors='coerce')
            elif dtype == 'datetime':
                df[column] = pd.to_datetime(df[column], errors='coerce')
            elif dtype == 'string':
                df[column] = df[column].astype(str)
        except Exception as e:
            st.warning(f"Could not convert column {column} to {dtype}: {str(e)}")
    return df

def detect_outliers(df: pd.DataFrame, column: str, threshold: float = 1.5) -> pd.Series:
    """Detect outliers using IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    return (df[column] < lower_bound) | (df[column] > upper_bound)

def suggest_data_cleaning(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate AI-powered suggestions for data cleaning."""
    suggestions = {
        'missing_values': {},
        'outliers': {},
        'data_type_suggestions': {},
        'duplicate_rows': 0,
        'recommendations': []
    }
    
    # Analyze missing values
    missing_stats = df.isnull().sum()
    total_rows = len(df)
    
    for column in df.columns:
        missing_count = missing_stats[column]
        missing_percentage = (missing_count / total_rows) * 100
        
        if missing_count > 0:
            suggestions['missing_values'][column] = {
                'count': int(missing_count),
                'percentage': round(missing_percentage, 2),
                'suggestion': ''
            }
            
            # Suggest handling method based on percentage
            if missing_percentage > 70:
                suggestions['missing_values'][column]['suggestion'] = 'Consider dropping this column'
            elif missing_percentage < 5:
                if df[column].dtype in ['int64', 'float64']:
                    suggestions['missing_values'][column]['suggestion'] = 'Fill with median'
                else:
                    suggestions['missing_values'][column]['suggestion'] = 'Fill with mode'
            else:
                suggestions['missing_values'][column]['suggestion'] = 'Fill with mean/mode or use advanced imputation'
    
    # Detect potential outliers in numeric columns
    numeric_columns = df.select_dtypes(include=['number']).columns
    for column in numeric_columns:
        outliers = detect_outliers(df, column)
        outlier_count = outliers.sum()
        if outlier_count > 0:
            suggestions['outliers'][column] = {
                'count': int(outlier_count),
                'percentage': round((outlier_count / total_rows) * 100, 2)
            }
    
    # Suggest data type conversions
    for column in df.columns:
        # Check for potential datetime columns
        if df[column].dtype == 'object':
            try:
                pd.to_datetime(df[column], errors='raise')
                suggestions['data_type_suggestions'][column] = 'datetime'
            except:
                # Check for numeric columns stored as strings
                if df[column].str.match(r'^-?\d*\.?\d+$').all():
                    suggestions['data_type_suggestions'][column] = 'numeric'
    
    # Check for duplicate rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        suggestions['duplicate_rows'] = int(duplicate_count)
    
    # Generate general recommendations
    if suggestions['duplicate_rows'] > 0:
        suggestions['recommendations'].append(
            f"Found {suggestions['duplicate_rows']} duplicate rows. Consider removing them."
        )
    
    for col, missing in suggestions['missing_values'].items():
        suggestions['recommendations'].append(
            f"Column '{col}' has {missing['count']} missing values ({missing['percentage']}%). {missing['suggestion']}."
        )
    
    for col, outliers in suggestions['outliers'].items():
        suggestions['recommendations'].append(
            f"Column '{col}' has {outliers['count']} potential outliers ({outliers['percentage']}%). Consider investigating."
        )
    
    return suggestions

# Add language support
LANGUAGES = {
    'English': {
        'title': 'Advanced Data Sweeper',
        'upload_text': 'Upload your files (CSV or Excel):',
        'processing': 'Processing...',
        'success': 'Success!',
        # Add more translations as needed
    },
    'Urdu': {
        'title': 'ایڈوانسڈ ڈیٹا سویپر',
        'upload_text': 'اپنی فائلیں اپ لوڈ کریں (CSV یا Excel):',
        'processing': 'پروسیسنگ جاری ہے...',
        'success': 'کامیاب!',
        # Add more translations as needed
    }
}

def create_download_link(df: pd.DataFrame, format_type: str) -> tuple:
    """Create a download link for different file formats."""
    buffer = BytesIO()
    
    if format_type == 'csv':
        df.to_csv(buffer, index=False)
        mime_type = "text/csv"
        file_ext = ".csv"
    elif format_type == 'excel':
        df.to_excel(buffer, index=False, engine='openpyxl')
        mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        file_ext = ".xlsx"
    elif format_type == 'json':
        buffer.write(df.to_json(orient='records').encode())
        mime_type = "application/json"
        file_ext = ".json"
    
    buffer.seek(0)
    return buffer, mime_type, file_ext

def main():
    # Language selection
    language = st.sidebar.selectbox("Select Language", list(LANGUAGES.keys()))
    texts = LANGUAGES[language]
    
    # Theme selection
    theme = st.sidebar.radio("Theme", ["Dark", "Light"])
    if theme == "Light":
        st.markdown("""
            <style>
                .main { background-color: #ffffff; }
                .block-container { background-color: #f0f2f6; }
                h1, h2, h3, h4, h5, h6 { color: #1f1f1f; }
            </style>
        """, unsafe_allow_html=True)
    
    st.title(texts['title'])
    
    uploaded_files = st.file_uploader(
        texts['upload_text'],
        type=["csv", "xlsx"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        # Initialize session state for storing DataFrames if not exists
        if 'dataframes' not in st.session_state:
            st.session_state.dataframes = {}
        
        for file in uploaded_files:
            file_extension = os.path.splitext(file.name)[-1].lower()
            
            # Show progress bar for file loading
            with st.spinner(texts['processing']):
                progress_bar = st.progress(0)
                
                try:
                    if file_extension == ".csv":
                        df = pd.read_csv(file)
                    elif file_extension == ".xlsx":
                        df = pd.read_excel(file)
                    
                    st.session_state.dataframes[file.name] = df
                    progress_bar.progress(100)
                    
                except Exception as e:
                    st.error(f"Error loading file {file.name}: {str(e)}")
                    continue
            
            # Create tabs for different operations
            tabs = st.tabs([
                "Data Preview", "Cleaning", "Analysis",
                "Visualization", "Filtering", "Export"
            ])
            
            with tabs[0]:
                st.subheader("Data Preview")
                st.dataframe(df.head())
                st.write(f"Total rows: {len(df)}, Total columns: {len(df.columns)}")
            
            with tabs[1]:
                st.subheader("Data Cleaning")
                
                # Initialize suggestions variable
                suggestions = {}
                
                # AI Suggestions Section
                st.subheader("🤖 AI Suggestions")
                if st.checkbox("Show AI-powered cleaning suggestions"):
                    suggestions = suggest_data_cleaning(df)
                    
                    # Display recommendations
                    st.write("### 📋 Recommendations:")
                    for rec in suggestions['recommendations']:
                        st.info(rec)
                    
                    # Display detailed analysis in expandable sections
                    if suggestions['missing_values']:
                        with st.expander("📊 Missing Values Analysis"):
                            st.write("Detailed analysis of missing values in your dataset:")
                            for col, info in suggestions['missing_values'].items():
                                st.write(f"**{col}:**")
                                st.write(f"- Missing values: {info['count']} ({info['percentage']}%)")
                                st.write(f"- Suggestion: {info['suggestion']}")
                    
                    if suggestions['outliers']:
                        with st.expander("🔍 Outliers Detection"):
                            st.write("Potential outliers found in numeric columns:")
                            for col, info in suggestions['outliers'].items():
                                st.write(f"**{col}:** {info['count']} outliers ({info['percentage']}%)")
                    
                    if suggestions['data_type_suggestions']:
                        with st.expander("🔄 Data Type Suggestions"):
                            st.write("Suggested data type conversions:")
                            for col, suggested_type in suggestions['data_type_suggestions'].items():
                                st.write(f"- Convert '{col}' to {suggested_type}")
                
                # Automatic Cleaning Actions
                if st.checkbox("Enable Automatic Cleaning"):
                    auto_actions = []
                    
                    if suggestions.get('duplicate_rows', 0) > 0:
                        if st.checkbox("Remove duplicate rows"):
                            auto_actions.append(('remove_duplicates', None))
                    
                    for col, info in suggestions.get('missing_values', {}).items():
                        if info['suggestion'].startswith('Fill with'):
                            method = st.selectbox(
                                f"Handle missing values in {col}",
                                ['Skip', 'Mean', 'Median', 'Mode', 'Custom Value']
                            )
                            if method != 'Skip':
                                auto_actions.append(('fill_missing', (col, method)))
                
                    if auto_actions and st.button("Apply Automatic Cleaning"):
                        with st.spinner("Applying automatic cleaning..."):
                            for action, params in auto_actions:
                                if action == 'remove_duplicates':
                                    df = df.drop_duplicates()
                                elif action == 'fill_missing':
                                    col, method = params
                                    if method == 'Mean':
                                        df[col] = df[col].fillna(df[col].mean())
                                    elif method == 'Median':
                                        df[col] = df[col].fillna(df[col].median())
                                    elif method == 'Mode':
                                        df[col] = df[col].fillna(df[col].mode()[0])
                            
                            st.session_state.dataframes[file.name] = df
                            st.success("Automatic cleaning completed!")
                
                # Existing manual cleaning options
                st.subheader("Manual Cleaning Options")
                cleaning_options = {
                    'drop_null_rows': st.checkbox("Remove rows with all NULL values"),
                    'drop_null_cols': st.checkbox("Remove columns with all NULL values"),
                    'custom_fill_value': st.text_input("Custom value for filling NAs (leave empty to skip)"),
                    'columns_to_drop': st.multiselect("Select columns to drop", df.columns)
                }
                
                # Column type conversion
                st.subheader("Convert Column Types")
                col_to_convert = st.selectbox("Select column to convert", df.columns)
                new_type = st.selectbox("Convert to type", ['integer', 'float', 'datetime', 'string'])
                
                if st.button("Apply Changes"):
                    with st.spinner("Applying changes..."):
                        df = clean_data(df, cleaning_options)
                        if col_to_convert:
                            df = convert_column_types(df, {col_to_convert: new_type})
                        st.session_state.dataframes[file.name] = df
                        st.success("Changes applied successfully!")

            with tabs[2]:
                st.subheader("Data Analysis")
                
                # Descriptive Statistics
                if st.checkbox("Show Descriptive Statistics"):
                    st.write("### Descriptive Statistics")
                    st.dataframe(generate_descriptive_stats(df))
                
                # Correlation Matrix
                if st.checkbox("Show Correlation Matrix"):
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 0:
                        st.write("### Correlation Matrix")
                        corr_matrix = generate_correlation_matrix(df)
                        fig = px.imshow(corr_matrix,
                                      labels=dict(color="Correlation"),
                                      color_continuous_scale="RdBu")
                        st.plotly_chart(fig)
                    else:
                        st.warning("No numeric columns found for correlation analysis")
                
                # Value Counts for Selected Column
                if st.checkbox("Show Value Counts"):
                    selected_col = st.selectbox("Select column for value counts", df.columns)
                    st.write(f"### Value Counts for {selected_col}")
                    value_counts = get_value_counts(df, selected_col)
                    st.bar_chart(value_counts)

            with tabs[3]:
                st.subheader("Data Visualization")
                
                chart_type = st.selectbox(
                    "Select Chart Type",
                    ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Box Plot"]
                )
                
                numeric_columns = df.select_dtypes(include=['number']).columns
                
                if chart_type == "Bar Chart":
                    x_col = st.selectbox("Select X-axis column", df.columns)
                    y_col = st.selectbox("Select Y-axis column", numeric_columns)
                    fig = px.bar(df, x=x_col, y=y_col)
                    st.plotly_chart(fig)
                
                elif chart_type == "Line Chart":
                    x_col = st.selectbox("Select X-axis column", df.columns)
                    y_col = st.selectbox("Select Y-axis column", numeric_columns)
                    fig = px.line(df, x=x_col, y=y_col)
                    st.plotly_chart(fig)
                
                elif chart_type == "Scatter Plot":
                    x_col = st.selectbox("Select X-axis column", numeric_columns)
                    y_col = st.selectbox("Select Y-axis column", numeric_columns)
                    color_col = st.selectbox("Select Color column (optional)", ["None"] + list(df.columns))
                    if color_col == "None":
                        fig = px.scatter(df, x=x_col, y=y_col)
                    else:
                        fig = px.scatter(df, x=x_col, y=y_col, color=color_col)
                    st.plotly_chart(fig)
                
                elif chart_type == "Pie Chart":
                    value_col = st.selectbox("Select Value column", numeric_columns)
                    name_col = st.selectbox("Select Name column", df.columns)
                    fig = px.pie(df, values=value_col, names=name_col)
                    st.plotly_chart(fig)
                
                elif chart_type == "Box Plot":
                    y_col = st.selectbox("Select column for box plot", numeric_columns)
                    x_col = st.selectbox("Select grouping column (optional)", ["None"] + list(df.columns))
                    if x_col == "None":
                        fig = px.box(df, y=y_col)
                    else:
                        fig = px.box(df, x=x_col, y=y_col)
                    st.plotly_chart(fig)

            with tabs[4]:
                st.subheader("Data Filtering")
                
                # Search functionality
                search_col = st.selectbox("Select column to search", df.columns)
                search_term = st.text_input("Enter search term")
                
                # Numeric range filter
                if df[search_col].dtype in ['int64', 'float64']:
                    min_val, max_val = st.slider(
                        f"Filter range for {search_col}",
                        float(df[search_col].min()),
                        float(df[search_col].max()),
                        (float(df[search_col].min()), float(df[search_col].max()))
                    )
                    filtered_df = df[
                        (df[search_col] >= min_val) & 
                        (df[search_col] <= max_val)
                    ]
                else:
                    if search_term:
                        filtered_df = df[df[search_col].astype(str).str.contains(search_term, case=False)]
                    else:
                        filtered_df = df
                
                st.write(f"Showing {len(filtered_df)} rows")
                st.dataframe(filtered_df)

            with tabs[5]:
                st.subheader("Export Data")
                
                # Export format selection
                export_format = st.selectbox(
                    "Select export format",
                    ["CSV", "Excel", "JSON"]
                )
                
                # Export options
                if st.checkbox("Export filtered data only"):
                    export_df = filtered_df
                else:
                    export_df = df
                
                if st.button(f"Export as {export_format}"):
                    buffer, mime_type, file_ext = create_download_link(
                        export_df,
                        export_format.lower()
                    )
                    
                    st.download_button(
                        label=f"Download {export_format} file",
                        data=buffer,
                        file_name=f"processed_data{file_ext}",
                        mime=mime_type
                    )
        
        st.success(texts['success'])  # Display success message when all files are processed

if __name__ == "__main__":
    main()
