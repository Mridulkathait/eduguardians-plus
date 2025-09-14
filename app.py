import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import joblib
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
import tempfile
from datetime import datetime
import base64
from io import BytesIO

# Set page config
st.set_page_config(
    page_title="EduGuardians+",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
def local_css():
    st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .card {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        .risk-low {
            color: #4CAF50;
            font-weight: bold;
        }
        .risk-medium {
            color: #FFC107;
            font-weight: bold;
        }
        .risk-high {
            color: #F44336;
            font-weight: bold;
        }
        .stButton button {
            background-color: #0b67a4;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            font-weight: bold;
            transition: background-color 0.3s;
        }
        .stButton button:hover {
            background-color: #095587;
        }
        .stDownloadButton button {
            background-color: #ffb400;
            color: black;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            font-weight: bold;
            transition: background-color 0.3s;
        }
        .stDownloadButton button:hover {
            background-color: #e6a200;
        }
        .css-1d391kg {
            background-color: #f8f9fa;
        }
        .st-at {
            background-color: #0b67a4;
            color: white;
        }
        div[data-testid="stMetricValue"] {
            background-color: #f8f9fa;
            border-radius: 5px;
            padding: 10px;
        }
        .stDataFrame {
            border-radius: 5px;
            overflow: hidden;
        }
        .stSlider {
            color: #0b67a4;
        }
    </style>
    """, unsafe_allow_html=True)

# Apply custom CSS
local_css()

# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = ['attendance_pct', 'avg_score', 'attempts', 'fees_pending']
if 'db_connection' not in st.session_state:
    st.session_state.db_connection = None
if 'db_mode' not in st.session_state:
    st.session_state.db_mode = False

# Load the pre-trained model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('rf_model.joblib')
        return model
    except:
        return None

# Function to create database connection
def create_db_connection():
    conn = sqlite3.connect('edu_guardians.db')
    return conn

# Function to create tables if they don't exist
def create_tables(conn):
    cursor = conn.cursor()
    
    # Create students table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS students (
        student_id TEXT PRIMARY KEY,
        name TEXT,
        attendance_pct REAL,
        avg_score REAL,
        attempts INTEGER,
        fees_pending REAL,
        dropout_probability REAL,
        risk_label TEXT,
        last_updated TEXT
    )
    ''')
    
    conn.commit()

# Function to insert or update student data
def upsert_student_data(conn, student_data):
    cursor = conn.cursor()
    
    for _, row in student_data.iterrows():
        cursor.execute('''
        INSERT OR REPLACE INTO students 
        (student_id, name, attendance_pct, avg_score, attempts, fees_pending, dropout_probability, risk_label, last_updated)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            row['student_id'],
            row['name'],
            row['attendance_pct'],
            row['avg_score'],
            row['attempts'],
            row['fees_pending'],
            row.get('dropout_probability', 0),
            row.get('risk_label', 'Low'),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
    
    conn.commit()

# Function to fetch all student data from database
def fetch_all_students(conn):
    query = "SELECT * FROM students"
    return pd.read_sql_query(query, conn)

# Function to merge datasets
def merge_datasets(attendance_df, assessments_df, fees_df):
    # Merge attendance and assessments
    merged_df = pd.merge(attendance_df, assessments_df, on='student_id', how='outer')
    
    # Merge with fees
    merged_df = pd.merge(merged_df, fees_df, on='student_id', how='outer')
    
    # Fill missing values
    merged_df['attendance_pct'] = merged_df['attendance_pct'].fillna(0)
    merged_df['avg_score'] = merged_df['avg_score'].fillna(0)
    merged_df['attempts'] = merged_df['attempts'].fillna(0)
    merged_df['fees_pending'] = merged_df['fees_pending'].fillna(0)
    
    # Fill missing names
    if 'name' not in merged_df.columns:
        merged_df['name'] = merged_df['student_id'].apply(lambda x: f"Student {x[3:]}")
    else:
        merged_df['name'] = merged_df['name'].fillna(merged_df['student_id'].apply(lambda x: f"Student {x[3:]}"))
    
    return merged_df

# Function to predict dropout risk
def predict_dropout_risk(model, data):
    if model is None:
        return None
    
    # Extract features
    X = data[st.session_state.feature_names]
    
    # Make predictions
    probabilities = model.predict_proba(X)[:, 1] * 100  # Convert to percentage
    
    # Add predictions to dataframe
    data['dropout_probability'] = probabilities
    
    # Assign risk labels
    conditions = [
        (data['dropout_probability'] < 40),
        (data['dropout_probability'] >= 40) & (data['dropout_probability'] < 70),
        (data['dropout_probability'] >= 70)
    ]
    choices = ['Low', 'Medium', 'High']
    data['risk_label'] = np.select(conditions, choices)
    
    return data

# Function to get feature importance
def get_feature_importance(model):
    if model is None:
        return None
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create a dataframe for visualization
    feature_importance_df = pd.DataFrame({
        'feature': st.session_state.feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return feature_importance_df

# Function to generate PDF report for a student
def generate_student_pdf(student_data, feature_importance):
    pdf = FPDF()
    pdf.add_page()
    
    # Add title
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'EduGuardians+ Student Risk Report', ln=True, align='C')
    pdf.ln(10)
    
    # Add student details
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, f'Student ID: {student_data["student_id"]}', ln=True)
    pdf.cell(0, 10, f'Name: {student_data["name"]}', ln=True)
    pdf.ln(5)
    
    # Add risk information
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Risk Assessment:', ln=True)
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f'Dropout Probability: {student_data["dropout_probability"]:.2f}%', ln=True)
    pdf.cell(0, 10, f'Risk Category: {student_data["risk_label"]}', ln=True)
    pdf.ln(5)
    
    # Add performance metrics
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Performance Metrics:', ln=True)
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f'Attendance: {student_data["attendance_pct"]:.2f}%', ln=True)
    pdf.cell(0, 10, f'Average Score: {student_data["avg_score"]:.2f}', ln=True)
    pdf.cell(0, 10, f'Assessment Attempts: {student_data["attempts"]}', ln=True)
    pdf.cell(0, 10, f'Fees Pending: ${student_data["fees_pending"]:.2f}', ln=True)
    pdf.ln(5)
    
    # Add contributing factors
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Key Contributing Factors:', ln=True)
    pdf.set_font('Arial', '', 12)
    
    # Sort features by importance for this student
    features = st.session_state.feature_names
    student_features = [student_data[feature] for feature in features]
    
    # Calculate contribution (simplified for demo)
    contributions = []
    for i, feature in enumerate(features):
        normalized_value = student_features[i] / 100 if feature in ['attendance_pct', 'avg_score'] else student_features[i]
        importance = feature_importance[feature_importance['feature'] == feature]['importance'].values[0]
        contribution = normalized_value * importance
        contributions.append((feature, contribution))
    
    # Sort by contribution
    contributions.sort(key=lambda x: x[1], reverse=True)
    
    # Display top 3 contributing factors
    for feature, contribution in contributions[:3]:
        pdf.cell(0, 10, f'- {feature.replace("_", " ").title()}: High Impact', ln=True)
    
    # Add recommendations
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Recommendations:', ln=True)
    pdf.set_font('Arial', '', 12)
    
    if student_data["risk_label"] == "Low":
        pdf.cell(0, 10, '- Student is performing well. Continue monitoring.', ln=True)
    elif student_data["risk_label"] == "Medium":
        if student_data["attendance_pct"] < 75:
            pdf.cell(0, 10, '- Improve attendance by attending all classes.', ln=True)
        if student_data["avg_score"] < 60:
            pdf.cell(0, 10, '- Focus on academic performance through tutoring.', ln=True)
        if student_data["fees_pending"] > 0:
            pdf.cell(0, 10, '- Clear pending fees to avoid financial stress.', ln=True)
    else:  # High risk
        pdf.cell(0, 10, '- Immediate intervention required.', ln=True)
        pdf.cell(0, 10, '- Schedule counseling session with student.', ln=True)
        pdf.cell(0, 10, '- Develop personalized improvement plan.', ln=True)
        if student_data["attendance_pct"] < 60:
            pdf.cell(0, 10, '- Address attendance issues urgently.', ln=True)
        if student_data["avg_score"] < 50:
            pdf.cell(0, 10, '- Provide additional academic support.', ln=True)
        if student_data["fees_pending"] > 500:
            pdf.cell(0, 10, '- Discuss financial aid options.', ln=True)
    
    # Add footer
    pdf.ln(10)
    pdf.set_font('Arial', 'I', 10)
    pdf.cell(0, 10, f'Report generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', ln=True, align='C')
    
    return pdf

# Function to create download link for PDF
def create_download_link(pdf, filename):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        pdf.output(tmp_file.name, 'F')
        
        with open(tmp_file.name, 'rb') as f:
            pdf_data = f.read()
        
        b64 = base64.b64encode(pdf_data).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">Download PDF Report</a>'
        return href

# Function to create sample data
def create_sample_data():
    # Create sample attendance data
    attendance_data = {
        'student_id': [f'STU{i:04d}' for i in range(1, 101)],
        'name': [f'Student {i}' for i in range(1, 101)],
        'attendance_pct': np.random.normal(75, 15, 100).clip(0, 100)
    }
    attendance_df = pd.DataFrame(attendance_data)
    
    # Create sample assessments data
    assessments_data = {
        'student_id': [f'STU{i:04d}' for i in range(1, 101)],
        'avg_score': np.random.normal(65, 20, 100).clip(0, 100),
        'attempts': np.random.randint(1, 5, 100)
    }
    assessments_df = pd.DataFrame(assessments_data)
    
    # Create sample fees data
    fees_data = {
        'student_id': [f'STU{i:04d}' for i in range(1, 101)],
        'fees_pending': np.random.exponential(300, 100).clip(0, 2000)
    }
    fees_df = pd.DataFrame(fees_data)
    
    return attendance_df, assessments_df, fees_df

# Function to create sample CSV files
def create_sample_csv_files():
    # Create sample data
    attendance_df, assessments_df, fees_df = create_sample_data()
    
    # Create directory if it doesn't exist
    os.makedirs('sample_data', exist_ok=True)
    
    # Save to CSV
    attendance_df.to_csv('sample_data/attendance.csv', index=False)
    assessments_df.to_csv('sample_data/assessments.csv', index=False)
    fees_df.to_csv('sample_data/fees.csv', index=False)
    
    return attendance_df, assessments_df, fees_df

# Sidebar
def sidebar():
    st.sidebar.title("🎓 EduGuardians+")
    st.sidebar.markdown("AI-powered Dropout Prediction & Counseling System")
    
    # Load model
    model = load_model()
    if model is None:
        st.sidebar.error("Model not found. Please train the model first.")
        st.sidebar.code("python generate_model.py")
        return
    
    st.session_state.model = model
    
    # Data source selection
    st.sidebar.subheader("Data Source")
    data_source = st.sidebar.radio("Select data source:", ["Upload CSV Files", "Use Sample Data", "Database Mode"])
    
    if data_source == "Upload CSV Files":
        st.session_state.db_mode = False
        
        # File uploaders
        st.sidebar.subheader("Upload Data")
        attendance_file = st.sidebar.file_uploader("Upload Attendance CSV", type=["csv"])
        assessments_file = st.sidebar.file_uploader("Upload Assessments CSV", type=["csv"])
        fees_file = st.sidebar.file_uploader("Upload Fees CSV", type=["csv"])
        
        if attendance_file and assessments_file and fees_file:
            try:
                attendance_df = pd.read_csv(attendance_file)
                assessments_df = pd.read_csv(assessments_file)
                fees_df = pd.read_csv(fees_file)
                
                # Merge datasets
                merged_df = merge_datasets(attendance_df, assessments_df, fees_df)
                
                # Predict dropout risk
                merged_df = predict_dropout_risk(model, merged_df)
                
                st.session_state.data = merged_df
                st.sidebar.success("Data uploaded and processed successfully!")
                
            except Exception as e:
                st.sidebar.error(f"Error processing files: {str(e)}")
    
    elif data_source == "Use Sample Data":
        st.session_state.db_mode = False
        
        if st.sidebar.button("Generate Sample Data"):
            try:
                attendance_df, assessments_df, fees_df = create_sample_data()
                
                # Merge datasets
                merged_df = merge_datasets(attendance_df, assessments_df, fees_df)
                
                # Predict dropout risk
                merged_df = predict_dropout_risk(model, merged_df)
                
                st.session_state.data = merged_df
                st.sidebar.success("Sample data generated successfully!")
            except Exception as e:
                st.sidebar.error(f"Error generating sample data: {str(e)}")
    
    elif data_source == "Database Mode":
        st.session_state.db_mode = True
        
        # Create database connection if it doesn't exist
        if st.session_state.db_connection is None:
            try:
                conn = create_db_connection()
                create_tables(conn)
                st.session_state.db_connection = conn
                st.sidebar.success("Connected to database successfully!")
            except Exception as e:
                st.sidebar.error(f"Database connection error: {str(e)}")
                return
        
        # Database operations
        db_operation = st.sidebar.selectbox("Database Operation:", ["View Data", "Upload to Database", "Clear Database"])
        
        if db_operation == "View Data":
            if st.sidebar.button("Load Data from Database"):
                try:
                    data = fetch_all_students(st.session_state.db_connection)
                    if not data.empty:
                        st.session_state.data = data
                        st.sidebar.success("Data loaded from database!")
                    else:
                        st.sidebar.warning("No data found in database.")
                except Exception as e:
                    st.sidebar.error(f"Error loading data: {str(e)}")
        
        elif db_operation == "Upload to Database":
            if st.session_state.data is not None:
                if st.sidebar.button("Save Current Data to Database"):
                    try:
                        upsert_student_data(st.session_state.db_connection, st.session_state.data)
                        st.sidebar.success("Data saved to database!")
                    except Exception as e:
                        st.sidebar.error(f"Error saving data: {str(e)}")
            else:
                st.sidebar.warning("No data to save. Please upload or generate data first.")
        
        elif db_operation == "Clear Database":
            if st.sidebar.button("Clear All Data", key="clear_db"):
                try:
                    cursor = st.session_state.db_connection.cursor()
                    cursor.execute("DELETE FROM students")
                    st.session_state.db_connection.commit()
                    st.session_state.data = None
                    st.sidebar.success("Database cleared!")
                except Exception as e:
                    st.sidebar.error(f"Error clearing database: {str(e)}")
    
    # Risk thresholds
    st.sidebar.subheader("Risk Thresholds")
    medium_threshold = st.sidebar.slider("Medium Risk Threshold (%)", 30, 60, 40, 5)
    high_threshold = st.sidebar.slider("High Risk Threshold (%)", 60, 90, 70, 5)
    
    # About section
    st.sidebar.subheader("About")
    st.sidebar.info(
        "EduGuardians+ is an AI-powered system designed to identify at-risk students "
        "and provide actionable insights for educational institutions."
    )

# Main app
def main():
    # Render sidebar
    sidebar()
    
    # Main content
    st.title("🎓 EduGuardians+ Dashboard")
    
    # Check if data is available
    if st.session_state.data is None:
        st.info("Please upload data or use sample data to get started.")
        
        # Display sample data format
        st.subheader("Sample Data Format")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Attendance CSV**")
            st.code("""
student_id,name,attendance_pct
STU0001,Student 1,85.5
STU0002,Student 2,72.3
            """)
        
        with col2:
            st.markdown("**Assessments CSV**")
            st.code("""
student_id,avg_score,attempts
STU0001,78.5,3
STU0002,65.3,2
            """)
        
        with col3:
            st.markdown("**Fees CSV**")
            st.code("""
student_id,fees_pending
STU0001,150.0
STU0002,450.5
            """)
        
        # Create sample CSV files button
        if st.button("Create Sample CSV Files"):
            try:
                create_sample_csv_files()
                st.success("Sample CSV files created in 'sample_data' directory!")
            except Exception as e:
                st.error(f"Error creating sample files: {str(e)}")
        
        return
    
    data = st.session_state.data
    
    # Get feature importance
    feature_importance = get_feature_importance(st.session_state.model)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔍 Student Search", "🎮 What-If Simulator", "📋 Reports"])
    
    with tab1:
        st.header("Student Risk Dashboard")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Students", len(data))
        
        with col2:
            low_risk = len(data[data['risk_label'] == 'Low'])
            st.metric("Low Risk", low_risk, f"{low_risk/len(data)*100:.1f}%")
        
        with col3:
            medium_risk = len(data[data['risk_label'] == 'Medium'])
            st.metric("Medium Risk", medium_risk, f"{medium_risk/len(data)*100:.1f}%")
        
        with col4:
            high_risk = len(data[data['risk_label'] == 'High'])
            st.metric("High Risk", high_risk, f"{high_risk/len(data)*100:.1f}%")
        
        # Risk distribution chart
        st.subheader("Risk Distribution")
        
        # Create pie chart
        risk_counts = data['risk_label'].value_counts()
        fig_pie = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            color=risk_counts.index,
            color_discrete_map={'Low': '#4CAF50', 'Medium': '#FFC107', 'High': '#F44336'},
            hole=0.4
        )
        fig_pie.update_layout(
            title_text="Student Risk Distribution",
            showlegend=True,
            height=400
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Feature importance
        st.subheader("Feature Importance")
        
        if feature_importance is not None:
            fig_bar = px.bar(
                feature_importance,
                x='importance',
                y='feature',
                orientation='h',
                color='importance',
                color_continuous_scale='Blues'
            )
            fig_bar.update_layout(
                title_text="Global Feature Importance",
                xaxis_title="Importance",
                yaxis_title="Feature",
                height=400
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Student table
        st.subheader("Student Data")
        
        # Add filter for risk level
        risk_filter = st.selectbox("Filter by Risk Level:", ["All", "Low", "Medium", "High"])
        
        if risk_filter != "All":
            filtered_data = data[data['risk_label'] == risk_filter]
        else:
            filtered_data = data
        
        # Display the table with color coding
        st.dataframe(
            filtered_data.style.applymap(
                lambda x: 'background-color: #4CAF50' if x == 'Low' else 
                         'background-color: #FFC107' if x == 'Medium' else 
                         'background-color: #F44336' if x == 'High' else '',
                subset=['risk_label']
            ),
            use_container_width=True,
            height=400
        )
    
    with tab2:
        st.header("Student Search")
        
        # Search input
        search_term = st.text_input("Search by Student ID or Name:")
        
        if search_term:
            # Filter data based on search term
            search_results = data[
                data['student_id'].str.contains(search_term, case=False) | 
                data['name'].str.contains(search_term, case=False)
            ]
            
            if len(search_results) > 0:
                # Display search results
                st.write(f"Found {len(search_results)} matching students:")
                
                for _, student in search_results.iterrows():
                    with st.expander(f"{student['name']} ({student['student_id']})"):
                        # Create two columns
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Student Information")
                            st.write(f"**Student ID:** {student['student_id']}")
                            st.write(f"**Name:** {student['name']}")
                            
                            st.subheader("Performance Metrics")
                            st.write(f"**Attendance:** {student['attendance_pct']:.2f}%")
                            st.write(f"**Average Score:** {student['avg_score']:.2f}")
                            st.write(f"**Assessment Attempts:** {student['attempts']}")
                            st.write(f"**Fees Pending:** ${student['fees_pending']:.2f}")
                        
                        with col2:
                            st.subheader("Risk Assessment")
                            st.write(f"**Dropout Probability:** {student['dropout_probability']:.2f}%")
                            
                            # Display risk label with color
                            if student['risk_label'] == 'Low':
                                st.markdown(f"**Risk Level:** <span style='color:green; font-weight:bold'>{student['risk_label']}</span>", unsafe_allow_html=True)
                            elif student['risk_label'] == 'Medium':
                                st.markdown(f"**Risk Level:** <span style='color:orange; font-weight:bold'>{student['risk_label']}</span>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"**Risk Level:** <span style='color:red; font-weight:bold'>{student['risk_label']}</span>", unsafe_allow_html=True)
                            
                            # Top contributing factors
                            st.subheader("Top Contributing Factors")
                            
                            # Calculate feature contributions for this student
                            features = st.session_state.feature_names
                            student_features = [student[feature] for feature in features]
                            
                            # Create a simple contribution score
                            contributions = []
                            for i, feature in enumerate(features):
                                normalized_value = student_features[i] / 100 if feature in ['attendance_pct', 'avg_score'] else student_features[i] / 10
                                importance = feature_importance[feature_importance['feature'] == feature]['importance'].values[0]
                                contribution = normalized_value * importance
                                contributions.append((feature, contribution))
                            
                            # Sort by contribution
                            contributions.sort(key=lambda x: x[1], reverse=True)
                            
                            # Display top 3
                            for feature, contribution in contributions[:3]:
                                feature_name = feature.replace("_", " ").title()
                                st.write(f"- {feature_name}")
                        
                        # Download buttons
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # CSV download
                            csv = student.to_frame().T.to_csv(index=False)
                            st.download_button(
                                label="Download CSV Report",
                                data=csv,
                                file_name=f"{student['student_id']}_report.csv",
                                mime='text/csv'
                            )
                        
                        with col2:
                            # PDF download
                            pdf = generate_student_pdf(student, feature_importance)
                            st.markdown(
                                create_download_link(pdf, f"{student['student_id']}_report.pdf"),
                                unsafe_allow_html=True
                            )
            else:
                st.warning("No students found matching your search.")
    
    with tab3:
        st.header("What-If Simulator")
        
        # Select a student
        student_id = st.selectbox("Select a student:", data['student_id'].unique())
        
        if student_id:
            # Get student data
            student = data[data['student_id'] == student_id].iloc[0]
            
            # Display current values
            st.subheader("Current Values")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Attendance", f"{student['attendance_pct']:.2f}%")
            
            with col2:
                st.metric("Average Score", f"{student['avg_score']:.2f}")
            
            with col3:
                st.metric("Attempts", student['attempts'])
            
            with col4:
                st.metric("Fees Pending", f"${student['fees_pending']:.2f}")
            
            # Display current risk
            if student['risk_label'] == 'Low':
                st.markdown(f"**Current Risk Level:** <span style='color:green; font-weight:bold'>{student['risk_label']} ({student['dropout_probability']:.2f}%)</span>", unsafe_allow_html=True)
            elif student['risk_label'] == 'Medium':
                st.markdown(f"**Current Risk Level:** <span style='color:orange; font-weight:bold'>{student['risk_label']} ({student['dropout_probability']:.2f}%)</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"**Current Risk Level:** <span style='color:red; font-weight:bold'>{student['risk_label']} ({student['dropout_probability']:.2f}%)</span>", unsafe_allow_html=True)
            
            # Sliders for what-if analysis
            st.subheader("Simulate Changes")
            
            col1, col2 = st.columns(2)
            
            with col1:
                new_attendance = st.slider("Attendance (%)", 0, 100, int(student['attendance_pct']))
                new_score = st.slider("Average Score", 0, 100, int(student['avg_score']))
            
            with col2:
                new_attempts = st.slider("Assessment Attempts", 1, 10, int(student['attempts']))
                new_fees = st.slider("Fees Pending ($)", 0, 2000, int(student['fees_pending']))
            
            # Create a new student record with simulated values
            simulated_student = student.copy()
            simulated_student['attendance_pct'] = new_attendance
            simulated_student['avg_score'] = new_score
            simulated_student['attempts'] = new_attempts
            simulated_student['fees_pending'] = new_fees
            
            # Predict risk with simulated values
            simulated_df = pd.DataFrame([simulated_student])
            simulated_df = predict_dropout_risk(st.session_state.model, simulated_df)
            simulated_student = simulated_df.iloc[0]
            
            # Display simulated risk
            st.subheader("Simulated Risk")
            
            if simulated_student['risk_label'] == 'Low':
                st.markdown(f"**Simulated Risk Level:** <span style='color:green; font-weight:bold'>{simulated_student['risk_label']} ({simulated_student['dropout_probability']:.2f}%)</span>", unsafe_allow_html=True)
            elif simulated_student['risk_label'] == 'Medium':
                st.markdown(f"**Simulated Risk Level:** <span style='color:orange; font-weight:bold'>{simulated_student['risk_label']} ({simulated_student['dropout_probability']:.2f}%)</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"**Simulated Risk Level:** <span style='color:red; font-weight:bold'>{simulated_student['risk_label']} ({simulated_student['dropout_probability']:.2f}%)</span>", unsafe_allow_html=True)
            
            # Compare current and simulated risk
            st.subheader("Comparison")
            
            # Create a comparison chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Current',
                x=['Risk Probability'],
                y=[student['dropout_probability']],
                marker_color='#0b67a4'
            ))
            
            fig.add_trace(go.Bar(
                name='Simulated',
                x=['Risk Probability'],
                y=[simulated_student['dropout_probability']],
                marker_color='#ffb400'
            ))
            
            fig.update_layout(
                title='Current vs Simulated Risk',
                xaxis_title='Metric',
                yaxis_title='Dropout Probability (%)',
                barmode='group'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Generate recommendations
            st.subheader("Recommendations")
            
            risk_change = simulated_student['dropout_probability'] - student['dropout_probability']
            
            if risk_change < -10:
                st.success(f"Great! The simulated changes would reduce the dropout risk by {abs(risk_change):.2f} percentage points.")
            elif risk_change > 10:
                st.error(f"Warning! The simulated changes would increase the dropout risk by {risk_change:.2f} percentage points.")
            else:
                st.info("The simulated changes have a minimal impact on the dropout risk.")
            
            # Specific recommendations based on changes
            recommendations = []
            
            if new_attendance > student['attendance_pct'] + 10:
                recommendations.append(f"Improving attendance by {new_attendance - student['attendance_pct']:.2f}% points has a positive impact.")
            elif new_attendance < student['attendance_pct'] - 10:
                recommendations.append(f"Decreasing attendance by {student['attendance_pct'] - new_attendance:.2f}% points has a negative impact.")
            
            if new_score > student['avg_score'] + 10:
                recommendations.append(f"Improving average score by {new_score - student['avg_score']:.2f} points has a positive impact.")
            elif new_score < student['avg_score'] - 10:
                recommendations.append(f"Decreasing average score by {student['avg_score'] - new_score:.2f} points has a negative impact.")
            
            if new_fees < student['fees_pending'] - 100:
                recommendations.append(f"Reducing pending fees by ${student['fees_pending'] - new_fees:.2f} has a positive impact.")
            elif new_fees > student['fees_pending'] + 100:
                recommendations.append(f"Increasing pending fees by ${new_fees - student['fees_pending']:.2f} has a negative impact.")
            
            if recommendations:
                for rec in recommendations:
                    st.write(f"- {rec}")
            else:
                st.write("The changes made are not significant enough to generate specific recommendations.")
    
    with tab4:
        st.header("Reports")
        
        # Bulk CSV download
        st.subheader("Bulk Student Data Report")
        
        if st.button("Generate CSV Report"):
            csv = data.to_csv(index=False)
            st.download_button(
                label="Download CSV Report",
                data=csv,
                file_name="student_risk_report.csv",
                mime='text/csv'
            )
        
        # Individual student reports
        st.subheader("Individual Student Reports")
        
        # Select a student
        report_student_id = st.selectbox("Select a student for individual report:", data['student_id'].unique())
        
        if report_student_id:
            # Get student data
            report_student = data[data['student_id'] == report_student_id].iloc[0]
            
            # Display student info
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Student ID:** {report_student['student_id']}")
                st.write(f"**Name:** {report_student['name']}")
                st.write(f"**Risk Level:** {report_student['risk_label']} ({report_student['dropout_probability']:.2f}%)")
            
            with col2:
                # Download buttons
                # CSV download
                csv = report_student.to_frame().T.to_csv(index=False)
                st.download_button(
                    label="Download CSV Report",
                    data=csv,
                    file_name=f"{report_student['student_id']}_report.csv",
                    mime='text/csv'
                )
                
                # PDF download
                pdf = generate_student_pdf(report_student, feature_importance)
                st.markdown(
                    create_download_link(pdf, f"{report_student['student_id']}_report.pdf"),
                    unsafe_allow_html=True
                )

# Run the app
if __name__ == "__main__":
    main()
