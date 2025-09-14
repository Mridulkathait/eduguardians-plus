# EduGuardians+ - Streamlit App (Improved Prototype)
# Run: streamlit run app.py
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
from fpdf import FPDF
from datetime import datetime
import tempfile, base64
from io import BytesIO

st.set_page_config(page_title="EduGuardians+", page_icon="🎓", layout="wide")

# ---------- Utility / Styling ----------
def local_css():
    st.markdown(
        """
    <style>
    .card {background-color: white;border-radius:10px;padding:16px;box-shadow:0 4px 8px rgba(0,0,0,0.08);margin-bottom:16px;}
    .risk-low{color:#1b9e77;font-weight:bold;}
    .risk-med{color:#f39c12;font-weight:bold;}
    .risk-high{color:#d9534f;font-weight:bold;}
    .stButton>button {background-color:#0b67a4;color:white;border-radius:6px;}
    .stDownloadButton>button {background-color:#ffb400;color:black;border-radius:6px;}
    </style>
    """, unsafe_allow_html=True)

local_css()

# ---------- Paths ----------
MODEL_PATH = "rf_model.joblib"
DB_PATH = "edu_guardians.db"
SAMPLE_DIR = "sample_data"

# ---------- Helpers ----------
@st.cache_resource
def load_model(path=MODEL_PATH):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            st.error("Error loading model: " + str(e))
            return None
    return None

def save_model(model, path=MODEL_PATH):
    joblib.dump(model, path)

def ensure_sample_files():
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    # If already exists, skip
    att = os.path.join(SAMPLE_DIR, "attendance.csv")
    if not os.path.exists(att):
        # create small sample
        n=50
        ids=[f"STU{i:04d}" for i in range(1,n+1)]
        attendance = pd.DataFrame({"student_id":ids, "name":[f"Student {i}" for i in range(1,n+1)],
                                   "attendance_pct": np.random.normal(75,12,n).clip(0,100)})
        assessments = pd.DataFrame({"student_id":ids, "avg_score": np.random.normal(65,18,n).clip(0,100),
                                    "attempts": np.random.randint(1,4,n)})
        fees = pd.DataFrame({"student_id":ids, "fees_pending": np.random.exponential(200,n).clip(0,1500)})
        attendance.to_csv(att, index=False)
        assessments.to_csv(os.path.join(SAMPLE_DIR,"assessments.csv"), index=False)
        fees.to_csv(os.path.join(SAMPLE_DIR,"fees.csv"), index=False)

def merge_datasets(att, ass, fees):
    # Accept either pre-aggregated attendance or daily rows containing 'date'/'present'
    if 'attendance_pct' not in att.columns and {'date','present'}.issubset(att.columns):
        # compute percent present per student (present as 1/0)
        att_proc = att.groupby('student_id')['present'].mean().reset_index().rename(columns={'present':'attendance_pct'})
        att = att_proc
    merged = pd.merge(att, ass, on='student_id', how='outer')
    merged = pd.merge(merged, fees, on='student_id', how='outer')
    # Fill defaults
    merged['attendance_pct'] = merged.get('attendance_pct', pd.Series(dtype=float)).fillna(0)
    merged['avg_score'] = merged.get('avg_score', pd.Series(dtype=float)).fillna(0)
    merged['attempts'] = merged.get('attempts', pd.Series(dtype=int)).fillna(0).astype(int)
    merged['fees_pending'] = merged.get('fees_pending', pd.Series(dtype=float)).fillna(0)
    # name fallback
    if 'name' not in merged.columns:
        merged['name'] = merged['student_id']
    else:
        merged['name'] = merged['name'].fillna(merged['student_id'])
    return merged

def rule_score_row(row):
    # Map to 0-100 per earlier spec
    # Attendance factor
    att = row['attendance_pct']
    if att < 50:
        att_r = 100
    elif att < 70:
        att_r = 60
    else:
        att_r = 20
    # Score factor
    s = row['avg_score']
    if s < 40:
        s_r = 100
    elif s < 60:
        s_r = 60
    else:
        s_r = 20
    # Attempts
    a = row['attempts']
    if a == 0:
        attm = 0
    elif a == 1:
        attm = 20
    elif a == 2:
        attm = 40
    else:
        attm = 100
    # Fees
    fees = row['fees_pending']
    fees_r = 100 if fees>0 else 0
    # weights default
    w_att, w_score, w_attempts, w_fees = 0.3, 0.4, 0.15, 0.15
    score = att_r*w_att + s_r*w_score + attm*w_attempts + fees_r*w_fees
    return score

def compute_rule_scores(df):
    df = df.copy()
    df['rule_risk_score'] = df.apply(rule_score_row, axis=1)
    df['rule_risk_label'] = pd.cut(df['rule_risk_score'], bins=[-1,39,69,100], labels=['Low','Medium','High'])
    return df

def train_random_forest(df):
    # create synthetic target from rule for demo if no label
    df2 = df.copy()
    if 'target_dropout' not in df2.columns:
        df2['target_dropout'] = (df2['rule_risk_score']>=70).astype(int)
    features = ['attendance_pct','avg_score','attempts','fees_pending']
    X = df2[features].fillna(0)
    y = df2['target_dropout']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y if y.nunique()>1 else None)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test,y_pred,output_dict=True) if y_test.nunique()>1 else {"note":"single-class in test"}
    cm = confusion_matrix(y_test,y_pred) if y_test.nunique()>1 else None
    save_model(model)
    return model, report, cm

def predict_with_model(model, df):
    if model is None:
        df['dropout_probability'] = np.nan
        df['risk_label'] = 'Unknown'
        return df
    features = ['attendance_pct','avg_score','attempts','fees_pending']
    X = df[features].fillna(0)
    probs = model.predict_proba(X)[:,1]*100
    df['dropout_probability'] = probs
    # combine rule and ml: default ml weight 0.6
    if 'rule_risk_score' not in df.columns:
        df = compute_rule_scores(df)
    ml_w = st.session_state.get('ml_weight', 0.6)
    df['final_risk_score'] = ml_w*df['dropout_probability'] + (1-ml_w)*df['rule_risk_score']
    df['risk_label'] = pd.cut(df['final_risk_score'], bins=[-1,39,69,100], labels=['Low','Medium','High'])
    return df

def generate_pdf_bytes(student_row, feature_importance_df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial","B",16)
    pdf.cell(0,10,"EduGuardians+ Student Report",ln=True,align="C")
    pdf.ln(6)
    pdf.set_font("Arial","",12)
    for k in ['student_id','name','attendance_pct','avg_score','attempts','fees_pending','dropout_probability','final_risk_score','risk_label']:
        if k in student_row.index:
            pdf.cell(0,8,f"{k.replace('_',' ').title()}: {student_row[k]}", ln=True)
    pdf.ln(6)
    pdf.set_font("Arial","B",12)
    pdf.cell(0,8,"Recommendations:", ln=True)
    pdf.set_font("Arial","",11)
    rl = student_row.get('risk_label','Low')
    if rl=='High':
        pdf.multi_cell(0,7,"Immediate intervention: schedule counseling, remedial classes, check fees and provide financial aid if needed.")
    elif rl=='Medium':
        pdf.multi_cell(0,7,"Monitor student and consider remedial support or attendance improvement plan.")
    else:
        pdf.multi_cell(0,7,"Student is low risk. Keep monitoring.")
    # return bytes
    return pdf.output(dest='S').encode('latin1')

# ---------- UI ----------

st.sidebar.title("EduGuardians+")
st.sidebar.write("AI Dropout Prediction & Counseling — Prototype")

mode = st.sidebar.selectbox("Mode", ["Quick Upload", "Use Sample Data", "Database Mode"])
st.session_state['ml_weight'] = st.sidebar.slider("ML Weight in final score", 0.0, 1.0, 0.6, 0.05)

# File upload or sample
if mode=="Quick Upload":
    att_file = st.sidebar.file_uploader("Attendance CSV", type=['csv'])
    ass_file = st.sidebar.file_uploader("Assessments CSV", type=['csv'])
    fees_file = st.sidebar.file_uploader("Fees CSV", type=['csv'])
    if st.sidebar.button("Load & Merge") and att_file and ass_file and fees_file:
        try:
            att = pd.read_csv(att_file)
            ass = pd.read_csv(ass_file)
            fees = pd.read_csv(fees_file)
            merged = merge_datasets(att, ass, fees)
            merged = compute_rule_scores(merged)
            # load model and predict
            model = load_model()
            merged = predict_with_model(model, merged)
            st.session_state['data'] = merged
            st.success("Data loaded and processed.")
        except Exception as e:
            st.error("Error parsing files: "+str(e))
elif mode=="Use Sample Data":
    ensure_sample_files()
    if st.sidebar.button("Load Sample Data"):
        att = pd.read_csv(os.path.join(SAMPLE_DIR,"attendance.csv"))
        ass = pd.read_csv(os.path.join(SAMPLE_DIR,"assessments.csv"))
        fees = pd.read_csv(os.path.join(SAMPLE_DIR,"fees.csv"))
        merged = merge_datasets(att, ass, fees)
        merged = compute_rule_scores(merged)
        model = load_model()
        merged = predict_with_model(model, merged)
        st.session_state['data'] = merged
        st.success("Sample data loaded.")
elif mode=="Database Mode":
    conn = sqlite3.connect(DB_PATH)
    ensure = st.sidebar.button("Init DB with sample data")
    if ensure:
        att = pd.read_csv(os.path.join(SAMPLE_DIR,"attendance.csv")) if os.path.exists(os.path.join(SAMPLE_DIR,"attendance.csv")) else None
        if att is None:
            ensure_sample_files()
            att = pd.read_csv(os.path.join(SAMPLE_DIR,"attendance.csv"))
        ass = pd.read_csv(os.path.join(SAMPLE_DIR,"assessments.csv"))
        fees = pd.read_csv(os.path.join(SAMPLE_DIR,"fees.csv"))
        merged = merge_datasets(att,ass,fees)
        merged = compute_rule_scores(merged)
        model = load_model()
        merged = predict_with_model(model, merged)
        merged.to_sql("students", conn, if_exists='replace', index=False)
        st.success("DB initialized with sample data.")
    if st.sidebar.button("Load DB to App"):
        try:
            df = pd.read_sql("SELECT * FROM students", conn)
            st.session_state['data'] = df
            st.success("Loaded DB data.")
        except Exception as e:
            st.error("DB load error: "+str(e))

# Train model option
st.sidebar.markdown("---")
if st.sidebar.button("Train RandomForest (on current data)"):
    if 'data' in st.session_state and st.session_state['data'] is not None:
        with st.spinner("Training model..."):
            model, report, cm = train_random_forest(st.session_state['data'])
            save_model(model)
            st.success("Model trained and saved to rf_model.joblib")
            st.json(report)
    else:
        st.sidebar.warning("Load data first to train model (Quick Upload or Sample).")

# Main layout
st.title("🎓 EduGuardians+ — Dashboard")
data = st.session_state.get('data', None)
model = load_model()

if data is None:
    st.info("No data loaded. Use the sidebar to upload CSVs or load sample data.")
else:
    st.header("Overview")
    cols = st.columns(4)
    cols[0].metric("Students", len(data))
    cols[1].metric("Low Risk", int((data['risk_label']=='Low').sum()))
    cols[2].metric("Medium Risk", int((data['risk_label']=='Medium').sum()))
    cols[3].metric("High Risk", int((data['risk_label']=='High').sum()))
    # charts
    fig = px.pie(data, names='risk_label', title="Risk Distribution", color_discrete_map={'Low':'#1b9e77','Medium':'#f39c12','High':'#d9534f'})
    st.plotly_chart(fig, use_container_width=True)
    # feature importance if model exists
    if model is not None:
        fi = pd.DataFrame({'feature':['attendance_pct','avg_score','attempts','fees_pending'],
                           'importance': model.feature_importances_})
        fi = fi.sort_values('importance', ascending=True)
        st.subheader("Feature Importance")
        st.bar_chart(fi.set_index('feature'))
    st.subheader("Student Table (first 200 rows)")
    st.dataframe(data.head(200), use_container_width=True)

    # Tabs for search, simulator, reports
    tab1,tab2,tab3 = st.tabs(["Search Student","What-If Simulator","Reports"])
    with tab1:
        q = st.text_input("Search by student_id or name")
        if q:
            filt = data[data['student_id'].str.contains(q, case=False) | data['name'].str.contains(q, case=False)]
            if filt.empty:
                st.warning("No student found")
            else:
                for _,row in filt.iterrows():
                    st.markdown(f"### {row['name']} — {row['student_id']}")
                    c1,c2 = st.columns(2)
                    with c1:
                        st.write("Attendance:", f"{row['attendance_pct']:.2f}%")
                        st.write("Avg Score:", f"{row['avg_score']:.2f}")
                        st.write("Attempts:", int(row['attempts']))
                        st.write("Fees Pending:", f"${row['fees_pending']:.2f}")
                    with c2:
                        st.write("Dropout Probability:", f"{row.get('dropout_probability',0):.2f}%")
                        st.write("Final Risk Score:", f"{row.get('final_risk_score',0):.2f}")
                        st.write("Risk Label:", row.get('risk_label','Unknown'))
                        # downloads
                        csv = row.to_frame().T.to_csv(index=False)
                        st.download_button("Download CSV", data=csv, file_name=f"{row['student_id']}_report.csv", mime="text/csv")
                        pdf_bytes = generate_pdf_bytes(row, fi if model is not None else pd.DataFrame())
                        st.download_button("Download PDF", data=pdf_bytes, file_name=f"{row['student_id']}_report.pdf", mime="application/pdf")
    with tab2:
        sid = st.selectbox("Select student for simulation", data['student_id'].unique())
        st.write("Adjust parameters to simulate")
        stud = data[data['student_id']==sid].iloc[0]
        na = st.slider("Attendance %", 0,100,int(stud['attendance_pct']))
        ns = st.slider("Avg Score", 0,100,int(stud['avg_score']))
        nat = st.slider("Attempts", 0,10,int(stud['attempts']))
        nf = st.slider("Fees Pending ($)", 0,5000,int(stud['fees_pending']))
        sim = stud.copy()
        sim['attendance_pct']=na; sim['avg_score']=ns; sim['attempts']=nat; sim['fees_pending']=nf
        sim_df = pd.DataFrame([sim])
        sim_df = compute_rule_scores(sim_df)
        sim_df = predict_with_model(model, sim_df)
        st.write("Current Risk:", stud['risk_label'], f"({stud.get('final_risk_score',stud.get('rule_risk_score',0)):.2f}%)")
        st.write("Simulated Risk:", sim_df.iloc[0]['risk_label'], f"({sim_df.iloc[0]['final_risk_score']:.2f}%)")
    with tab3:
        st.write("Download full dataset or individual reports")
        if st.button("Download Full CSV"):
            st.download_button("Download full CSV", data=data.to_csv(index=False), file_name="edu_guardians_full.csv", mime="text/csv")
        sel = st.selectbox("Select student for PDF report", data['student_id'].unique())
        if st.button("Generate PDF for selected"):
            row = data[data['student_id']==sel].iloc[0]
            pdfb = generate_pdf_bytes(row, fi if model is not None else pd.DataFrame())
            st.download_button("Download PDF", data=pdfb, file_name=f"{sel}_report.pdf", mime="application/pdf")

st.markdown("---")
st.caption("EduGuardians+ prototype. For SIH demo. Future: add sentiment analysis, notifications, ERP integration.")
