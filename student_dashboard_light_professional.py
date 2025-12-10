# student_dashboard_light_professional.py
"""
Student Performance — Polished Single-file Streamlit app (full-width, no right panel)
Includes:
 - SHAP background fix (match model_features exactly)
 - User-friendly input explanations + "About inputs" expander with references (APA)
 - Automatic Model & Methodology explanation area
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st
import streamlit.components.v1 as components
import math
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings

# suppress noisy sklearn warnings optionally
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    pass

# Plotly / palette (muted)
MUTED_SEQ = ["#cdb79a", "#9aa3a8", "#b9b3a6", "#a7b7c2", "#d6c9b0"]
try:
    px.defaults.color_discrete_sequence = MUTED_SEQ
    pio.templates.default = "plotly_white"
    px.defaults.template = "plotly_white"
except Exception:
    pass

mpl.rcParams['figure.facecolor'] = '#f7f5f2'
mpl.rcParams['axes.facecolor'] = '#f7f5f2'
mpl.rcParams['savefig.facecolor'] = '#f7f5f2'
plt.rcdefaults()

# Streamlit config
st.set_page_config(page_title="Student Performance — Polished UI", page_icon="📊", layout="wide")

# ---- CSS & JS (keeps visual look) ----
CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&family=Inter:wght@300;400;600;700&display=swap');
:root{
 --bg: #f7f5f2;
 --card: #ffffff;
 --muted: #6b7280;
 --accent1: #bda78a;
 --accent2: #a7b7c2;
 --glass: rgba(255,255,255,0.95);
 --stroke: rgba(16,24,40,0.04);
 --radius: 14px;
}
html, body, .stApp, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  color: #0b1220 !important;
  font-family: 'Inter', 'Poppins', sans-serif !important;
  -webkit-font-smoothing: antialiased;
  font-size:15px !important;
}
.app-header { display:flex; align-items:center; justify-content:space-between; padding:12px 24px; margin-bottom:12px; background: linear-gradient(90deg, rgba(189,167,138,0.06), rgba(167,183,194,0.02)); border-radius: var(--radius); box-shadow: 0 8px 24px rgba(16,24,40,0.03); }
.app-header h2 { margin:0; font-size:20px; font-weight:700; }
.app-header .sub { color:var(--muted); font-size:13px; margin-top:3px; }
.hero { padding:18px; border-radius:var(--radius); background:var(--glass); box-shadow: 0 6px 20px rgba(16,24,40,0.03); }
.hero h1 { margin:0; font-size:28px; font-weight:700; }
.hero p { margin:6px 0 0 0; color:var(--muted); font-size:13px; }
.kpi-grid { display:grid; grid-template-columns: repeat(4, 1fr); gap:14px; margin-top:14px; }
@media (max-width: 1100px) { .kpi-grid { grid-template-columns: repeat(2, 1fr); } }
.kpi-card { background: var(--card); border-radius:12px; padding:12px; border:1px solid var(--stroke); box-shadow: 0 10px 20px rgba(16,24,40,0.02); transition: transform .18s ease, box-shadow .18s ease; cursor: default; }
.kpi-card:hover { transform: translateY(-6px); box-shadow: 0 18px 30px rgba(16,24,40,0.06); }
.kpi-title { font-weight:600; color:#0b1220; font-size:13px; }
.kpi-value { font-weight:700; font-size:20px; margin-top:6px; }
.kpi-bar-wrap { height:12px; background: linear-gradient(90deg, rgba(11,18,32,0.03), rgba(11,18,32,0.01)); border-radius:8px; overflow:hidden; margin-top:10px; }
.kpi-bar { height:100%; width:0%; border-radius:8px; background: linear-gradient(90deg, var(--accent1), var(--accent2)); transition: width 1.0s cubic-bezier(.2,.9,.2,1); }
.card { background:var(--card); padding:12px; border-radius:var(--radius); border:1px solid var(--stroke); box-shadow: 0 8px 20px rgba(16,24,40,0.02); margin-bottom:14px; }
.card-title { font-weight:600; margin-bottom:8px; color:#0b1220; }
.small { color:var(--muted); font-size:13px; }
.app-footer { margin-top:20px; color:var(--muted); font-size:13px; }
.scroll-top { position: fixed; right: 18px; bottom: 18px; z-index:9999; background: linear-gradient(90deg,var(--accent2),var(--accent1)); color: white; border-radius: 10px; padding:8px 10px; box-shadow: 0 10px 30px rgba(16,24,40,0.12); border:none; cursor:pointer; }
.help-pill { display:inline-block; padding:6px 10px; border-radius:8px; background:#f3f2f1; font-size:13px; color:var(--muted); margin-top:6px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

JS = r"""
<script>
window.setTimeout(function(){
  try {
    document.querySelectorAll('.kpi-bar').forEach(function(el, idx){
      var fill = el.getAttribute('data-fill') || '0%';
      if(!fill.includes('%') && !isNaN(parseFloat(fill))){
        fill = parseFloat(fill) + '%';
      }
      if(!fill || fill === '0%') fill = '6%';
      setTimeout(function(){ el.style.width = fill; }, 100 + idx*80);
    });
  } catch(e){ console.log(e); }
}, 120);
function scrollTop() { window.scrollTo({top:0, behavior:'smooth'}); }
</script>
"""
components.html(JS, height=0)

# ---- Optional SHAP import ----
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    shap = None
    SHAP_AVAILABLE = False

# ---- TwoStageRegressor compatibility (unchanged) ----
class TwoStageRegressor:
    def __init__(self, clf=None, m_same=None, m_other=None, cols=None, preprocessor=None):
        self.clf = clf
        self.m_same = m_same
        self.m_other = m_other
        self.cols = cols
        self.preprocessor = preprocessor

    def predict(self, X):
        try:
            if hasattr(self, "preprocessor") and self.preprocessor is not None:
                Xp = pd.DataFrame(self.preprocessor.transform(X), columns=self.cols, index=X.index)
                X = Xp
            else:
                X = X[self.cols]
        except Exception:
            pass
        regime = self.clf.predict(X)
        yhat = np.zeros(len(X))
        mask_same = (regime == 1)
        mask_other = (regime == 0)
        if mask_same.any():
            yhat[mask_same] = self.m_same.predict(X.loc[mask_same])
        if mask_other.any():
            yhat[mask_other] = self.m_other.predict(X.loc[mask_other])
        return yhat

# ---- Paths (adjust if needed) ----
DATA_PATH  = "C:/studentproject/data/cleaned_dataset.csv"
MODEL_PATH = "C:/studentproject/models/student_performance_model.pkl"
FEAT_PATH  = "C:/studentproject/models/model_features.json"

# ---- Loaders with caching ----
@st.cache_data(show_spinner=False)
def load_data(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Normalize Gender
    if "Gender" in df.columns:
        raw = df["Gender"]
        g = raw.astype(str).str.strip().str.lower().replace({"nan": None, "none": None, "": None, "na": None, "n/a": None})
        male_vals = {"m", "male", "man", "male.", "m.", "masculine"}
        female_vals = {"f", "female", "woman", "female.", "f.", "fem"}
        def norm_gender(val):
            if val is None or pd.isna(val): return pd.NA
            v2 = str(val).strip().lower()
            if v2 in male_vals or v2.startswith("m"): return "Male"
            if v2 in female_vals or v2.startswith("f"): return "Female"
            return pd.NA
        df["Gender"] = g.apply(lambda x: norm_gender(x))
        df = df[df["Gender"].isin(["Male", "Female"])].copy()
        df["Gender"] = df["Gender"].astype(str)
    return df

@st.cache_resource(show_spinner=False)
def load_model_and_features(mpath, fpath):
    model = None
    features = []
    model_error = None
    if os.path.exists(mpath):
        try:
            model = joblib.load(mpath)
        except Exception as e:
            model = None
            model_error = str(e)
    else:
        model_error = f"Model file not found: {mpath}"
    if os.path.exists(fpath):
        try:
            with open(fpath, "r") as f:
                features = json.load(f)
        except Exception:
            features = []
    return model, features, model_error

# ---- Load data & model ----
df = load_data(DATA_PATH)
model, model_features, model_load_error = load_model_and_features(MODEL_PATH, FEAT_PATH)

# ---- SHAP setup (background builder that matches model_features exactly) ----
explainer = None
shap_diagnostics = []
if SHAP_AVAILABLE:
    shap_diagnostics.append(f"shap import OK (v{getattr(shap,'__version__', '?')})")
else:
    shap_diagnostics.append("shap not importable in environment")
if model is None:
    shap_diagnostics.append(f"Model not loaded: {model_load_error}")
else:
    shap_diagnostics.append(f"Model loaded: {model.__class__.__name__}")

def make_predict_wrapper(m, feat_list=None):
    def predict_fn(X):
        try:
            if isinstance(X, np.ndarray):
                if feat_list is not None and len(feat_list) == X.shape[1]:
                    try:
                        dfX = pd.DataFrame(X, columns=feat_list)
                        out = m.predict(dfX)
                        return np.asarray(out).ravel()
                    except Exception:
                        out = m.predict(X)
                        return np.asarray(out).ravel()
                else:
                    out = m.predict(X)
                    return np.asarray(out).ravel()
            else:
                out = m.predict(X)
                return np.asarray(out).ravel()
        except Exception:
            try:
                out = m.predict(X.values if hasattr(X, "values") else X)
                return np.asarray(out).ravel()
            except Exception as ex:
                raise ex
    return predict_fn

# Build background that has EXACTLY the same columns and order as model_features
background = None
try:
    if isinstance(model_features, list) and len(model_features) > 0 and not df.empty:
        df_bg = pd.DataFrame(index=df.index)
        dummies = pd.get_dummies(df, dummy_na=False)
        for feat in model_features:
            if feat in df.columns:
                df_bg[feat] = df[feat]
            elif feat in dummies.columns:
                df_bg[feat] = dummies[feat]
            else:
                # engineered / interaction fallback: compute if base exists or default zero
                if "_" in feat:
                    base, val = feat.split("_", 1)
                    if base in df.columns:
                        df_bg[feat] = (df[base].astype(str) == val).astype(int)
                    else:
                        df_bg[feat] = 0
                else:
                    df_bg[feat] = 0
        df_bg = df_bg.reindex(columns=model_features, fill_value=0)
        df_bg_clean = df_bg.dropna(how="all")
        if len(df_bg_clean) > 0:
            background = df_bg_clean.sample(n=min(200, len(df_bg_clean)), random_state=42)
            shap_diagnostics.append(f"Background built correctly: shape={background.shape}, matches model_features={len(model_features)}")
        else:
            background = None
            shap_diagnostics.append("Background build: no usable rows after dropna(how='all')")
except Exception as e:
    background = None
    shap_diagnostics.append(f"Background error: {e}")

if SHAP_AVAILABLE and model is not None and background is not None:
    if st.session_state.get("__shap_explainer") is not None:
        explainer = st.session_state["__shap_explainer"]
        shap_diagnostics.append("Re-using explainer from session_state")
    else:
        try:
            pred_wrap = make_predict_wrapper(model, feat_list=model_features if isinstance(model_features, list) else None)
            lowname = model.__class__.__name__.lower()
            if hasattr(model, "feature_importances_") or "xgb" in lowname or "lgb" in lowname or "forest" in lowname:
                try:
                    expl = shap.TreeExplainer(model, data=background, feature_perturbation="interventional")
                    explainer = expl
                    shap_diagnostics.append("Created shap.TreeExplainer(interventional)")
                except Exception as e:
                    shap_diagnostics.append(f"TreeExplainer failed ({e}), trying generic Explainer")
                    expl = shap.Explainer(pred_wrap, background)
                    explainer = expl
                    shap_diagnostics.append("Created shap.Explainer(pred_wrap, background)")
            else:
                expl = shap.Explainer(pred_wrap, background)
                explainer = expl
                shap_diagnostics.append("Created shap.Explainer(pred_wrap, background)")
            if explainer is not None:
                st.session_state["__shap_explainer"] = explainer
        except Exception as e:
            explainer = None
            shap_diagnostics.append(f"Explainer creation error: {e}")
else:
    shap_diagnostics.append("Explainer not created (missing shap/model/background)")

# ---------------- UI: Header & hero ----------------
st.markdown("<div class='app-header'><div><h2>Student Performance</h2><div class='sub'>AI-driven analysis — explore, predict & recommend</div></div></div>", unsafe_allow_html=True)
st.markdown("<div class='hero'><h1>Dashboard</h1><p>Overview and interactive tools to understand student performance at a glance.</p></div>", unsafe_allow_html=True)

# ---------------- KPI values ----------------
total = len(df)
avg_grade = df['FinalGrade'].mean() if "FinalGrade" in df.columns and len(df)>0 else 0.0
avg_hours = df['StudyHours'].mean() if "StudyHours" in df.columns and len(df)>0 else 0.0
pass_rate = ((df['FinalGrade'] >= 50).mean() * 100) if "FinalGrade" in df.columns and len(df)>0 else 0.0
avg_hours_fill = min(100.0, avg_hours * 5.0)
total_fill = 100.0 if total > 0 else 0.0

kpi_html = f"""
<div class='kpi-grid'>
  <div class='kpi-card card'>
    <div class='kpi-title'>🏠 Total Records</div>
    <div class='kpi-value'>{total:,}</div>
    <div class='kpi-bar-wrap'><div class='kpi-bar' data-fill='{total_fill:.1f}%'></div></div>
    <div class='small' style='margin-top:8px'>Dataset size</div>
  </div>
  <div class='kpi-card card'>
    <div class='kpi-title'>🏆 Avg Final Grade</div>
    <div class='kpi-value'>{avg_grade:.2f}</div>
    <div class='kpi-bar-wrap'><div class='kpi-bar' data-fill='{min(100, avg_grade):.1f}%'></div></div>
    <div class='small' style='margin-top:8px'>Average of FinalGrade column</div>
  </div>
  <div class='kpi-card card'>
    <div class='kpi-title'>⏳ Avg Study Hours</div>
    <div class='kpi-value'>{avg_hours:.1f} hrs</div>
    <div class='kpi-bar-wrap'><div class='kpi-bar' data-fill='{avg_hours_fill:.1f}%'></div></div>
    <div class='small' style='margin-top:8px'>Typical daily study hours</div>
  </div>
  <div class='kpi-card card'>
    <div class='kpi-title'>🎯 Pass Rate (>=50)</div>
    <div class='kpi-value'>{pass_rate:.0f}%</div>
    <div class='kpi-bar-wrap'><div class='kpi-bar' data-fill='{min(100, pass_rate):.1f}%'></div></div>
    <div class='small' style='margin-top:8px'>Percentage of students passing</div>
  </div>
</div>
"""
st.markdown(kpi_html, unsafe_allow_html=True)

# ---------------- Main content (full-width) ----------------
st.markdown("<div class='full-width'>", unsafe_allow_html=True)

# Navigation + page selection
with st.container():
    st.markdown("<div class='card'><div style='display:flex;align-items:center;justify-content:space-between'><div><strong style='font-size:16px'>Explore</strong><div class='small'>Choose a section below</div></div></div></div>", unsafe_allow_html=True)
page = st.radio("", ["Home","Dashboard","Deep Insights","Predict & Simulate"], index=0, label_visibility="collapsed", horizontal=True)

# ---------- ABOUT INPUTS & REFERENCES ----------
about_text = """
This section explains every input used in the 'Predict & Simulate' page, the measurement ranges, and why those ranges were chosen.
Use these texts in your thesis methodology where you explain variables and measurement scales.
"""
st.markdown("<div style='margin-top:6px'></div>", unsafe_allow_html=True)
with st.expander("About inputs, scales & references (click to expand)", expanded=False):
    st.markdown(f"<div class='card'><div class='card-title'>What each input means and measurement rationale</div><div class='small'>{about_text}</div></div>", unsafe_allow_html=True)

    st.markdown("### Key variables (brief)")
    st.markdown("""
- **LearningStyle_id (1/2/3)** — categorical encoding representing commonly used learning-style categories. In this app:
  - **1 = Visual learner** (prefers pictures, diagrams)
  - **2 = Auditory learner** (prefers listening, lectures)
  - **3 = Kinesthetic learner** (prefers hands-on activities)
  - *Rationale:* we encode learning style as a small categorical variable because many student surveys capture a dominant preference. Use carefully — evidence for the pedagogical impact of matching instruction to learning styles is mixed (see references).
- **Extracurricular (0/1)** — binary: 0 = no involvement in extracurricular activities; 1 = participates in at least one extracurricular activity. Chosen because extracurricular participation often correlates with engagement and soft skills.
- **StudyHours (0–20)** — measured as typical daily study hours. Chosen range 0–20 to cover realistic extremes (0 = no study; >12 is rare for daily average).
- **Attendance (0–100)** — percent attendance at classes.
- **ExamScore (0–100)** — last major exam percent score.
- **Motivation (1–5)** — self-reported motivation (1 low — 5 high).
- **StressLevel (1–5)** — self-reported stress (1 low — 5 high).
- **AssignmentCompletion (0–100)** — percent of assignments completed/submitted.
- **Resources (0–10)** — access to learning resources index (0 none — 10 many).
- Interaction / engineered features such as `Hours_x_Attendance`, `Exam_per_Hour`, `StudyHours_sq` are computed features used by the model to capture interactions and nonlinear effects.
""")

    st.markdown("### Measurement rationale (short)")
    st.markdown("""
- Percent scales (0–100) are commonly used for attendance, exam scores and completion because they are interpretable.
- Motivation and stress use small Likert ranges (1–5) to make self-report simple; these are commonly used in educational psychology surveys.
- Study hours use a daily-average realistic bound (0–20) — values very large are unlikely and may indicate input error.
""")
    st.markdown("### References (APA style)")
    st.markdown("""
- Pashler, H., McDaniel, M., Rohrer, D., & Bjork, R. (2008). *Learning styles: Concepts and evidence*. _Psychological Science in the Public Interest, 9_(3), 105–119. https://doi.org/10.1111/j.1539-6053.2008.00028.x  
- Ryan, R. M., & Deci, E. L. (2000). *Self-determination theory and the facilitation of intrinsic motivation, social development, and well-being*. _American Psychologist, 55_(1), 68–78. https://doi.org/10.1037/0003-066X.55.1.68  
- Fleming, N. D. (2001). *VARK: A guide to learning styles* (website). VARK Learn Limited. https://vark-learn.com/  
- Lundberg, S. M., & Lee, S.-I. (2017). *A unified approach to interpreting model predictions*. Advances in Neural Information Processing Systems, 30. (arXiv:1705.07874). https://arxiv.org/abs/1705.07874
- Chen, T., & Guestrin, C. (2016). *XGBoost: A scalable tree boosting system*. Proceedings of the 22nd ACM SIGKDD. https://arxiv.org/abs/1603.02754
""")

# ---------- Page: Home ----------
if page == "Home":
    st.markdown("<div class='card'><div class='card-title'>Overview</div><div class='small'>This dashboard helps you explore student performance, identify drivers and simulate suggested improvements.</div></div>", unsafe_allow_html=True)
    with st.expander("Data snapshot (first 10 rows)", expanded=False):
        if df.empty:
            st.info("No data loaded (check DATA_PATH).")
        else:
            st.dataframe(df.head(10), use_container_width=True)

# ---------- Page: Dashboard ----------
elif page == "Dashboard":
    st.markdown("<div class='card'><div class='card-title'>Final Grade Distribution</div></div>", unsafe_allow_html=True)
    if "FinalGrade" in df.columns and len(df)>0:
        fig = px.histogram(df, x="FinalGrade", nbins=30, color_discrete_sequence=[MUTED_SEQ[0]])
        fig.update_traces(marker_line_width=0)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#0b1220")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("FinalGrade missing or dataset empty.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='card'><div class='card-title'>Study vs Final</div></div>", unsafe_allow_html=True)
        if {"StudyHours", "FinalGrade"}.issubset(df.columns) and len(df) > 0:
            df_plot = df.copy()
            fig2 = px.scatter(df_plot, x="StudyHours", y="FinalGrade", opacity=0.75, color_discrete_sequence=[MUTED_SEQ[1]])
            fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#0b1220")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("StudyHours or FinalGrade missing.")
    with c2:
        st.markdown("<div class='card'><div class='card-title'>Top Students</div></div>", unsafe_allow_html=True)
        if "FinalGrade" in df.columns and len(df)>0:
            cols_to_show = [c for c in ["FinalGrade","StudyHours","Attendance"] if c in df.columns]
            st.table(df.sort_values("FinalGrade", ascending=False).head(5)[cols_to_show])
        else:
            st.info("No data to show.")

# ---------- Page: Deep Insights ----------
elif page == "Deep Insights":
    st.markdown("<div class='card'><div class='card-title'>Deep Insights — Interactive Data Playground</div><div class='small'>Filter, visualize, run PCA / clustering, export, or apply the model to filtered rows.</div></div>", unsafe_allow_html=True)
    df_work = df.copy()
    if df_work.empty:
        st.info("No dataset loaded (check DATA_PATH).")
    else:
        num_cols = df_work.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df_work.select_dtypes(exclude=[np.number]).columns.tolist()

        # Filters
        st.markdown("<div class='card'><div class='card-title'>Filters</div></div>", unsafe_allow_html=True)
        with st.expander("Filter data (numeric ranges and categorical selection)", expanded=True):
            numeric_filters_local = {}
            if num_cols:
                var_sorted = sorted(num_cols, key=lambda c: float(np.nanstd(df[c]) if c in df else 0), reverse=True)
                range_cols = var_sorted[:6]
                for c in range_cols:
                    try:
                        mn = float(np.nanmin(df[c])) if df[c].dropna().size>0 else 0.0
                        mx = float(np.nanmax(df[c])) if df[c].dropna().size>0 else 1.0
                    except Exception:
                        mn, mx = 0.0, 1.0
                    keyname = f"filter_range_{c}"
                    default = st.session_state.get("filter_numeric_filters", {}).get(c, (mn, mx))
                    step = (mx-mn)/50 if mx>mn else 1.0
                    lo, hi = st.slider(f"{c} range", mn, mx, default, step=step, key=keyname)
                    numeric_filters_local[c] = (lo, hi)
                st.session_state["filter_numeric_filters"] = numeric_filters_local
                for c,(lo,hi) in numeric_filters_local.items():
                    df_work = df_work[(df_work[c].isna()) | ((df_work[c] >= lo) & (df_work[c] <= hi))]
            else:
                st.info("No numeric columns available for range filters.")

            if cat_cols:
                cat_choice = st.multiselect("Filter categories (by column)", cat_cols[:6], default=st.session_state.get("filter_cat_choice", cat_cols[:min(2,len(cat_cols))]), key="filter_cat_choice")
                for c in cat_choice:
                    if c == "Gender":
                        opts = ["Male", "Female"]
                    else:
                        opts = sorted(df[c].dropna().unique().tolist())
                    prev = st.session_state.get(f"filter_values_{c}", opts[:min(6,len(opts))])
                    sel = st.multiselect(f"Values for {c}", opts, default=prev, key=f"filter_values_{c}")
                    if sel:
                        df_work = df_work[df_work[c].isin(sel)]
            else:
                st.info("No categorical columns available for filtering.")

        st.markdown(f"<div class='small' style='margin-top:8px'>Filtered rows: <strong style='color:#0b1220'>{len(df_work):,}</strong> (from {len(df):,})</div>", unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Visualizer (omitted repeating code for brevity, same as earlier)
        vis_cols = st.multiselect("Choose numeric columns to visualize (1-6)", num_cols, default=num_cols[:3])
        plot_type = st.radio("Plot type", ["Histogram","Box","Violin","Scatter"], index=0, horizontal=True)
        # ... same plotting logic as before (kept) ...
        if plot_type in ["Histogram","Box","Violin"]:
            for c in vis_cols[:4]:
                st.markdown(f"<div class='card'><div class='card-title'>{plot_type}: {c}</div></div>", unsafe_allow_html=True)
                if plot_type == "Histogram":
                    fig = px.histogram(df_work, x=c, nbins=30, color_discrete_sequence=[MUTED_SEQ[0]])
                    fig.update_traces(marker_line_width=0)
                elif plot_type == "Box":
                    groupby = st.selectbox(f"Group boxplot by (optional) for {c}", [None] + cat_cols, index=0, key=f"box_group_{c}")
                    if groupby:
                        fig = px.box(df_work, x=groupby, y=c, points="all", color_discrete_sequence=MUTED_SEQ)
                    else:
                        fig = px.box(df_work, y=c, points="all", color_discrete_sequence=[MUTED_SEQ[0]])
                else:
                    groupby = st.selectbox(f"Group violin by (optional) for {c}", [None] + cat_cols, index=0, key=f"violin_group_{c}")
                    if groupby:
                        fig = px.violin(df_work, x=groupby, y=c, box=True, points="all", color_discrete_sequence=MUTED_SEQ)
                    else:
                        fig = px.violin(df_work, y=c, box=True, points="all", color_discrete_sequence=[MUTED_SEQ[0]])
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#0b1220")
                st.plotly_chart(fig, use_container_width=True)
        elif plot_type == "Scatter":
            if len(vis_cols) < 2:
                st.info("Select at least 2 numeric columns for scatter.")
            else:
                xcol = st.selectbox("X axis", vis_cols, index=0, key="scatter_x")
                ycol = st.selectbox("Y axis", vis_cols, index=1 if len(vis_cols)>1 else 0, key="scatter_y")
                colorcol = st.selectbox("Color by (optional)", [None] + cat_cols, index=0, key="scatter_color")
                df_plot2 = df_work.copy()
                if colorcol == "Gender" and "Gender" in df_plot2.columns:
                    df_plot2["Gender"] = df_plot2["Gender"].astype(str).str.strip()
                    df_plot2 = df_plot2[df_plot2["Gender"].isin(["Male","Female"])]
                    if df_plot2.shape[0] == 0:
                        fig = px.scatter(df_work, x=xcol, y=ycol, opacity=0.75, color_discrete_sequence=[MUTED_SEQ[1]])
                    else:
                        fig = px.scatter(df_plot2, x=xcol, y=ycol, color="Gender", opacity=0.75, color_discrete_sequence=MUTED_SEQ)
                else:
                    fig = px.scatter(df_plot2, x=xcol, y=ycol, opacity=0.75, color_discrete_sequence=[MUTED_SEQ[1]])
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#0b1220")
                st.plotly_chart(fig, use_container_width=True)

        # PCA & Clustering (same logic as earlier)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Apply model to filtered data (keeps original robust prediction batching)
        st.markdown("<div class='card'><div class='card-title'>Apply model to filtered data</div></div>", unsafe_allow_html=True)
        st.markdown("<div class='small'>Preview sample rows then confirm to run predictions on the filtered dataset. For large sets (>10,000 rows) you must type CONFIRM to proceed.</div>", unsafe_allow_html=True)
        preview_btn = st.button("Preview sample (up to 10 rows) for prediction", key="preview_predict")
        if preview_btn:
            if df_work.shape[0] == 0:
                st.warning("No rows in filtered set.")
            else:
                sample_n = min(10, len(df_work))
                sample_preview = df_work.head(sample_n)
                st.markdown(f"#### Sample preview (first {sample_n} rows)")
                st.dataframe(sample_preview, use_container_width=True)
                st.session_state["preview_ready"] = True
                st.session_state["preview_count"] = len(df_work)

        if st.session_state.get("preview_ready", False):
            st.markdown("<div class='small' style='color:var(--muted)'>Sample preview shown — confirm to run predictions on the entire filtered set.</div>", unsafe_allow_html=True)
            total_rows = len(df_work)
            max_guard = 10000
            if total_rows > max_guard:
                st.warning(f"Filtered set has {total_rows:,} rows (>{max_guard:,}). For safety, type CONFIRM below to proceed.")
                confirm_text = st.text_input("Type CONFIRM to allow prediction on this large set", key="large_confirm_text")
                confirm_ok = (confirm_text.strip().upper() == "CONFIRM")
                if not confirm_ok:
                    st.info("Prediction blocked until you type CONFIRM.")
                confirm_run = st.button(f"Confirm and run predictions on {total_rows:,} rows (large set)", key="confirm_run_large")
            else:
                confirm_ok = True
                confirm_run = st.button(f"Confirm: run predictions on {total_rows:,} rows", key="confirm_run")

            if confirm_run:
                if not confirm_ok:
                    st.error("You must type CONFIRM to proceed for large datasets.")
                else:
                    if model is None:
                        st.error("Model not loaded.")
                    elif not isinstance(model_features, list) or len(model_features) == 0:
                        st.error("Model features not loaded (model_features.json).")
                    else:
                        try:
                            df_model_input = pd.DataFrame(index=df_work.index)
                            dummies = pd.get_dummies(df_work, dummy_na=False)
                            for feat in model_features:
                                if feat in df_work.columns:
                                    df_model_input[feat] = df_work[feat]
                                elif feat in dummies.columns:
                                    df_model_input[feat] = dummies[feat]
                                else:
                                    if "_" in feat:
                                        base, val = feat.split("_", 1)
                                        if base in df_work.columns:
                                            df_model_input[feat] = (df_work[base].astype(str) == val).astype(int)
                                        else:
                                            df_model_input[feat] = 0
                                    else:
                                        df_model_input[feat] = 0
                            df_model_input = df_model_input.reindex(columns=model_features, fill_value=0)

                            total = len(df_model_input)
                            batch_size = 500
                            n_batches = math.ceil(total / batch_size)
                            progress = st.progress(0.0)
                            status = st.empty()
                            preds_list = []
                            start_time = time.time()

                            for i in range(n_batches):
                                start = i * batch_size
                                end = min(total, (i+1) * batch_size)
                                chunk = df_model_input.iloc[start:end]
                                try:
                                    chunk_preds = model.predict(chunk)
                                except Exception as e1:
                                    try:
                                        chunk_preds = model.predict(df_work.iloc[start:end])
                                    except Exception as e2:
                                        status.error(f"Prediction failed on batch {i+1}/{n_batches}: {e1} / {e2}")
                                        preds_list = None
                                        break
                                preds_list.append(np.array(chunk_preds).ravel())
                                processed = end
                                progress.progress(min(1.0, processed / total))
                                status.text(f"Predicting... batch {i+1}/{n_batches} — processed {processed:,}/{total:,} rows")
                                time.sleep(0.01)

                            if preds_list is not None:
                                preds = np.concatenate(preds_list, axis=0)
                                end_time = time.time()
                                status.success(f"Prediction completed in {end_time - start_time:.1f}s — processed {total:,} rows.")
                                progress.progress(1.0)
                                results_df = df_work.copy()
                                results_df["PredictedGrade"] = preds
                                if "FinalGrade" in results_df.columns:
                                    results_df["Delta"] = results_df["PredictedGrade"] - results_df["FinalGrade"]
                                st.markdown("#### Predictions for filtered rows (first 200 shown)")
                                st.dataframe(results_df.head(200), use_container_width=True)
                                csv_out = results_df.to_csv(index=False)
                                st.download_button("Download predictions CSV", csv_out, file_name="predictions_filtered.csv", mime="text/csv")
                            else:
                                st.error("Prediction aborted due to errors in batches.")
                            st.session_state["preview_ready"] = False
                        except Exception as e:
                            st.error(f"Failed to build model input / predict: {e}")
                            st.session_state["preview_ready"] = False

# ---------- Page: Predict & Simulate (with user-friendly help texts) ----------
elif page == "Predict & Simulate":
    st.markdown("<div class='card'><div class='card-title'>Predict & Simulate</div></div>", unsafe_allow_html=True)
    # layout three columns for inputs
    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.selectbox("Gender", ["Male","Female"])
        st.caption("Student's self-reported gender; used only for modeling demographic differences.")
        age = st.number_input("Age", 15, 60, 21)
        st.caption("Age in years (typical university range used here).")
        study = st.slider("Study Hours / day", 0.0, 20.0, 8.0, 0.5)
        st.caption("Typical daily study hours (0–20). Chosen bound covers normal daily study variations.")
        learn = st.selectbox("LearningStyle id (1=Visual,2=Auditory,3=Kinesthetic)", [1,2,3], index=0)
        st.caption("Primary learning preference. 1=Visual, 2=Auditory, 3=Kinesthetic. See VARK literature.")
        extra = st.selectbox("Extracurricular (0 = No, 1 = Yes)", [0,1], index=0)
        st.caption("Participation in extracurricular activities; 1 indicates involvement in at least one activity.")
        online = st.selectbox("Online Courses (0 = No, 1 = Yes)", [0,1], index=0)
        st.caption("Whether the student takes additional online courses or MOOCs (binary).")
    with c2:
        attendance = st.slider("Attendance (%)", 0, 100, 90)
        st.caption("Class attendance percentage (0–100).")
        exam = st.slider("Exam Score (%)", 0, 100, 70)
        st.caption("Latest major exam score (percent).")
        assign = st.slider("Assignment Completion (%)", 0, 100, 80)
        st.caption("Percent of assignments completed/submitted (0–100).")
    with c3:
        motivation = st.slider("Motivation (1-5)", 1, 5, 3)
        st.caption("Self-reported motivation; Likert scale 1 (low) to 5 (high).")
        stress = st.slider("Stress Level (1-5)", 1, 5, 2)
        st.caption("Self-reported stress; Likert scale 1 (low) to 5 (high).")
        resources = st.slider("Resources (0-10)", 0, 10, 5)
        st.caption("Access to learning resources index (0 none — 10 many).")

    st.markdown("<div class='card'><div class='card-title'>Recommendation actions (select which to simulate)</div></div>", unsafe_allow_html=True)
    col_a1, col_a2, col_a3 = st.columns(3)
    with col_a1:
        act_study = st.checkbox("Study +2 hours/day", value=True)
        act_attendance = st.checkbox("Attendance +5%", value=True)
        act_assignment = st.checkbox("Assignment +10%", value=True)
    with col_a2:
        act_motivation = st.checkbox("Motivation +1", value=True)
        act_stress = st.checkbox("Reduce Stress -1", value=True)
        act_exam = st.checkbox("Exam Score +5", value=True)
    with col_a3:
        act_resources = st.checkbox("Use +2 Resources", value=True)

    # Show short help pill and link to About inputs
    st.markdown("<div class='help-pill'>Tip: open 'About inputs' at top to read measurement justifications and references.</div>", unsafe_allow_html=True)

    # Build X_input matching model_features exactly
    if not isinstance(model_features, list) or len(model_features) == 0:
        st.error("Model features not loaded. Predictions disabled until model_features.json is available.")
    else:
        x = {col: 0 for col in model_features}
        def set_if_present(k, v):
            if k in x:
                x[k] = v
        set_if_present("Age", age)
        set_if_present("StudyHours", study)
        set_if_present("Attendance", attendance)
        set_if_present("ExamScore", exam)
        set_if_present("Motivation", motivation)
        set_if_present("StressLevel", stress)
        set_if_present("AssignmentCompletion", assign)
        set_if_present("Extracurricular", extra)
        set_if_present("OnlineCourses", online)
        # computed engineered features
        eps = 1e-6
        if "Hours_x_Attendance" in x:
            x["Hours_x_Attendance"] = x.get("StudyHours", study) * x.get("Attendance", attendance)
        if "Exam_per_Hour" in x:
            x["Exam_per_Hour"] = x.get("ExamScore", exam) / (x.get("StudyHours", study) + eps)
        if "Mot_minus_Stress" in x:
            x["Mot_minus_Stress"] = x.get("Motivation", motivation) - x.get("StressLevel", stress)
        if "StudyHours_sq" in x:
            x["StudyHours_sq"] = x.get("StudyHours", study) ** 2
        if "Attendance_sq" in x:
            x["Attendance_sq"] = x.get("Attendance", attendance) ** 2
        # learning style one-hot (if model expects prefix)
        if "LearningStyle_1" in x or "LearningStyle_2" in x or "LearningStyle_3" in x:
            for ls in [1,2,3]:
                key = f"LearningStyle_{ls}"
                x[key] = 1 if learn == ls else 0

        try:
            X_input = pd.DataFrame([x])[model_features]
        except Exception as e:
            X_input = None
            st.error(f"Could not build model input: {e}")

        if st.button("🚀 Predict Now"):
            if model is None:
                st.error("Model not loaded.")
            elif X_input is None:
                st.error("Invalid model input (check model_features.json).")
            else:
                try:
                    pred = float(model.predict(X_input)[0])
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    pred = None

                if pred is not None:
                    if pred >= 85:
                        color, emoji = "#10b981", "🏆 Excellent"
                    elif pred >= 70:
                        color, emoji = "#7c3aed", "🎯 Good"
                    elif pred >= 55:
                        color, emoji = "#f59e0b", "⚡ Average"
                    else:
                        color, emoji = "#ef4444", "💤 Poor"
                    st.markdown(f"<div class='card' style='text-align:center;background:linear-gradient(90deg,{color},#ffffff);'><h2 style='margin:8px 0;color:#071023'>{pred:.2f}/100</h2><div style='color:#071023'>{emoji}</div></div>", unsafe_allow_html=True)

                    # simulate selected actions
                    actions_all = [
                        ("Study +2 hours/day", "StudyHours", lambda v: v + 2),
                        ("Attendance +5%", "Attendance", lambda v: min(100, v + 5)),
                        ("Assignment +10%", "AssignmentCompletion", lambda v: min(100, v + 10)),
                        ("Motivation +1", "Motivation", lambda v: min(5, v + 1)),
                        ("Reduce Stress -1", "StressLevel", lambda v: max(1, v - 1)),
                        ("Exam Score +5", "ExamScore", lambda v: min(100, v + 5)),
                        ("Use +2 Resources", "Resources", lambda v: v + 2),
                    ]
                    mapping_toggle = {
                        "Study +2 hours/day": act_study,
                        "Attendance +5%": act_attendance,
                        "Assignment +10%": act_assignment,
                        "Motivation +1": act_motivation,
                        "Reduce Stress -1": act_stress,
                        "Exam Score +5": act_exam,
                        "Use +2 Resources": act_resources
                    }
                    selected_actions = [act for act in actions_all if mapping_toggle.get(act[0], False)]
                    baseline = X_input.iloc[0].to_dict()
                    results = []
                    for desc, feat, change_fn in selected_actions:
                        if feat not in model_features:
                            continue
                        new = baseline.copy()
                        old_val = new.get(feat, 0)
                        new_val = change_fn(old_val)
                        new[feat] = new_val
                        # recompute engineered if present
                        if "Hours_x_Attendance" in model_features:
                            new["Hours_x_Attendance"] = new.get("StudyHours", baseline.get("StudyHours", study)) * new.get("Attendance", baseline.get("Attendance", attendance))
                        if "Exam_per_Hour" in model_features:
                            eps = 1e-6
                            new["Exam_per_Hour"] = new.get("ExamScore", baseline.get("ExamScore", exam)) / (new.get("StudyHours", baseline.get("StudyHours", study)) + eps)
                        if "Mot_minus_Stress" in model_features:
                            new["Mot_minus_Stress"] = new.get("Motivation", baseline.get("Motivation", motivation)) - new.get("StressLevel", baseline.get("StressLevel", stress))
                        if "StudyHours_sq" in model_features:
                            new["StudyHours_sq"] = new.get("StudyHours", baseline.get("StudyHours", study)) ** 2
                        if "Attendance_sq" in model_features:
                            new["Attendance_sq"] = new.get("Attendance", baseline.get("Attendance", attendance)) ** 2
                        try:
                            new_df = pd.DataFrame([new])[model_features]
                            new_pred = float(model.predict(new_df)[0])
                            delta = new_pred - pred
                            results.append({"action": desc, "feature": feat, "old_value": old_val, "new_value": new_val, "predicted_grade": new_pred, "delta": delta})
                        except Exception:
                            results.append({"action": desc, "feature": feat, "old_value": old_val, "new_value": new_val, "predicted_grade": None, "delta": None})
                    if len(results) == 0:
                        st.info("No recommendation actions selected or none applicable for model_features.")
                    else:
                        rec_df = pd.DataFrame(results).sort_values("delta", ascending=False, na_position="last").reset_index(drop=True)
                        st.markdown("#### Top Suggestions")
                        st.dataframe(rec_df[["action","old_value","new_value","predicted_grade","delta"]], use_container_width=True)

                    # SHAP explainability (if available)
                    if SHAP_AVAILABLE and explainer is not None:
                        st.markdown("### SHAP Explainability")
                        try:
                            shap_input = X_input.copy()
                            shap_vals = explainer(shap_input, check_additivity=False)
                        except Exception as e:
                            shap_vals = None
                            st.warning(f"SHAP compute failed: {e}")
                        if shap_vals is not None:
                            try:
                                sv = shap_vals.values[0] if hasattr(shap_vals, "values") else np.asarray(shap_vals)[0]
                                fnames = X_input.columns.tolist()
                                imp_df = pd.DataFrame({"feature": fnames, "shap_abs": np.abs(sv)}).sort_values("shap_abs", ascending=False).head(12)
                                fig = px.bar(imp_df, x="shap_abs", y="feature", orientation="h", color_discrete_sequence=[MUTED_SEQ[2]])
                                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#0b1220")
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.warning(f"Could not render SHAP plots: {e}")
                        else:
                            st.info("SHAP values not available for this input (see diagnostics).")
                    else:
                        st.info("SHAP not available in this environment or explainer not created. See logs/diagnostics.")

# close full-width wrapper
st.markdown("</div>", unsafe_allow_html=True)

# ---------- Model & Methodology (auto explanation) ----------
st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
with st.expander("Model & Methodology (what model is used and why)", expanded=True):
    if model is None:
        st.markdown("<div class='small'>No model file loaded. Place your trained model at MODEL_PATH to enable predictions. The UI will explain the model once loaded.</div>", unsafe_allow_html=True)
        st.markdown("<div class='small'>Model load error (if any):</div>")
        st.text(model_load_error)
    else:
        mname = model.__class__.__name__
        st.markdown(f"<div class='card'><div class='card-title'>Model detected: {mname}</div></div>", unsafe_allow_html=True)
        low = mname.lower()
        # provide tailored explanation
        if "xgb" in low or "xgboost" in low:
            st.markdown("""
**Recommended model: XGBoost (gradient boosting trees)**  
**Why chosen:** XGBoost is a fast, regularized gradient-boosted tree implementation that works very well on tabular data, handles nonlinearities and interactions, and is robust to different feature scales. It also provides feature importance measures and integrates well with SHAP for explanations (TreeExplainer).  
**Strengths:** handles missing values, strong predictive accuracy on tabular data, interpretable with SHAP.  
**Reference:** Chen, T., & Guestrin, C. (2016). *XGBoost: A scalable tree boosting system*. Proceedings of the 22nd ACM SIGKDD. https://arxiv.org/abs/1603.02754
""", unsafe_allow_html=True)
        elif "random" in low or "forest" in low:
            st.markdown("""
**Model type: Random Forest (ensemble of decision trees)**  
**Why used:** Random Forests are robust, reduce overfitting by averaging many decision trees, and provide feature importances. They work well when relationships are nonlinear and there are interactions between predictors. SHAP can explain tree models via TreeExplainer.  
**Reference:** Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324
""", unsafe_allow_html=True)
        elif "linear" in low or "ridge" in low or "lasso" in low:
            st.markdown("""
**Model type: Linear model (e.g., Linear Regression, Ridge, Lasso)**  
**Why used:** Linear models are simple and interpretable; coefficients show direct effect sizes. They are good baseline models and helpful when relationships are expected to be approximately linear.  
**When to prefer:** if interpretability and simple assumptions are required, otherwise consider tree-based models for complex interactions.
""", unsafe_allow_html=True)
        else:
            st.markdown("""
**Model type:** The loaded model is a custom or composite model. Tree-based ensemble models (like XGBoost or Random Forest) are commonly used for student performance prediction because they handle tabular and heterogeneous features well and work with SHAP for interpretability.  
**Interpretability:** Use SHAP (if available) to understand per-feature contributions to predictions.  
**SHAP reference:** Lundberg, S. M., & Lee, S.-I. (2017). *A unified approach to interpreting model predictions*. NeurIPS. https://arxiv.org/abs/1705.07874
""", unsafe_allow_html=True)

    # Provide guidance on how to re-train or replace model (short)
    st.markdown("<div class='small' style='margin-top:8px'>To replace the model: train a model that expects features listed in <code>model_features.json</code>, save with joblib.dump(model, MODEL_PATH), and restart Streamlit. The feature order and names must match exactly for predictions and SHAP to work.</div>", unsafe_allow_html=True)

# Floating 'scroll to top' button
st.markdown("<button class='scroll-top' onclick='scrollTop()'>↑ Top</button>", unsafe_allow_html=True)

# Footer
st.markdown("<div class='app-footer'>Built with Streamlit — UI upgraded (full-width, clear inputs & explanations). References included for methodology.</div>", unsafe_allow_html=True)
