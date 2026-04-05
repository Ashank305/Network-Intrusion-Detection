"""
🛡️ Network Intrusion Detection System
Streamlit Frontend Application

A Machine Learning-powered system that detects network intrusions
using Random Forest classifier trained on the KDD Cup 1999 dataset.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import joblib

# Page config — MUST be first Streamlit command
st.set_page_config(
    page_title="🛡️ Network Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  CUSTOM CSS — Dark Glassmorphism Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* ===== Import Font ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ===== Global ===== */
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #1a1a2e 30%, #16213e 60%, #0f3460 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* ===== Sidebar ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d1f 0%, #1a1a2e 100%) !important;
        border-right: 1px solid rgba(233, 69, 96, 0.3);
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #e94560 !important;
    }
    
    /* ===== Glass Card ===== */
    .glass-card {
        background: rgba(22, 33, 62, 0.6);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(233, 69, 96, 0.15);
        border-radius: 16px;
        padding: 24px;
        margin: 12px 0;
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(233, 69, 96, 0.4);
        box-shadow: 0 8px 32px rgba(233, 69, 96, 0.1);
        transform: translateY(-2px);
    }
    
    /* ===== Hero Section ===== */
    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #e94560, #ff6b6b, #e94560);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0;
        animation: glow 3s ease-in-out infinite;
    }
    @keyframes glow {
        0%, 100% { filter: brightness(1); }
        50% { filter: brightness(1.3); }
    }
    
    .hero-subtitle {
        font-size: 1.15rem;
        color: #8892b0;
        text-align: center;
        font-weight: 300;
        margin-top: 8px;
    }
    
    /* ===== Metric Cards ===== */
    .metric-card {
        background: linear-gradient(135deg, rgba(22, 33, 62, 0.8), rgba(15, 52, 96, 0.6));
        border: 1px solid rgba(233, 69, 96, 0.2);
        border-radius: 16px;
        padding: 20px 24px;
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(233, 69, 96, 0.15);
        border-color: rgba(233, 69, 96, 0.5);
    }
    .metric-value {
        font-size: 2.4rem;
        font-weight: 800;
        color: #e94560;
        line-height: 1.2;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8892b0;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 6px;
        font-weight: 500;
    }
    
    /* ===== Status Badge ===== */
    .status-normal {
        background: linear-gradient(135deg, #00b894, #00cec9);
        color: white;
        padding: 8px 24px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(0, 184, 148, 0.3);
    }
    .status-attack {
        background: linear-gradient(135deg, #e94560, #ff6348);
        color: white;
        padding: 8px 24px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(233, 69, 96, 0.3);
        animation: pulse 1.5s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { box-shadow: 0 4px 15px rgba(233, 69, 96, 0.3); }
        50% { box-shadow: 0 4px 25px rgba(233, 69, 96, 0.6); }
    }
    
    /* ===== Architecture Diagram ===== */
    .arch-step {
        background: rgba(22, 33, 62, 0.7);
        border-left: 4px solid #e94560;
        padding: 16px 20px;
        margin: 10px 0;
        border-radius: 0 12px 12px 0;
        transition: all 0.3s ease;
    }
    .arch-step:hover {
        background: rgba(233, 69, 96, 0.1);
        transform: translateX(8px);
    }
    .arch-step-num {
        color: #e94560;
        font-weight: 800;
        font-size: 1.3rem;
    }
    .arch-step-title {
        color: #e0e0e0;
        font-weight: 600;
        font-size: 1.05rem;
    }
    .arch-step-desc {
        color: #8892b0;
        font-size: 0.9rem;
    }
    
    /* ===== Section Headers ===== */
    .section-header {
        font-size: 1.6rem;
        font-weight: 700;
        color: #e0e0e0;
        border-bottom: 2px solid rgba(233, 69, 96, 0.3);
        padding-bottom: 8px;
        margin: 24px 0 16px 0;
    }
    
    /* ===== Streamlit Overrides ===== */
    .stButton > button {
        background: linear-gradient(135deg, #e94560, #c0392b) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 32px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(233, 69, 96, 0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(233, 69, 96, 0.5) !important;
    }
    
    [data-testid="stMetric"] {
        background: rgba(22, 33, 62, 0.6);
        border: 1px solid rgba(233, 69, 96, 0.15);
        border-radius: 12px;
        padding: 16px;
    }
    
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(22, 33, 62, 0.6) !important;
        border-radius: 10px !important;
        color: #8892b0 !important;
        border: 1px solid rgba(233, 69, 96, 0.1) !important;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(233, 69, 96, 0.2) !important;
        color: #e94560 !important;
        border-color: #e94560 !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #e94560, #ff6b6b) !important;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(233, 69, 96, 0.3) !important;
        border-radius: 16px !important;
        padding: 20px !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #555;
        font-size: 0.8rem;
        margin-top: 60px;
        padding: 20px;
        border-top: 1px solid rgba(233, 69, 96, 0.1);
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ NIDS")
    st.markdown("**Network Intrusion Detection System**")
    st.markdown("---")
    
    page = st.radio(
        "Navigate",
        ["🏠 Home", "📊 Dataset Explorer", "🧠 Model Training", "📈 Evaluation", "🔍 Live Prediction"],
        index=0,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ Model Info")
    
    model_exists = os.path.exists('saved_model/random_forest.pkl')
    if model_exists:
        st.success("✅ Model Trained")
        model_size = os.path.getsize('saved_model/random_forest.pkl') / (1024 * 1024)
        st.caption(f"Model size: {model_size:.1f} MB")
    else:
        st.warning("⚠️ No trained model found")
        st.caption("Go to Model Training page")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; color:#555; font-size: 0.75rem;'>
        Built with ❤️ using<br>
        Scikit-learn & Streamlit<br><br>
        <b>KDD Cup 1999 Dataset</b><br>
        Random Forest Classifier
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  PAGE: HOME
# ─────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown('<h1 class="hero-title">🛡️ Network Intrusion Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Machine Learning-powered network security analysis using Random Forest on KDD Cup 1999 dataset</p>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Key stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">41</div>
            <div class="metric-label">Features</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">5</div>
            <div class="metric-label">Attack Types</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">RF</div>
            <div class="metric-label">Algorithm</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">99%+</div>
            <div class="metric-label">Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Architecture
    st.markdown('<div class="section-header">🏗️ System Architecture</div>', unsafe_allow_html=True)
    
    steps = [
        ("01", "📥 Data Collection", "KDD Cup 1999 dataset — network traffic records with 41 features"),
        ("02", "🔧 Data Preprocessing", "Label encoding categorical features, Standard scaling, Binary label creation"),
        ("03", "✂️ Train-Test Split", "80% training / 20% testing with stratified sampling"),
        ("04", "🌳 Model Training", "Random Forest Classifier — 100 decision trees, ensemble learning"),
        ("05", "📊 Model Evaluation", "Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC Curve"),
        ("06", "🔍 Intrusion Detection", "Real-time prediction — classify network traffic as Normal or Attack"),
    ]
    
    for num, title, desc in steps:
        st.markdown(f"""
        <div class="arch-step">
            <span class="arch-step-num">{num}</span> &nbsp;
            <span class="arch-step-title">{title}</span><br>
            <span class="arch-step-desc">{desc}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Quick explanation section
    st.markdown('<div class="section-header">📚 About This Project</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #e94560; margin-top:0;">🎯 What is Network Intrusion Detection?</h3>
            <p style="color: #c0c0c0; line-height: 1.8;">
                Network Intrusion Detection uses <b style="color:#e94560;">machine learning</b> to analyze 
                network traffic and identify malicious activities by distinguishing 
                <b style="color:#00b894;">normal</b> and <b style="color:#ff6348;">abnormal</b> patterns. 
                Instead of manually defining rules, the ML model learns patterns from data to 
                automatically classify network connections.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #e94560; margin-top:0;">⚔️ Types of Attacks Detected</h3>
            <ul style="color: #c0c0c0; line-height: 2;">
                <li><b style="color:#ff6b6b;">DoS</b> — Denial of Service (flooding the server)</li>
                <li><b style="color:#ffa502;">Probe</b> — Scanning for vulnerabilities</li>
                <li><b style="color:#ff4757;">R2L</b> — Remote to Local (unauthorized remote access)</li>
                <li><b style="color:#ff6348;">U2R</b> — User to Root (privilege escalation)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="footer">
        🛡️ Network Intrusion Detection System • Machine Learning Project • KDD Cup 1999 • Random Forest Classifier
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  PAGE: DATASET EXPLORER
# ─────────────────────────────────────────────
elif page == "📊 Dataset Explorer":
    st.markdown('<h1 class="hero-title">📊 Dataset Explorer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Explore the KDD Cup 1999 network traffic dataset</p>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    @st.cache_data(show_spinner=False)
    def load_data():
        from model.preprocess import load_dataset, create_binary_label, create_multiclass_label
        df = load_dataset(percent10=True)
        df = create_binary_label(df)
        df = create_multiclass_label(df)
        return df
    
    with st.spinner("📥 Loading dataset... (first time may take a moment)"):
        df = load_data()
    
    # Dataset stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Features", "41")
    with col3:
        normal_pct = (df['binary_label'] == 0).mean() * 100
        st.metric("Normal Traffic", f"{normal_pct:.1f}%")
    with col4:
        attack_pct = (df['binary_label'] == 1).mean() * 100
        st.metric("Attack Traffic", f"{attack_pct:.1f}%")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📄 Raw Data", "📊 Distributions", "📈 Statistics"])
    
    with tab1:
        st.markdown('<div class="section-header">Sample Data (First 500 rows)</div>', unsafe_allow_html=True)
        display_cols = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 
                       'count', 'srv_count', 'attack_category', 'binary_label']
        st.dataframe(df[display_cols].head(500), use_container_width=True, height=400)
    
    with tab2:
        from model.evaluate import plot_class_distribution
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-header">Binary Classification</div>', unsafe_allow_html=True)
            fig = plot_class_distribution(df['binary_label'].values, "Normal vs Attack")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown('<div class="section-header">Attack Categories</div>', unsafe_allow_html=True)
            cat_counts = df['attack_category'].value_counts()
            import plotly.graph_objects as go
            fig = go.Figure(data=go.Bar(
                x=cat_counts.index,
                y=cat_counts.values,
                marker=dict(
                    color=['#0f3460', '#e94560', '#ffa502', '#ff4757', '#ff6348', '#aaa'],
                    line=dict(color='rgba(233,69,96,0.5)', width=1)
                )
            ))
            fig.update_layout(
                title=dict(text='Attack Category Distribution', font=dict(size=18, color='#e0e0e0')),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(26,26,46,0.5)',
                font=dict(color='#e0e0e0'),
                height=400
            )
            fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
            fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
            st.plotly_chart(fig, use_container_width=True)
        
        # Protocol distribution
        st.markdown('<div class="section-header">Protocol Type Distribution</div>', unsafe_allow_html=True)
        proto_counts = df['protocol_type'].value_counts()
        fig = go.Figure(data=go.Pie(
            labels=proto_counts.index,
            values=proto_counts.values,
            hole=0.4,
            marker=dict(
                colors=['#e94560', '#0f3460', '#ffa502'],
                line=dict(color='#16213e', width=2)
            ),
            textinfo='label+percent',
            textfont=dict(color='white')
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e0e0e0'),
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown('<div class="section-header">Feature Statistics</div>', unsafe_allow_html=True)
        # Convert feature columns to numeric (KDD data may load as bytes/objects)
        from model.preprocess import COLUMN_NAMES, CATEGORICAL_COLS
        stat_cols = [c for c in COLUMN_NAMES if c not in CATEGORICAL_COLS and c in df.columns]
        if stat_cols:
            stats_df = df[stat_cols].apply(pd.to_numeric, errors='coerce')
            st.dataframe(stats_df.describe().T.round(3), use_container_width=True, height=500)
        else:
            st.info("No numeric feature columns found to display statistics.")


# ─────────────────────────────────────────────
#  PAGE: MODEL TRAINING
# ─────────────────────────────────────────────
elif page == "🧠 Model Training":
    st.markdown('<h1 class="hero-title">🧠 Model Training</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Train the Random Forest classifier on KDD Cup 1999 data</p>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Hyperparameters
    st.markdown('<div class="section-header">⚙️ Hyperparameters</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        n_estimators = st.slider("Number of Trees", 10, 200, 100, 10)
    with col2:
        max_depth = st.slider("Max Depth", 5, 50, 20, 5)
    with col3:
        test_size = st.slider("Test Size (%)", 10, 40, 20, 5)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Algorithm explanation
    with st.expander("📖 How Random Forest Works", expanded=False):
        st.markdown("""
        <div class="glass-card">
            <ol style="color: #c0c0c0; line-height: 2;">
                <li><b style="color:#e94560;">Bootstrap Sampling</b> — Create random subsets of training data</li>
                <li><b style="color:#e94560;">Decision Tree Training</b> — Train a decision tree on each subset</li>
                <li><b style="color:#e94560;">Random Feature Selection</b> — Each tree uses random features</li>
                <li><b style="color:#e94560;">Ensemble Voting</b> — All trees vote → majority wins</li>
            </ol>
            <p style="color: #8892b0;">This reduces overfitting and improves generalization compared to a single decision tree.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Train button
    if st.button("🚀 Train Model", use_container_width=True, type="primary"):
        
        progress_bar = st.progress(0)
        status = st.empty()
        
        # Step 1: Load data
        status.info("📥 Step 1/5: Loading KDD Cup 99 dataset...")
        progress_bar.progress(10)
        
        from model.preprocess import load_dataset, preprocess_data, split_data, save_artifacts
        df = load_dataset(percent10=True)
        progress_bar.progress(25)
        
        # Step 2: Preprocess
        status.info("🔧 Step 2/5: Preprocessing data...")
        X, y, encoders, scaler = preprocess_data(df)
        progress_bar.progress(40)
        
        # Step 3: Split
        status.info("✂️ Step 3/5: Splitting train/test data...")
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size/100)
        progress_bar.progress(50)
        
        # Step 4: Train
        status.info(f"🧠 Step 4/5: Training Random Forest ({n_estimators} trees)... This may take a moment.")
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        
        start_time = time.time()
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        progress_bar.progress(85)
        
        # Step 5: Save
        status.info("💾 Step 5/5: Saving model and artifacts...")
        os.makedirs('saved_model', exist_ok=True)
        joblib.dump(model, 'saved_model/random_forest.pkl')
        save_artifacts(encoders, scaler)
        
        # Save test data
        np.savez(
            'saved_model/test_data.npz',
            X_test=X_test, y_test=y_test,
            X_train=X_train, y_train=y_train
        )
        progress_bar.progress(100)
        
        # Calculate metrics
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        
        status.success(f"✅ Training complete in {train_time:.2f} seconds!")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">📊 Training Results</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{test_acc:.2%}</div>
                <div class="metric-label">Test Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{train_acc:.2%}</div>
                <div class="metric-label">Train Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{train_time:.1f}s</div>
                <div class="metric-label">Training Time</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{n_estimators}</div>
                <div class="metric-label">Trees Used</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.balloons()
    
    elif model_exists:
        st.info("✅ A trained model already exists. You can retrain with different parameters or proceed to Evaluation.")


# ─────────────────────────────────────────────
#  PAGE: EVALUATION
# ─────────────────────────────────────────────
elif page == "📈 Evaluation":
    st.markdown('<h1 class="hero-title">📈 Model Evaluation</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Comprehensive performance analysis of the trained model</p>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    if not os.path.exists('saved_model/random_forest.pkl'):
        st.warning("⚠️ No trained model found. Please go to **Model Training** page first.")
    else:
        # Load model and test data
        model = joblib.load('saved_model/random_forest.pkl')
        test_data = np.load('saved_model/test_data.npz')
        X_test = test_data['X_test']
        y_test = test_data['y_test']
        
        # Generate predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        from model.evaluate import (
            get_all_metrics, get_classification_report,
            plot_confusion_matrix, plot_roc_curve, 
            plot_feature_importance, plot_metrics_radar
        )
        
        metrics = get_all_metrics(y_test, y_pred)
        
        # Metric cards
        col1, col2, col3, col4 = st.columns(4)
        metric_items = [
            ("Accuracy", metrics['accuracy']),
            ("Precision", metrics['precision']),
            ("Recall", metrics['recall']),
            ("F1-Score", metrics['f1_score'])
        ]
        for col, (label, value) in zip([col1, col2, col3, col4], metric_items):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{value:.2%}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Confusion Matrix", "📈 ROC Curve", "🌳 Feature Importance", "📋 Classification Report"])
        
        with tab1:
            col1, col2 = st.columns([1, 1])
            with col1:
                fig = plot_confusion_matrix(y_test, y_pred)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = plot_metrics_radar(metrics)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig, roc_auc = plot_roc_curve(y_test, y_prob)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"""
            <div class="glass-card" style="text-align:center;">
                <h3 style="color:#e94560; margin:0;">AUC Score: {roc_auc:.4f}</h3>
                <p style="color:#8892b0;">An AUC of 1.0 represents a perfect classifier. Values > 0.9 are considered excellent.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with tab3:
            fig = plot_feature_importance(model, top_n=15)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
            <div class="glass-card">
                <p style="color:#8892b0;">
                    Feature importance shows which network traffic attributes contribute most to distinguishing attacks from normal traffic.
                    Higher importance features are the primary indicators the model uses for intrusion detection.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with tab4:
            report = get_classification_report(y_test, y_pred)
            st.markdown('<div class="section-header">Classification Report</div>', unsafe_allow_html=True)
            st.code(report, language='text')


# ─────────────────────────────────────────────
#  PAGE: LIVE PREDICTION
# ─────────────────────────────────────────────
elif page == "🔍 Live Prediction":
    st.markdown('<h1 class="hero-title">🔍 Live Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Upload CSV data to detect network intrusions in real-time</p>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    if not os.path.exists('saved_model/random_forest.pkl'):
        st.warning("⚠️ No trained model found. Please go to **Model Training** page first.")
    else:
        model = joblib.load('saved_model/random_forest.pkl')
        from model.preprocess import load_artifacts, COLUMN_NAMES, CATEGORICAL_COLS
        encoders, scaler = load_artifacts()
        
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #e94560; margin-top:0;">📤 Upload Network Traffic Data</h3>
            <p style="color: #8892b0;">
                Upload a CSV file containing network traffic records. The file should have 41 feature columns 
                matching the KDD Cup 1999 format. Column names should match:
                <code>duration, protocol_type, service, flag, src_bytes, dst_bytes, ...</code>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Upload CSV file", 
            type=['csv'],
            help="Upload a CSV file with network traffic features (41 columns)"
        )
        
        # Sample data download
        with st.expander("📥 Download Sample CSV Template"):
            st.markdown("Use this template to format your data correctly:")
            
            @st.cache_data
            def create_sample_csv():
                from model.preprocess import load_dataset
                df = load_dataset(percent10=True)
                sample = df.drop('label', axis=1).head(20)
                return sample
            
            try:
                sample_df = create_sample_csv()
                csv_data = sample_df.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download Sample CSV",
                    data=csv_data,
                    file_name="sample_network_traffic.csv",
                    mime="text/csv"
                )
                st.dataframe(sample_df.head(5), use_container_width=True)
            except Exception as e:
                st.error(f"Could not generate sample: {e}")
        
        if uploaded_file is not None:
            try:
                # Read uploaded CSV
                input_df = pd.read_csv(uploaded_file)
                
                st.markdown('<div class="section-header">📄 Uploaded Data Preview</div>', unsafe_allow_html=True)
                st.dataframe(input_df.head(10), use_container_width=True)
                st.caption(f"Rows: {len(input_df)} | Columns: {len(input_df.columns)}")
                
                # Check if columns match
                if len(input_df.columns) == 41:
                    input_df.columns = COLUMN_NAMES
                elif len(input_df.columns) == 42:
                    # Has label column — drop it
                    input_df = input_df.iloc[:, :41]
                    input_df.columns = COLUMN_NAMES
                elif set(COLUMN_NAMES).issubset(set(input_df.columns)):
                    input_df = input_df[COLUMN_NAMES]
                else:
                    st.error(f"❌ Expected 41 feature columns, got {len(input_df.columns)}. Please check your CSV format.")
                    st.stop()
                
                if st.button("🚀 Run Intrusion Detection", use_container_width=True, type="primary"):
                    with st.spinner("🔍 Analyzing network traffic..."):
                        # Encode categorical features
                        df_processed = input_df.copy()
                        for col in CATEGORICAL_COLS:
                            df_processed[col] = df_processed[col].astype(str)
                            known = set(encoders[col].classes_)
                            df_processed[col] = df_processed[col].apply(
                                lambda x: x if x in known else encoders[col].classes_[0]
                            )
                            df_processed[col] = encoders[col].transform(df_processed[col])
                        
                        X_input = df_processed.values.astype(np.float64)
                        X_input = scaler.transform(X_input)
                        
                        # Predict
                        predictions = model.predict(X_input)
                        probabilities = model.predict_proba(X_input)
                        
                        time.sleep(0.5)  # Visual delay for UX
                    
                    # Results
                    st.markdown('<div class="section-header">🎯 Detection Results</div>', unsafe_allow_html=True)
                    
                    n_normal = (predictions == 0).sum()
                    n_attack = (predictions == 1).sum()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{len(predictions)}</div>
                            <div class="metric-label">Total Records</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value" style="color: #00b894;">{n_normal}</div>
                            <div class="metric-label">Normal ✅</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{n_attack}</div>
                            <div class="metric-label">Attack 🚨</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Detailed results table
                    results_df = input_df.copy()
                    results_df['Prediction'] = ['🟢 Normal' if p == 0 else '🔴 Attack' for p in predictions]
                    results_df['Confidence'] = [f"{max(prob)*100:.1f}%" for prob in probabilities]
                    results_df['Attack_Probability'] = [f"{prob[1]*100:.1f}%" for prob in probabilities]
                    
                    # Show result columns first
                    display_cols = ['Prediction', 'Confidence', 'Attack_Probability'] + \
                                   ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes']
                    st.dataframe(results_df[display_cols], use_container_width=True, height=400)
                    
                    # Download results
                    csv_results = results_df.to_csv(index=False)
                    st.download_button(
                        label="⬇️ Download Results CSV",
                        data=csv_results,
                        file_name="intrusion_detection_results.csv",
                        mime="text/csv"
                    )
                    
                    # Summary visualization
                    if n_attack > 0:
                        st.markdown(f"""
                        <div class="glass-card" style="border-color: rgba(233, 69, 96, 0.5);">
                            <h3 style="color: #e94560; margin-top:0;">⚠️ Intrusions Detected!</h3>
                            <p style="color: #c0c0c0;">
                                <b>{n_attack}</b> out of <b>{len(predictions)}</b> network connections 
                                ({n_attack/len(predictions)*100:.1f}%) were classified as potential intrusions.
                                Review the flagged connections for further investigation.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="glass-card" style="border-color: rgba(0, 184, 148, 0.5);">
                            <h3 style="color: #00b894; margin-top:0;">✅ All Clear!</h3>
                            <p style="color: #c0c0c0;">
                                All <b>{len(predictions)}</b> network connections appear to be normal traffic. 
                                No intrusions detected.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                st.info("Please make sure your CSV has the correct format with 41 feature columns.")
