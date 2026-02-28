import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random

from model_building import data_transformation_for_model, model_building

st.set_page_config(page_title="Cyber Attack Detection", layout="wide")


page = st.sidebar.selectbox("Navigation", ["Home", "Prediction Interface", "Data Dashboard"])


if page == "Home":

    st.markdown("""
    <style>

    /* ===== HERO ===== */
    .hero-title {
        font-size: 58px;
        font-weight: 900;
        text-align: center;
        color: #FFFFFF;
        margin-bottom: 10px;
    }

    .subtitle {
        font-size: 24px;
        text-align: center;
        color: #C9D1D9;
        margin-bottom: 40px;
    }

    

    .section-title {
        font-size: 26px;
        font-weight: 700;
        margin-bottom: 20px;
        color: #58A6FF;
    }


    

    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,3,1])

    with col2:
        st.image("logo.png", width=420)

    st.markdown("""
        <div class="hero-title">
            Cyber Attack Detection Platform
        </div>
    """, unsafe_allow_html=True)
    st.markdown("""
        <div class="subtitle">
            Real-time anomaly detection and intelligent threat classification powered by Machine Learning.
    </div>
    """, unsafe_allow_html=True)


    st.markdown("""

    <!-- Load Font Awesome -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">

    <style>
    .link-container {
        display: flex;
        gap: 20px; /* space between buttons */
        justify-content: center; /* center horizontally */
        margin-bottom: 20px;
    }

    .link-btn {
        padding: 7px 10px;
        background-color: #1A4F72;
        border-radius: 10px;
        text-decoration: none !important;
        color: white;
        font-weight: 600;
        text-align: center;
        transition: 0.2s;
    }

    .link-btn:hover {
        background-color: #234257;
        text-decoration: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="link-container">
        <a class="link-btn" href="https://github.com/xbdxllxhi2/A25-machine-learning-project" target="_blank">
            <i class="fab fa-github"></i>
            GitHub Repository 
        </a>
        <a class="link-btn" href="https://drive.google.com/file/d/1QPxSA8fiMSFcmEWQGAJ68H_hcOR8nzyP/view?usp=sharing" target="_blank">📄 Project Report</a>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    st.markdown("""
        <style>
        .section-title {
         font-size: 26px;
         font-weight: 700;
         margin-bottom: 20px;
         color: #58A6FF;
         margin-left: 60px;  /* moves the title to the right */
     }
        .team-columns {
        margin-left: 200px;  /* move team section to the right */
        gap: 40px;          /* spacing between two columns */
        display: flex;
     }

    .member {
        padding: 8px 0;
        font-size: 16px;
     }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">👥 Team</div>', unsafe_allow_html=True)

    # Wrap the two columns in a div for left margin
    st.markdown("""
    <div class="team-columns">
        <div style="flex:1;">
            <div class="member">⚙️ Vu Anh Dinh — Data Engineer</div>
            <div class="member">⚙️ Shayma Ridene — Data Engineer</div>
            <div class="member">⚙️ Wassim Elmoufakkir — Data Engineer</div>
            <div class="member">⚙️ Abdellahi Abdellahi — Data Engineer</div>
        </div>
        <div style="flex:1;">
            <div class="member">🧠 Lina Thuresson — Data Analyst</div>
            <div class="member">🧠 Sai Aditya Lakkum — Data Scientist</div>
            <div class="member">🧠 Tuan Nam Pham — Data Scientist</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
        
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    f1, f2, f3 = st.columns(3)

    f1.success("🧠 ML-based Attack Classification")
    f2.info("📊 SOC-style Monitoring Dashboard")
    f3.warning("⚡ Real-time Threat Insights")

# PAGE 1 
if page == "Prediction Interface":
    
    st.title("Cybersecurity Attack Type Detection ")
    
    # CSV Upload 
    st.divider()
    st.header(" Batch Prediction via CSV")
    
    uploaded_file = st.file_uploader("Upload attack records CSV", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state["df"] = df  # store dataframe safely
            st.success("File uploaded successfully!")
            st.dataframe(df.head())
        except Exception:
            st.error("Invalid CSV file.")
            st.stop()

    st.divider()

    if st.button("Predict Attack Type"):

        if "df" not in st.session_state:
            st.warning("Please upload a CSV file first.")
            st.stop()

        df = st.session_state["df"]
        mal, ddos, intru = data_transformation_for_model(df)
        malware, ddos, intrusion, combined, cm, data_results = model_building(mal, 
                       ddos, 
                       intru, 
                       'preprocessor_pca.pkl', 
                       'best_mlp_model_2.pkl', 
                       'pca_model.pkl', 
                       'label_encoder_pca.pkl'
            )
        
        st.subheader("Model Results")
        df = pd.DataFrame(data_results)

        st.success(f"We get a model accuracy of {round(data_results['Accuracy'][3]*100)}% with F1 score {round(data_results['F1-Score'][3]*100)}%.")
        st.dataframe(df)

        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        st.subheader("Accuracy Comparison")

        classes = ['Malware', 'DDoS', 'Intrusion', 'Combined']
        # accuracies data_results['Accuracy'][malware_acc, ddos_acc, intrusion_acc, combined_acc]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        bars = axes[0].bar(classes, data_results['Accuracy'], color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        axes[0].set_ylabel('Accuracy', fontsize=6, fontweight='bold')
        axes[0].set_title('Model Accuracy by Attack Type', fontsize=7, fontweight='bold')
        axes[0].set_ylim([0, 1])
        axes[0].grid(axis='y', alpha=0.1)

        for bar in bars:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2%}', ha='center', va='bottom', fontweight='bold', fontsize=8)

        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Malware', 'DDoS', 'Intrusion'],
                    yticklabels=['Malware', 'DDoS', 'Intrusion'],
                    cbar_kws={'label': 'Count'},
                    ax=axes[1])
        axes[1].set_title('Confusion Matrix', fontsize=7, fontweight='bold')
        axes[1].set_ylabel('True Label', fontsize=5)
        axes[1].set_xlabel('Predicted Label', fontsize=5)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)


# PAGE 2 
elif page == "Data Dashboard":
    st.title("Uploaded Data Analysis Dashboard")

    if "df" not in st.session_state:
        st.warning("Please upload a CSV file in the Prediction page first.")
    else:
        df = st.session_state["df"]

        
        # OVERVIEW METRICS
        st.subheader("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Records", len(df))
        col2.metric("Features", df.shape[1])
        col3.metric("Attack Classes", df["target"].nunique())

        st.divider()

        # TARGET DISTRIBUTION
        st.subheader("Attack Class Distribution")
        st.bar_chart(df["target"].value_counts())

        # PROTOCOL DISTRIBUTION
        st.subheader("Protocol Usage")
        st.bar_chart(df["protocol"].value_counts())

        
        # PACKET LENGTH DISTRIBUTION
        st.subheader("Packet Length Distribution")

        fig1, ax1 = plt.subplots(figsize=(18,6))
        sns.histplot(df["packet_length"], kde=True, ax=ax1)
        ax1.set_title("Packet Length Frequency")
        st.pyplot(fig1)

        # ANOMALY SCORE DISTRIBUTION
        st.subheader("Anomaly Score Distribution")

        fig2, ax2 = plt.subplots(figsize=(18,6))
        sns.histplot(df["anomaly_scores"], kde=True, ax=ax2)
        ax2.set_title("Anomaly Score Spread")
        st.pyplot(fig2)

        # SEVERITY DISTRIBUTION
        st.subheader("Severity Distribution")
        st.bar_chart(df["severity_encoded"].value_counts())

        # CORRELATION HEATMAP
        st.subheader("Feature Correlation Heatmap")

        numeric_df = df.select_dtypes(include=['int64', 'float64'])

        fig3, ax3 = plt.subplots(figsize=(24,9))
        sns.heatmap(numeric_df.corr(), cmap="coolwarm", annot=False)
        st.pyplot(fig3)
    
