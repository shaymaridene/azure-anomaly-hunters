import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Cyber Attack Detection", layout="wide")

st.sidebar.title(" Cyber Attack Detection System")
page = st.sidebar.selectbox("Go to", ["Prediction Interface", "Data Dashboard"])


# PAGE 1 =================
if page == "Prediction Interface":
    

    st.title("Cybersecurity Attack Type Detection ")
    
    st.header("Manual Input (Single Prediction)")

    net1, net2, net3 = st.columns(3)
    with net1:
        source_ip = st.text_input("Source IP")
        destination_ip = st.text_input("Destination IP")
    with net2:
        source_port = st.number_input("Source Port", 0, 65535)
        destination_port = st.number_input("Destination Port", 0, 65535)
        protocol = st.selectbox("Protocol", ["TCP", "UDP"])
    with net3:
        network_segment = st.text_input("Network Segment")
        geo_location = st.text_input("Geo-location")
        proxy_info = st.text_input("Proxy Information")

    p1, p2, p3 = st.columns(3)
    with p1:
        packet_length = st.number_input("Packet Length", min_value=0)
        packet_type = st.text_input("Packet Type")
    with p2:
        traffic_type = st.text_input("Traffic Type")
    with p3:
        firewall_logs = st.text_input("Firewall Logs")
        ids_ips_alerts = st.text_input("IDS/IPS Alerts")

    s1, s2, s3 = st.columns(3)
    with s1:
        malware_indicators = st.text_input("Malware Indicators")
        anomaly_scores = st.text_input("Anomaly Scores")
    with s2:
        alerts = st.text_input("Alerts / Warnings")
        severity = st.selectbox("Severity Level", ["Low", "Medium", "High", "Critical"])
    
        

    st.divider()

    colA, colB = st.columns([1,1])

    with colA:
        if st.button("🚀 Predict Attack Type"):
            fake_prediction = random.choice(["DDoS", "Phishing", "Malware", "Ransomware", "Normal Traffic"])
            st.success(f"Predicted Attack Type: **{fake_prediction}**")

    with colB:
        st.info("Model will be integrated here once training is complete.")

    # CSV Upload 
    st.divider()
    st.header(" Batch Prediction via CSV")
    
    uploaded_file = st.file_uploader("Upload attack records CSV", type="csv")

    if uploaded_file is not None:
        st.session_state["uploaded_file"] = uploaded_file
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
        st.dataframe(df.head())



# PAGE 2 
elif page == "Data Dashboard":
    st.title(" Uploaded Data Analysis Dashboard")

    if "uploaded_file" not in st.session_state:
        st.warning("Please upload a CSV file in the Prediction page first.")
    else:
        uploaded_file = st.session_state["uploaded_file"]
        df = pd.read_csv(uploaded_file)



        st.subheader("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Records", len(df))
        col2.metric("Features", df.shape[1])
        col3.metric("Attack Types", df["Attack Type"].nunique())

        st.divider()

        # ===== Graph 1 =====
        st.subheader("Attack Type Distribution")
        st.bar_chart(df["Attack Type"].value_counts())

        # ===== Graph 2 =====
        st.subheader("Protocol Usage")
        st.bar_chart(df["Protocol"].value_counts())

        # ===== Graph 3 =====
        st.subheader("Packet Length Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df["Packet Length"], kde=True, ax=ax)
        st.pyplot(fig)

        # ===== Graph 4 =====
        st.subheader("Severity Levels")
        st.bar_chart(df["Severity Level"].value_counts())

