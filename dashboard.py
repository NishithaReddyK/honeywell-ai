import streamlit as st
import os
import datetime
import glob
import pandas as pd

# ----------------- Dashboard Layout -----------------
st.set_page_config(page_title="AI Surveillance Dashboard", layout="wide")

st.title("🛡️ AI Surveillance Dashboard")
st.markdown("Real-time anomaly detection with YOLO + OpenCV")

# Sidebar for navigation
menu = st.sidebar.radio("📌 Navigation", ["Live Feed", "Alert History", "System Info"])

# ----------------- Live Feed -----------------
if menu == "Live Feed":
    st.subheader("🚨 Live Suspicious Activity Feed")
    st.write("Timestamp:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Get the latest saved alert image
    alert_images = sorted(glob.glob("outputs/alert_*.jpg"), reverse=True)

    if alert_images:
        latest_image = alert_images[0]
        st.image(latest_image, caption=f"Latest Alert: {os.path.basename(latest_image)}", use_column_width=True)
    else:
        st.warning("⚠️ No alert image found yet.")

# ----------------- Alert History -----------------
elif menu == "Alert History":
    st.subheader("📂 Past Alerts")

    alert_images = sorted(glob.glob("outputs/alert_*.jpg"), reverse=True)

    if alert_images:
        cols = st.columns(3)
        for i, img in enumerate(alert_images[:12]):  # Show last 12 alerts
            with cols[i % 3]:
                st.image(img, caption=os.path.basename(img), use_column_width=True)

        # Table view
        timestamps = [
            os.path.basename(img).replace("alert_", "").replace(".jpg", "") for img in alert_images
        ]
        df = pd.DataFrame({"Alert Image": alert_images, "Timestamp": timestamps})
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No past alerts recorded.")

# ----------------- System Info -----------------
elif menu == "System Info":
    st.subheader("ℹ️ System Information")
    st.markdown("""
    - **Model:** YOLOv5 Small (Pretrained)  
    - **Backend:** OpenCV + PyTorch  
    - **Features:** Loitering, Object Abandonment, Unusual Movements  
    - **Storage:** Saving alerts in `outputs/` directory  
    - **Dashboard:** Streamlit  
    """)

    st.success("✅ System running smoothly!")

# dashboard.py (append below your existing code)
import pandas as pd

csv_path = "outputs/alerts.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    anomaly_filter = st.selectbox(
        "Filter by anomaly type", ["All", "loitering", "abandoned_object"]
    )
    if anomaly_filter != "All":
        df = df[df["type"] == anomaly_filter]
    st.subheader("Recent Alerts")
    st.dataframe(df.sort_values("timestamp", ascending=False).head(50), use_container_width=True)
else:
    st.info("No alerts logged yet.")
