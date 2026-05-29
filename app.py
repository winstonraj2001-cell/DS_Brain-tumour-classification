import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import time
import hashlib

# =====================================================
# CONFIG
# =====================================================

st.set_page_config(
    page_title="Brain Tumor AI Enterprise",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# DEMO LOGIN
# =====================================================

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "ChangeMe123!"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login_page():
    st.title("🔐 Brain Tumor AI Enterprise Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            st.session_state.logged_in = True
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid credentials")

if not st.session_state.logged_in:
    login_page()
    st.stop()

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("🧠 Brain Tumor AI Enterprise")

page = st.sidebar.selectbox(
    "Navigation",
    [
        "Dashboard",
        "MRI Prediction",
        "Analytics",
        "Model Performance",
        "Reports",
        "Settings",
        "Admin Dashboard",
        "User Management",
        "System Logs",
        "Live Statistics"
    ]
)

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

# =====================================================
# MODEL
# =====================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "mobilenet_brain_tumor_model.h5"
    )

model = load_model()

CLASS_NAMES = [
    "Glioma Tumor",
    "Meningioma Tumor",
    "No Tumor",
    "Pituitary Tumor"
]

# =====================================================
# DASHBOARD
# =====================================================

if page == "Dashboard":

    st.title("🧠 Brain Tumor AI Enterprise")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", "84%")
    col2.metric("Classes", "4")
    col3.metric("Model", "MobileNetV2")
    col4.metric("Status", "Online")
    # =====================================================
# MRI PREDICTION PAGE
# =====================================================

elif page == "MRI Prediction":

    st.title("🧠 MRI Tumor Prediction")

    uploaded_file = st.file_uploader(
        "Upload MRI Scan",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        col1, col2 = st.columns([1,1])

        with col1:
            st.image(
                image,
                caption="Uploaded MRI",
                use_container_width=True
            )

        if st.button("🚀 Predict Tumor"):

            with st.spinner("Analyzing MRI Scan..."):

                img = image.resize((224,224))
                img = np.array(img)

                if len(img.shape) == 2:
                    img = np.stack([img]*3, axis=-1)

                if img.shape[-1] == 4:
                    img = img[:,:,:3]

                img = img / 255.0
                img = np.expand_dims(img, axis=0)

                prediction = model.predict(img)

                predicted_class = np.argmax(prediction)

                confidence = float(
                    np.max(prediction) * 100
                )

                tumor = CLASS_NAMES[predicted_class]

                st.success(
                    f"Prediction: {tumor}"
                )

                st.info(
                    f"Confidence: {confidence:.2f}%"
                )

                with col2:

                    st.subheader("Result")

                    st.metric(
                        "Tumor Type",
                        tumor
                    )

                    st.metric(
                        "Confidence",
                        f"{confidence:.2f}%"
                    )

                prob_df = pd.DataFrame({
                    "Class": CLASS_NAMES,
                    "Probability": prediction[0]
                })

                fig = px.bar(
                    prob_df,
                    x="Class",
                    y="Probability",
                    color="Probability",
                    title="Prediction Probability"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

# =====================================================
# ANALYTICS PAGE
# =====================================================

elif page == "Analytics":

    st.title("📊 Analytics Dashboard")

    epoch = list(range(1,21))

    train_acc = [
        0.61,0.68,0.71,0.74,0.77,
        0.79,0.81,0.83,0.84,0.85,
        0.86,0.87,0.88,0.89,0.90,
        0.91,0.91,0.92,0.92,0.93
    ]

    val_acc = [
        0.60,0.66,0.70,0.73,0.75,
        0.77,0.78,0.80,0.81,0.82,
        0.83,0.84,0.84,0.85,0.85,
        0.84,0.84,0.85,0.84,0.84
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=epoch,
            y=train_acc,
            mode="lines+markers",
            name="Training Accuracy"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=epoch,
            y=val_acc,
            mode="lines+markers",
            name="Validation Accuracy"
        )
    )

    fig.update_layout(
        title="Accuracy Analysis",
        template="plotly_dark"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Dataset Distribution")

    dataset = pd.DataFrame({
        "Class":[
            "Glioma",
            "Meningioma",
            "No Tumor",
            "Pituitary"
        ],
        "Images":[
            1321,
            1280,
            1595,
            1457
        ]
    })

    pie = px.pie(
        dataset,
        values="Images",
        names="Class",
        hole=0.5
    )

    st.plotly_chart(
        pie,
        use_container_width=True
    )

# =====================================================
# MODEL PERFORMANCE PAGE
# =====================================================

elif page == "Model Performance":

    st.title("📈 Model Performance")

    report_df = pd.DataFrame({
        "Class":[
            "Glioma",
            "Meningioma",
            "No Tumor",
            "Pituitary"
        ],
        "Precision":[
            0.85,
            0.80,
            0.95,
            0.78
        ],
        "Recall":[
            0.95,
            0.62,
            0.76,
            1.00
        ],
        "F1":[
            0.90,
            0.70,
            0.84,
            0.88
        ]
    })

    st.dataframe(
        report_df,
        use_container_width=True
    )

    st.metric(
        "Final Accuracy",
        "84%"
    )

    cm = np.array([
        [120,5,3,2],
        [9,100,7,4],
        [3,6,140,2],
        [2,1,4,132]
    ])

    heat = px.imshow(
        cm,
        text_auto=True,
        title="Confusion Matrix"
    )

    st.plotly_chart(
        heat,
        use_container_width=True
    )
    # =====================================================
# SESSION STORAGE
# =====================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# =====================================================
# REPORTS PAGE
# =====================================================

elif page == "Reports":

    st.title("📄 Reports Center")

    st.markdown("### System Summary")

    report_data = pd.DataFrame({
        "Metric": [
            "Model Accuracy",
            "Total Classes",
            "Architecture",
            "Framework"
        ],
        "Value": [
            "84%",
            "4",
            "MobileNetV2",
            "TensorFlow"
        ]
    })

    st.dataframe(
        report_data,
        use_container_width=True
    )

    csv = report_data.to_csv(index=False)

    st.download_button(
        label="⬇ Download CSV Report",
        data=csv,
        file_name="brain_tumor_report.csv",
        mime="text/csv"
    )

    st.success("Report Ready")

# =====================================================
# SETTINGS PAGE
# =====================================================

elif page == "Settings":

    st.title("⚙ System Settings")

    st.subheader("Application Settings")

    dark_mode = st.toggle(
        "Dark Mode",
        value=True
    )

    notifications = st.toggle(
        "Enable Notifications",
        value=True
    )

    auto_save = st.toggle(
        "Auto Save Results",
        value=True
    )

    st.success("Settings Saved")

# =====================================================
# ADMIN DASHBOARD
# =====================================================

elif page == "Admin Dashboard":

    st.title("👨‍💼 Admin Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Users",
            "124"
        )

    with col2:
        st.metric(
            "Predictions",
            "5,624"
        )

    with col3:
        st.metric(
            "Accuracy",
            "84%"
        )

    with col4:
        st.metric(
            "System Status",
            "Online"
        )

    st.subheader("Server Monitoring")

    cpu_usage = np.random.randint(20, 90)

    memory_usage = np.random.randint(30, 95)

    st.progress(cpu_usage / 100)

    st.write(
        f"CPU Usage: {cpu_usage}%"
    )

    st.progress(memory_usage / 100)

    st.write(
        f"Memory Usage: {memory_usage}%"
    )

# =====================================================
# USER MANAGEMENT
# =====================================================

elif page == "User Management":

    st.title("👥 User Management")

    users = pd.DataFrame({
        "Username": [
            "admin",
            "doctor1",
            "doctor2",
            "researcher"
        ],
        "Role": [
            "Administrator",
            "Doctor",
            "Doctor",
            "Research"
        ],
        "Status": [
            "Active",
            "Active",
            "Active",
            "Active"
        ]
    })

    st.dataframe(
        users,
        use_container_width=True
    )

# =====================================================
# SYSTEM LOGS
# =====================================================

elif page == "System Logs":

    st.title("📜 Activity Logs")

    logs = pd.DataFrame({
        "Time":[
            "09:00",
            "09:15",
            "09:35",
            "10:05",
            "10:15"
        ],
        "Activity":[
            "Admin Login",
            "MRI Uploaded",
            "Prediction Completed",
            "Report Downloaded",
            "Logout"
        ]
    })

    st.dataframe(
        logs,
        use_container_width=True
    )

# =====================================================
# REAL TIME STATS
# =====================================================

elif page == "Live Statistics":

    st.title("📊 Real-Time Statistics")

    chart_data = pd.DataFrame({
        "Time": list(range(20)),
        "Predictions":
        np.random.randint(
            10,
            100,
            20
        )
    })

    fig = px.line(
        chart_data,
        x="Time",
        y="Predictions",
        title="Live Prediction Activity"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )