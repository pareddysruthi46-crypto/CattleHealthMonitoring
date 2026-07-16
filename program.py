import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lime.lime_tabular import LimeTabularExplainer

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Cattle Health Monitoring",
    page_icon="🐄",
    layout="wide"
)

st.title("🐄 Cattle Health Monitoring System")
st.markdown("---")

# -----------------------------
# SAMPLE DATASET
# -----------------------------
np.random.seed(42)

data = pd.DataFrame({
    "Body Temperature": np.random.normal(38.5,0.6,200),
    "Heart Rate": np.random.normal(72,8,200),
    "Respiration Rate": np.random.normal(28,4,200),
    "Rumination Time": np.random.normal(500,60,200),
    "Activity Level": np.random.normal(300,50,200),
    "Weight": np.random.normal(450,40,200),
    "Age": np.random.randint(2,12,200)
})

labels=[]

for i in range(len(data)):

    if data.loc[i,"Body Temperature"]>39.5 or data.loc[i,"Rumination Time"]<430:
        labels.append(2)

    elif data.loc[i,"Activity Level"]<240:
        labels.append(1)

    else:
        labels.append(0)

data["Health Status"]=labels

X=data.drop("Health Status",axis=1)
y=data["Health Status"]

scaler=StandardScaler()

X_scaled=scaler.fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

model=RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train,y_train)

# -----------------------------
# TABS
# -----------------------------
tab1,tab2,tab3=st.tabs([
    "📊 Input Parameters",
    "🔬 Prediction Results",
    "📈 Model Insights"
])

# -----------------------------
# TAB 1
# -----------------------------
with tab1:

    colA,colB=st.columns([1,2])

    with colA:
        st.image(
            "https://img.icons8.com/color/96/000000/cow.png",
            width=140
        )

    with colB:
        st.subheader("Enter Cattle Physiological Parameters")
        st.info("Fill all values and click Predict.")

    with st.form("input_form"):

        c1,c2,c3=st.columns(3)

        with c1:

            st.markdown("### 🌡️ Vital Signs")

            body_temp=st.number_input(
                "Body Temperature (°C)",
                35.0,
                42.0,
                38.5,
                step=0.1
            )

            heart_rate=st.number_input(
                "Heart Rate (bpm)",
                40,
                120,
                72
            )

            resp_rate=st.number_input(
                "Respiration Rate",
                10,
                60,
                28
            )

        with c2:

            st.markdown("### 🐄 Behaviour")

            rumination=st.number_input(
                "Rumination Time",
                300,
                700,
                500
            )

            activity=st.number_input(
                "Activity Level",
                100,
                500,
                300
            )

        with c3:

            st.markdown("### 📏 Physical")

            weight=st.number_input(
                "Weight",
                300,
                700,
                450
            )

            age=st.number_input(
                "Age",
                1,
                15,
                5
            )

        submitted=st.form_submit_button(
            "🔍 Predict Health Status",
            use_container_width=True
        )
        # -----------------------------
# TAB 2
# -----------------------------
with tab2:

    if submitted:

        with st.spinner("Analyzing cattle health..."):
            time.sleep(2)

            input_data = np.array([[
                body_temp,
                heart_rate,
                resp_rate,
                rumination,
                activity,
                weight,
                age
            ]])

            input_scaled = scaler.transform(input_data)

            prediction = model.predict(input_scaled)[0]
            probabilities = model.predict_proba(input_scaled)[0]

        status_map = {
            0: "Healthy",
            1: "At Risk",
            2: "Diseased"
        }

        status_icon = {
            0: "🟢",
            1: "🟡",
            2: "🔴"
        }

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Health Status",
                f"{status_icon[prediction]} {status_map[prediction]}"
            )

        with c2:

            risk = (
                0.35 * min(body_temp / 40, 1)
                + 0.20 * max((72 - heart_rate) / 72, 0)
                + 0.15 * max((500 - rumination) / 500, 0)
                + 0.15 * max((300 - activity) / 300, 0)
                + 0.10 * (resp_rate / 40)
                + 0.05 * (age / 10)
            )

            risk = min(risk, 1) * 100

            st.metric(
                "Health Risk Score",
                f"{risk:.1f}%"
            )

        with c3:

            confidence = max(probabilities) * 100

            st.metric(
                "Prediction Confidence",
                f"{confidence:.1f}%"
            )

        st.markdown("---")

        st.subheader("Prediction Probabilities")

        p1, p2, p3 = st.columns(3)

        with p1:
            st.write("Healthy")
            st.progress(float(probabilities[0]))
            st.write(f"{probabilities[0]*100:.1f}%")

        with p2:
            st.write("At Risk")
            st.progress(float(probabilities[1]))
            st.write(f"{probabilities[1]*100:.1f}%")

        with p3:
            st.write("Diseased")
            st.progress(float(probabilities[2]))
            st.write(f"{probabilities[2]*100:.1f}%")

        st.markdown("---")

        st.subheader("Input Summary")

        s1, s2, s3, s4 = st.columns(4)

        with s1:
            st.metric("Temperature", f"{body_temp} °C")
            st.metric("Heart Rate", f"{heart_rate} bpm")

        with s2:
            st.metric("Respiration", f"{resp_rate}/min")
            st.metric("Rumination", f"{rumination} min")

        with s3:
            st.metric("Activity", activity)
            st.metric("Weight", f"{weight} kg")

        with s4:
            st.metric("Age", f"{age} years")
        st.markdown("---")

        # ==========================
        # SHAP FEATURE IMPORTANCE
        # ==========================
        with st.expander("🔎 SHAP Feature Importance", expanded=True):

            st.subheader("Global Feature Importance")

            with st.spinner("Calculating SHAP values..."):

                explainer = shap.TreeExplainer(model)

                shap_values = explainer.shap_values(X_train)

                fig, ax = plt.subplots(figsize=(10,6))

                shap.summary_plot(
                    shap_values,
                    X_train,
                    feature_names=X.columns,
                    show=False
                )

                plt.tight_layout()

                st.pyplot(fig)

                plt.close()

                st.success("SHAP explanation generated successfully.")

        st.markdown("---")

        # ==========================
        # LIME EXPLANATION
        # ==========================
        with st.expander("🧠 LIME Local Explanation", expanded=True):

            st.subheader("Why did the model predict this result?")

            lime_explainer = LimeTabularExplainer(

                training_data=X_train,

                feature_names=list(X.columns),

                class_names=[
                    "Healthy",
                    "At Risk",
                    "Diseased"
                ],

                mode="classification"
            )

            explanation = lime_explainer.explain_instance(

                input_scaled[0],

                model.predict_proba,

                num_features=7

            )

            lime_results = []

            for feature, weight in explanation.as_list():

                lime_results.append({

                    "Feature": feature,

                    "Impact": round(weight,4),

                    "Direction":
                        "Positive"
                        if weight > 0
                        else "Negative"

                })

            lime_df = pd.DataFrame(lime_results)

            st.dataframe(
                lime_df,
                use_container_width=True
            )

            fig, ax = plt.subplots(figsize=(9,4))

            colors = [
                "green" if x > 0 else "red"
                for x in lime_df["Impact"]
            ]

            ax.barh(

                lime_df["Feature"],

                lime_df["Impact"],

                color=colors

            )

            ax.set_xlabel("Impact")

            ax.set_title("LIME Feature Contributions")

            ax.axvline(
                x=0,
                color="black"
            )

            plt.tight_layout()

            st.pyplot(fig)

            plt.close()

        st.markdown("---")

        st.balloons()

        st.success("✅ Prediction Completed Successfully!")

    else:

        st.info(
            "Fill the values in Input Parameters tab and click Predict."
        )
        # -----------------------------
# TAB 3
# -----------------------------
with tab3:

    st.header("📈 Model Performance Insights")

    col1, col2, col3 = st.columns(3)

    with col1:
        train_acc = model.score(X_train, y_train)
        st.metric(
            "Training Accuracy",
            f"{train_acc:.2%}"
        )

    with col2:
        test_acc = model.score(X_test, y_test)
        st.metric(
            "Testing Accuracy",
            f"{test_acc:.2%}"
        )

    with col3:
        st.metric(
            "Model",
            "Random Forest"
        )

        st.metric(
            "Trees",
            "100"
        )

    st.markdown("---")

    st.subheader("🌲 Random Forest Feature Importance")

    feature_importance = pd.DataFrame({

        "Feature": X.columns,

        "Importance": model.feature_importances_

    }).sort_values(
        by="Importance",
        ascending=False
    )

    left, right = st.columns(2)

    with left:

        st.dataframe(

            feature_importance,

            use_container_width=True

        )

    with right:

        fig, ax = plt.subplots(figsize=(8,5))

        ax.barh(

            feature_importance["Feature"],

            feature_importance["Importance"]

        )

        ax.set_xlabel("Importance")

        ax.set_title("Feature Importance")

        plt.tight_layout()

        st.pyplot(fig)

        plt.close()

    st.markdown("---")

    st.subheader("📊 Dataset Statistics")

    a, b, c, d = st.columns(4)

    with a:
        st.metric(
            "Samples",
            len(data)
        )

    with b:
        st.metric(
            "Features",
            len(X.columns)
        )

    with c:
        st.metric(
            "Healthy",
            len(data[data["Health Status"] == 0])
        )

    with d:
        st.metric(
            "At Risk / Diseased",
            len(data[data["Health Status"] != 0])
        )

    st.markdown("---")

    st.subheader("Health Status Distribution")

    class_dist = (
        data["Health Status"]
        .value_counts()
        .sort_index()
    )

    class_dist.index = [

        "Healthy",

        "At Risk",

        "Diseased"

    ]

    fig, ax = plt.subplots(figsize=(8,4))

    ax.bar(

        class_dist.index,

        class_dist.values

    )

    ax.set_ylabel("Count")

    ax.set_title("Class Distribution")

    for i, v in enumerate(class_dist.values):

        ax.text(

            i,

            v + 2,

            str(v),

            ha="center"

        )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close()

    st.markdown("---")

    st.subheader("Dataset Preview")

    st.dataframe(

        data.head(15),

        use_container_width=True

    )
    # -----------------------------
# FOOTER
# -----------------------------

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;">
        <h4>🐄 Cattle Health Monitoring System</h4>
        <p>Machine Learning Based Health Prediction using Random Forest</p>
        <p>Developed using <b>Python • Streamlit • Scikit-Learn • SHAP • LIME</b></p>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.title("🐄 Cattle Health")

st.sidebar.markdown("### About Project")

st.sidebar.info(
"""
This application predicts the health condition of cattle using
Machine Learning.

Health Classes:

🟢 Healthy

🟡 At Risk

🔴 Diseased
"""
)

st.sidebar.markdown("---")

st.sidebar.markdown("### Model Information")

st.sidebar.write("✔ Random Forest Classifier")

st.sidebar.write("✔ SHAP Explainability")

st.sidebar.write("✔ LIME Explainability")

st.sidebar.write("✔ Streamlit Dashboard")

st.sidebar.markdown("---")

st.sidebar.success("Python 3.11 Compatible")

# -----------------------------
# END
# -----------------------------