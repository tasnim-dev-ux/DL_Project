import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random

# =========================
# Page configuration
# =========================
st.set_page_config(
    page_title="RF Signal Classifier",
    page_icon="📡",
    layout="wide"
)

# =========================
# Load model (PyTorch)
# =========================
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(0.2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(0.2),
            nn.Conv1d(128, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
            nn.Linear(256*8, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 3)  # 3 classes: Good, Average, Poor
        )

    def forward(self, x):
        return self.net(x)

model = MyModel()
model.load_state_dict(torch.load("best_model.pth", map_location=torch.device('cpu'), weights_only=True))
model.eval()

# =========================
# Sidebar
# =========================
st.sidebar.title("About")
st.sidebar.info(
"""
This application classifies **RF signals** based on the
I/Q components.

Steps:
1. Enter 128 I values (comma-separated)
2. Enter 128 Q values (comma-separated)
3. Click Predict

The model predicts network quality: Good, Average, or Poor.
"""
)

# Dataset test helper
st.sidebar.title("Dataset Example")
try:
    with open("Data/RML2016.10a_dict.pkl", "rb") as f:
        data_dict = pickle.load(f, encoding="latin1")
    modulations = sorted({k[0] for k in data_dict.keys()})
    modulation_choice = st.sidebar.selectbox("Modulation", modulations)
    snr_values = sorted({k[1] for k in data_dict.keys() if k[0] == modulation_choice})
    snr_choice = st.sidebar.selectbox("SNR", snr_values)
    if st.sidebar.button("Load dataset sample"):
        key = (modulation_choice, snr_choice)
        sample_signal = random.choice(data_dict[key])
        # sample_signal shape (2, 128)
        I_example = ",".join(f"{v:.5f}" for v in sample_signal[0])
        Q_example = ",".join(f"{v:.5f}" for v in sample_signal[1])
        st.session_state.I_input = I_example
        st.session_state.Q_input = Q_example
except FileNotFoundError:
    st.sidebar.warning("Dataset file Data/RML2016.10a_dict.pkl not found.")
except Exception as e:
    st.sidebar.error(f"Erreur chargement dataset: {e}")

# =========================
# Title
# =========================
st.title("📡 RF Signal Classification System")
st.write("Enter the **I and Q signal values** to predict the modulation type.")

# =========================
# Input layout
# =========================
if "I_input" not in st.session_state:
    st.session_state.I_input = ""
if "Q_input" not in st.session_state:
    st.session_state.Q_input = ""

if st.button("🔁 Generate random I/Q signal"):
    I_rand = np.random.uniform(-1, 1, 128)
    Q_rand = np.random.uniform(-1, 1, 128)
    st.session_state.I_input = ",".join(f"{x:.5f}" for x in I_rand)
    st.session_state.Q_input = ",".join(f"{x:.5f}" for x in Q_rand)
    st.success("Random sample loaded; app ready to predict.")

col1, col2 = st.columns(2)

with col1:
    I_input = st.text_area(
        "I Signal Values (128 samples, comma-separated)",
        placeholder="Example: 0.1,0.2,0.3,... (128 values)",
        value=st.session_state.I_input,
        height=180,
    )

with col2:
    Q_input = st.text_area(
        "Q Signal Values (128 samples, comma-separated)",
        placeholder="Example: 0.05,0.1,0.2,... (128 values)",
        value=st.session_state.Q_input,
        height=180,
    )

st.session_state.I_input = I_input
st.session_state.Q_input = Q_input

# =========================
# Prediction button
# =========================
if st.button("🔍 Predict Signal Type"):
    try:
        I = np.array(list(map(float, I_input.split(","))))
        Q = np.array(list(map(float, Q_input.split(","))))
        
        if len(I) != 128 or len(Q) != 128:
            st.warning("128 values attendus. Le signal sera ajusté automatiquement (troncature ou remplissage).")
            if len(I) < 128:
                I = np.pad(I, (0, 128 - len(I)), mode="constant", constant_values=0.0)
            elif len(I) > 128:
                I = I[:128]
            if len(Q) < 128:
                Q = np.pad(Q, (0, 128 - len(Q)), mode="constant", constant_values=0.0)
            elif len(Q) > 128:
                Q = Q[:128]

        # Stack I and Q into (1, 2, 128) tensor
        signal = np.stack([I, Q], axis=0)  # (2, 128)
        signal_tensor = torch.FloatTensor(signal).unsqueeze(0)  # (1, 2, 128)
        
        with torch.no_grad():
            outputs = model(signal_tensor)
            pred = torch.argmax(outputs, dim=1).item()
        
        classes = ["Good", "Average", "Poor"]  # 0=Good, 1=Average, 2=Poor
        st.success(f"Predicted Network Quality: **{classes[pred]}**")
        
        # =========================
        # Plot signals
        # =========================
        fig, ax = plt.subplots()
        ax.plot(I, label="I Signal")
        ax.plot(Q, label="Q Signal")
        ax.set_title("I/Q Signal Visualization")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Amplitude")
        ax.legend()
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"⚠ Please enter valid numeric values separated by commas.\nError: {e}")