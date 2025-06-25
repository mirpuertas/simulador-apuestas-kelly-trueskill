import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kelly import simulate, performance_stats, compute_result

def main():
    st.set_page_config(page_title="Simulador de apuestas de Kelly", layout="wide")
    st.title("Simulador de apuestas de Kelly")

    # Sidebar layout
    with st.sidebar:
        st.header("Parámetros de simulación")
        
        uploaded_file = st.file_uploader("Elegí un archivo CSV", type="csv")
        if uploaded_file is None:
            st.warning("Subí un archivo CSV con predicciones para comenzar.")
            st.stop()

        df = pd.read_csv(uploaded_file, parse_dates=["date"])

        with st.expander("Parámetros Básicos", expanded=True):
            bankroll = st.number_input("Bankroll Inicial", value=1000.0, step=100.0)
            kelly_fraction = st.number_input("Kelly Fraction", value=0.01, step=0.01, format="%.4f")
            risk_cap = st.number_input("Risk Cap", value=0.03, step=0.01, format="%.4f")

            stake_mode = st.selectbox("Stake Mode", ["kelly", "frac", "flat", "edge"], index=0)
            bet_mode = st.selectbox("Bet Mode", ["winner", "loser", "best", "both"], index=1)

            graph_type = st.radio("Tipo de gráfico", ["linear", "log", "ambos"], index=2)
            adjust_vig = st.checkbox("Ajustar Vig", value=True)

        with st.expander("Parámetros Avanzados", expanded=False):
            min_edge = st.number_input("Edge mínimo", value=0.01, step=0.01, format="%.4f")
            beta = st.number_input("Beta", value=1.0, step=0.1)
            sigma_thresh = st.number_input("Sigma (Umbral)", value=2.0, step=0.1)

    if st.button("Simular"):
        try:
            df = df.sort_values("date")
            df = df.dropna(subset=["m_winner", "m_loser", "s_winner", "s_loser", "b_winner", "b_loser"])
            df[["b_winner", "b_loser"]] = df[["b_winner", "b_loser"]].apply(pd.to_numeric, errors="coerce")
            df["result"] = df.apply(compute_result, axis=1)
            accuracy = (df["result"].sum() / len(df)) * 100

            log = simulate(
                df,
                bankroll,
                kelly_fraction,
                risk_cap,
                stake_mode,
                bet_mode,
                adjust_vig,
                min_edge,
                sigma_thresh=sigma_thresh,
                beta=beta
            )
            log["date"] = pd.to_datetime(log["date"])
            log = log.sort_values("date")

            stats = performance_stats(log)

            col1, col2 = st.columns([1, 2])

            with col1:
                st.subheader("Estadísticas de Performance")
                col_a, col_b = st.columns(2)
                keys = list(stats.keys())
                for i in range(0, len(keys), 2):
                    col_a.metric(keys[i], f"{stats[keys[i]]:.4f}")
                    if i + 1 < len(keys):
                        col_b.metric(keys[i + 1], f"{stats[keys[i + 1]]:.4f}")

                st.markdown("---")
                st.subheader("Accuracy de la Predicción")
                st.metric("Accuracy del modelo", f"{accuracy:.2f}%")

                st.markdown("---")
                st.subheader("Información de Apuestas")
                st.metric("Total de apuestas realizadas", log.attrs["placed"])
                st.metric("Total de apuestas salteadas", log.attrs["skipped"])

            with col2:
                st.subheader("Evolución del Bankroll")
                fig = None

                if graph_type == "linear":
                    fig, ax = plt.subplots(figsize=(7, 4))
                    ax.plot(log.date, log.bankroll, color="tab:blue")
                    ax.set_title("Bankroll (escala lineal)")
                    ax.set_ylabel("Bankroll")
                    fig.tight_layout()
                    st.pyplot(fig)

                elif graph_type == "log":
                    fig, ax = plt.subplots(figsize=(7, 4))
                    ax.plot(log.date, log.bankroll, color="tab:green")
                    ax.set_yscale("log")
                    ax.set_title("Bankroll (escala logarítmica)")
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Bankroll (log)")
                    fig.tight_layout()
                    st.pyplot(fig)

                elif graph_type == "ambos":
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
                    ax1.plot(log.date, log.bankroll, color="tab:blue")
                    ax1.set_title("Bankroll (escala lineal)")
                    ax1.set_ylabel("Bankroll")


                    ax2.plot(log.date, log.bankroll, color="tab:green")
                    ax2.set_yscale("log")
                    ax2.set_title("Bankroll (escala logarítmica)")
                    ax2.set_ylabel("Bankroll (log)")
                    ax2.set_xlabel("Date")


                    fig.tight_layout()
                    st.pyplot(fig)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
