import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Profielenbibliotheek
PROFIELEN = {
    "IPE 100": (100, 55, 4.1, 5.7),
    "IPE 120": (120, 64, 4.4, 6.3),
    "IPE 140": (140, 73, 4.7, 6.9),
}

def bereken_I(h, b, tw, tf):
    return (b*h**3 - (b - tw)*(h - 2*tf)**3) / 12

def analyseer_balk(L, a, F, E, I):
    x = np.linspace(0, L, 500)
    V = np.where(x >= a, -F, 0)
    M = np.where(x >= a, -F*(x - a), 0)
    theta = np.cumsum(M / (E * I) * np.gradient(x))
    y = np.cumsum(theta * np.gradient(x))
    return x, V, M, y

def plot_resultaten(x, V, M, y):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=V, name="Dwarskracht (V)", line=dict(color="green")))
    fig.add_trace(go.Scatter(x=x, y=M, name="Moment (M)", line=dict(color="red")))
    fig.add_trace(go.Scatter(x=x, y=y*1000, name="Doorbuiging (mm)", line=dict(color="blue")))
    fig.update_layout(title="Balkanalyse", xaxis_title="Positie (mm)")
    return fig

def main():
    st.title("🔧 Eenvoudige Balkcalculator")

    profiel = st.selectbox("Kies profiel", list(PROFIELEN.keys()))
    h, b, tw, tf = PROFIELEN[profiel]
    I = bereken_I(h, b, tw, tf)

    col1, col2 = st.columns(2)
    with col1:
        L = st.number_input("Balklengte (mm)", value=3000.0)
        a = st.number_input("Belastingpositie (mm)", value=1500.0)
    with col2:
        F = st.number_input("Belasting (N)", value=5000.0)
        E = st.number_input("Elasticiteitsmodulus (N/mm²)", value=210000.0)

    if st.button("Bereken"):
        x, V, M, y = analyseer_balk(L, a, F, E, I)
        fig = plot_resultaten(x, V, M, y)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Resultaten")
        st.metric("Maximale doorbuiging (mm)", f"{max(abs(y))*1000:.2f}")
        st.metric("Maximaal moment (Nmm)", f"{max(abs(M)):.0f}")
        st.metric("Maximale dwarskracht (N)", f"{max(abs(V)):.0f}")

if __name__ == "__main__":
    main()
