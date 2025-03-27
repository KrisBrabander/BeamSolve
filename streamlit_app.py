import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os
import base64
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
import plotly.io as pio
from io import BytesIO


def calculate_reactions(beam_length, supports, loads):
    """Bereken reactiekrachten voor verschillende steunpuntconfiguraties."""
    if not supports or not loads:
        return {}

    # Sorteer steunpunten
    supports = sorted(supports, key=lambda x: x[0])
    n = len(supports)

    reactions = {}

    try:
        if n == 1:
            pos, type = supports[0]
            if type.lower() != "inklemming":
                st.error("❌ Systeem met één steunpunt moet een inklemming zijn")
                return None

            V_total = 0
            M_total = 0

            for load in loads:
                pos_load, value, load_type, *rest = load
                if load_type.lower() == "puntlast":
                    V_total += value
                    M_total += value * (pos_load - pos)
                elif load_type.lower() == "verdeelde last":
                    length = rest[0]
                    q = value
                    V_total += q * length
                    x_c = pos_load + length/2
                    M_total += q * length * (x_c - pos)
                elif load_type.lower() == "moment":
                    M_total += value

            reactions[pos] = -V_total
            reactions[f"M_{pos}"] = -M_total

        elif n == 2:
            x1, type1 = supports[0]
            x2, type2 = supports[1]
            L = x2 - x1

            if L == 0:
                st.error("❌ Steunpunten mogen niet op dezelfde positie liggen")
                return None

            V_total = 0
            M_1 = 0

            for load in loads:
                pos, value, load_type, *rest = load
                if load_type.lower() == "puntlast":
                    V_total += value
                    M_1 += value * (pos - x1)
                elif load_type.lower() == "verdeelde last":
                    length = rest[0]
                    q = value
                    Q = q * length
                    x_c = pos + length/2
                    V_total += Q
                    M_1 += Q * (x_c - x1)
                elif load_type.lower() == "moment":
                    M_1 += value

            R2 = M_1 / L
            R1 = V_total - R2

            reactions[x1] = -R1
            reactions[x2] = -R2

            if type1.lower() == "inklemming" and type2.lower() == "inklemming":
                reactions[f"M_{x1}"] = -M_1/2
                reactions[f"M_{x2}"] = M_1/2
            elif type1.lower() == "inklemming":
                reactions[f"M_{x1}"] = -M_1
            elif type2.lower() == "inklemming":
                reactions[f"M_{x2}"] = M_1

        else:
            V_total = 0
            M_total = 0
            x_ref = supports[0][0]

            for load in loads:
                pos, value, load_type, *rest = load
                if load_type.lower() == "puntlast":
                    V_total += value
                    M_total += value * (pos - x_ref)
                elif load_type.lower() == "verdeelde last":
                    length = rest[0]
                    q = value
                    Q = q * length
                    x_c = pos + length/2
                    V_total += Q
                    M_total += Q * (x_c - x_ref)
                elif load_type.lower() == "moment":
                    M_total += value

            if n == 3:
                x1, _ = supports[0]
                x2, _ = supports[1]
                x3, _ = supports[2]
                L1 = x2 - x1
                L2 = x3 - x2
                L = x3 - x1

                R2 = V_total * 0.5
                R3 = (M_total - R2*(x2 - x_ref)) / (x3 - x_ref)
                R1 = V_total - R2 - R3

                reactions[x1] = -R1
                reactions[x2] = -R2
                reactions[x3] = -R3
            else:
                R = V_total / n
                for pos, _ in supports:
                    reactions[pos] = -R

    except Exception as e:
        st.error(f"❌ Fout bij berekenen reactiekrachten: {str(e)}")
        return None

    return reactions


def plot_beam_diagram(beam_length, supports, loads):
    """Teken professioneel balkschema."""
    fig = go.Figure()

    colors = {
        'beam': '#2c3e50',
        'support': '#3498db',
        'load': '#e74c3c',
        'background': '#ffffff',
        'grid': '#ecf0f1'
    }

    fig.add_trace(go.Scatter(
        x=[0, beam_length],
        y=[0, 0],
        mode='lines',
        line=dict(color=colors['beam'], width=6),
        name='Balk'
    ))

    for pos, type in supports:
        x_pos = pos / 1000  # Convert to meters
        triangle_size = beam_length / 50

        if type.lower() == "inklemming":
            fig.add_trace(go.Scatter(
                x=[x_pos, x_pos, x_pos + triangle_size / 1000, x_pos + triangle_size / 1000, x_pos],
                y=[-triangle_size / 1000, triangle_size / 1000, triangle_size / 1000, -triangle_size / 1000, -triangle_size / 1000],
                fill="toself",
                mode='lines',
                line=dict(color=colors['support'], width=2),
                fillcolor=colors['support'],
                opacity=0.3,
                name='Inklemming'
            ))

    for load in loads:
        x_pos = load[0] / 1000
        value = load[1]
        load_type = load[2]

        if load_type.lower() == "puntlast":
            arrow_height = beam_length / 25
            fig.add_trace(go.Scatter(
                x=[x_pos],
                y=[arrow_height / 1000 + arrow_height / 4000],
                mode='text',
                text=[f'{value / 1000:.1f} kN'],
                textposition='top center',
                textfont=dict(size=14, color=colors['load']),
                showlegend=False
            ))

            fig.add_trace(go.Scatter(
                x=[x_pos, x_pos],
                y=[arrow_height / 1000, 0],
                mode='lines',
                line=dict(color=colors['load'], width=3),
                showlegend=True,
                name='Puntlast'
            ))

            fig.add_shape(
                type="path",
                path=f"M {x_pos - arrow_height / 3000} {arrow_height / 3000} L {x_pos} 0 L {x_pos + arrow_height / 3000} {arrow_height / 3000} Z",
                fillcolor=colors['load'],
                line=dict(color=colors['load'], width=0),
            )

    return fig


def main():
    st.set_page_config(page_title="BeamSolved", page_icon="🔧", layout="wide")
    
    st.title("BeamSolve Pro")
    st.markdown("#### Geavanceerde balkberekeningen voor constructeurs")
    
    with st.sidebar:
        st.header("Invoergegevens")
        
        beam_length = st.number_input("Overspanning [mm]", min_value=100.0, value=3000.0, step=100.0)
        
        profile_type = st.selectbox("Profieltype", ["Koker", "I-profiel", "HEA", "HEB", "IPE", "UNP"])
        
        height = st.number_input("Hoogte [mm]", value=100.0, min_value=10.0)
        width = st.number_input("Breedte [mm]", value=50.0, min_value=10.0)
        wall_thickness = st.number_input("Wanddikte [mm]", value=5.0, min_value=1.0)
        
        supports = [(0, "Inklemming"), (beam_length, "Scharnier")]
        
        loads = [(1500, 2000, "Puntlast"), (500, 1000, "Verdeelde last", 2000)]

        E = st.number_input("E-modulus [N/mm²]", value=210000.0, min_value=1000.0, step=1000.0)
    
    reactions = calculate_reactions(beam_length, supports, loads)
    
    beam_fig = plot_beam_diagram(beam_length, supports, loads)
    
    st.plotly_chart(beam_fig)

if __name__ == "__main__":
    main()
