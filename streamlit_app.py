import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(layout="wide", page_title="BeamCAE 2025")

# Modern engineering software styling
st.markdown("""
<style>
    /* Modern engineering theme */
    .main {
        background-color: #1e1e1e;
        color: #e0e0e0;
    }
    .stApp {
        max-width: 1800px;
        margin: 0 auto;
    }
    
    /* Professional header */
    .stApp header {
        background-color: #1e1e1e;
        border-bottom: 1px solid #333;
    }
    
    /* Side panel styling */
    .css-1d391kg {
        background-color: #252525;
        border-right: 1px solid #333;
        padding: 1rem;
    }
    
    /* Input fields */
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input,
    .stSelectbox > div > div {
        background-color: #2d2d2d !important;
        color: #e0e0e0 !important;
        border: 1px solid #404040 !important;
        border-radius: 4px;
    }
    
    /* Containers */
    .element-container {
        background-color: #252525;
        border: 1px solid #333;
        border-radius: 6px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Modern metrics */
    [data-testid="stMetricValue"] {
        background: linear-gradient(90deg, #00b4d8 0%, #0077be 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2rem !important;
        font-weight: 600 !important;
    }
    
    /* Labels */
    label {
        color: #00b4d8;
        font-weight: 500;
        text-transform: uppercase;
        font-size: 0.8rem;
        letter-spacing: 0.5px;
    }
    
    /* Custom title */
    .app-title {
        background: linear-gradient(90deg, #00b4d8 0%, #0077be 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin: 1rem 0;
    }
    
    /* Custom sidebar */
    .sidebar-section {
        background-color: #252525;
        border: 1px solid #333;
        border-radius: 6px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Modern buttons */
    .stButton > button {
        background: linear-gradient(90deg, #00b4d8 0%, #0077be 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }
    
    /* Section headers */
    .section-header {
        color: #00b4d8;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Custom tabs */
    .stTabs > div > div > div {
        background-color: #252525;
        border-radius: 4px 4px 0 0;
    }
    .stTabs > div > div > div > button {
        color: #e0e0e0 !important;
    }
    .stTabs > div > div > div > button:hover {
        color: #00b4d8 !important;
    }
    .stTabs > div > div > div > button[data-baseweb="tab"][aria-selected="true"] {
        color: #00b4d8 !important;
        border-bottom: 2px solid #00b4d8;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'load_count' not in st.session_state:
    st.session_state.load_count = 0
if 'loads' not in st.session_state:
    st.session_state.loads = []

# App title
st.markdown('<p class="app-title">BeamCAE 2025 Enterprise</p>', unsafe_allow_html=True)

# Create two columns: sidebar and main content
sidebar = st.sidebar
main = st.container()

# Sidebar sections
with sidebar:
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<p class="section-header"> Profiel</p>', unsafe_allow_html=True)
    
    profile_type = st.selectbox(
        "Type",
        ["Koker", "I-profiel", "H-profiel"],
        help="Selecteer het type profiel"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        height = st.number_input("Hoogte (mm)", 10.0, 1000.0, 100.0, step=10.0)
    with col2:
        width = st.number_input("Breedte (mm)", 10.0, 1000.0, 50.0, step=10.0)
    
    col1, col2 = st.columns(2)
    with col1:
        wall_thickness = st.number_input("Wanddikte (mm)", 0.1, 50.0, 5.0, step=0.1)
    with col2:
        if profile_type in ["I-profiel", "H-profiel"]:
            flange_thickness = st.number_input("Flensdikte (mm)", 0.1, 50.0, 5.0, step=0.1)
        else:
            flange_thickness = None
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<p class="section-header"> Materiaal</p>', unsafe_allow_html=True)
    E = st.number_input(
        "E-modulus (N/mm²)",
        1000.0, 300000.0, 210000.0,
        step=1000.0,
        help="Elasticiteitsmodulus van het materiaal"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<p class="section-header"> Overspanning</p>', unsafe_allow_html=True)
    beam_length = st.number_input("Lengte (mm)", 100.0, 10000.0, 1000.0, step=100.0)
    support_count = st.selectbox("Aantal steunpunten", [1, 2, 3], help="Aantal steunpunten")
    st.markdown('</div>', unsafe_allow_html=True)

# Main content
with main:
    # Create tabs
    tab1, tab2, tab3 = st.tabs([" Analyse", " Resultaten", " Rapport"])
    
    with tab1:
        # Modern visualization container
        st.markdown('<div style="background-color: #252525; padding: 1.5rem; border-radius: 8px; border: 1px solid #333;">', unsafe_allow_html=True)
        
        # Create advanced visualization
        fig = plt.figure(figsize=(15, 8))
        gs = GridSpec(2, 3, figure=fig)
        
        # Main beam plot
        ax_beam = fig.add_subplot(gs[:, :2])
        ax_beam.set_facecolor('#1e1e1e')
        fig.patch.set_facecolor('#1e1e1e')
        
        # Grid styling
        ax_beam.grid(True, linestyle='--', alpha=0.2, color='#666666')
        ax_beam.spines['bottom'].set_color('#666666')
        ax_beam.spines['top'].set_color('#666666')
        ax_beam.spines['left'].set_color('#666666')
        ax_beam.spines['right'].set_color('#666666')
        ax_beam.tick_params(colors='#666666')
        
        # Plot beam
        ax_beam.plot([0, beam_length], [0, 0], '-', color='#00b4d8', linewidth=3, label='Balk')
        
        # Add supports
        supports = []
        if support_count == 1:
            pos = st.slider("Positie inklemming (mm)", 0.0, beam_length, 0.0, key="inklemming_pos")
            supports.append((pos, "Inklemming"))
            # Draw fixed support
            rect_height = beam_length * 0.04
            rect = patches.Rectangle((pos-2, -rect_height/2), 4, rect_height,
                                  color='#0077be', alpha=0.8, zorder=4)
            ax_beam.add_patch(rect)
        else:
            for i in range(support_count):
                pos = st.slider(
                    f"Positie steunpunt {i+1} (mm)",
                    0.0, beam_length,
                    value=i * beam_length/(support_count-1) if support_count > 1 else 0.0,
                    key=f"support_pos_{i}"
                )
                type = st.selectbox(
                    "Type",
                    ["Scharnier", "Rol"],
                    key=f"support_type_{i}"
                )
                supports.append((pos, type))
                # Draw support
                if type == "Scharnier":
                    triangle = plt.Polygon([[pos, 0], 
                                         [pos - beam_length*0.02, -beam_length*0.02],
                                         [pos + beam_length*0.02, -beam_length*0.02]],
                                        color='#0077be', alpha=0.8, zorder=4)
                    ax_beam.add_patch(triangle)
                else:
                    circle = plt.Circle((pos, -beam_length*0.02), beam_length*0.01,
                                      color='#0077be', alpha=0.8, zorder=4)
                    ax_beam.add_patch(circle)
        
        # Plot settings
        ax_beam.set_xlabel("Lengte (mm)", color='#e0e0e0')
        ax_beam.set_ylabel("Doorbuiging (mm)", color='#e0e0e0')
        ax_beam.set_xlim(-beam_length*0.1, beam_length*1.1)
        ax_beam.set_ylim(-beam_length*0.15, beam_length*0.15)
        
        # Cross-section visualization
        ax_section = fig.add_subplot(gs[0, 2])
        ax_section.set_facecolor('#1e1e1e')
        
        if profile_type == "Koker":
            # Draw hollow section
            rect_outer = patches.Rectangle((0, 0), width, height, 
                                        facecolor='none', edgecolor='#00b4d8', linewidth=2)
            rect_inner = patches.Rectangle((wall_thickness, wall_thickness), 
                                        width-2*wall_thickness, height-2*wall_thickness,
                                        facecolor='#1e1e1e', edgecolor='#00b4d8', linewidth=2)
            ax_section.add_patch(rect_outer)
            ax_section.add_patch(rect_inner)
        else:
            # Draw I/H section
            # Flanges
            ax_section.add_patch(patches.Rectangle((0, 0), width, flange_thickness, 
                                                facecolor='#00b4d8', alpha=0.3, edgecolor='#00b4d8'))
            ax_section.add_patch(patches.Rectangle((0, height-flange_thickness), width, flange_thickness,
                                                facecolor='#00b4d8', alpha=0.3, edgecolor='#00b4d8'))
            # Web
            ax_section.add_patch(patches.Rectangle((width/2-wall_thickness/2, flange_thickness),
                                                wall_thickness, height-2*flange_thickness,
                                                facecolor='#00b4d8', alpha=0.3, edgecolor='#00b4d8'))
        
        ax_section.set_aspect('equal')
        ax_section.set_xlim(-width*0.1, width*1.1)
        ax_section.set_ylim(-height*0.1, height*1.1)
        ax_section.set_title("Doorsnede", color='#e0e0e0')
        ax_section.set_xticks([])
        ax_section.set_yticks([])
        
        # Show the plot
        st.pyplot(fig)
        
        st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    # Belastingen sectie
    st.markdown("""
        <div class="element-container">
            <div class="section-header">
                <span>Belastingen</span>
            </div>
    """, unsafe_allow_html=True)
    
    # Belastingen container
    st.markdown("""
        <div style='background-color: #2d2d2d; padding: 1rem; border-radius: 8px; border: 1px solid #404040;'>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    with col1:
        if st.button("Voeg belasting toe", 
                    help="Voeg een nieuwe belasting toe aan de balk",
                    use_container_width=True):
            if 'load_count' not in st.session_state:
                st.session_state.load_count = 0
            st.session_state.load_count += 1
            st.session_state.loads.append((0, 1000, "Puntlast"))
    with col2:
        if st.button("Wis alles", 
                    help="Verwijder alle belastingen",
                    use_container_width=True):
            st.session_state.load_count = 0
            st.session_state.loads = []
    
    # Toon bestaande belastingen
    if st.session_state.load_count == 0:
        st.markdown("""
            <div style='text-align: center; padding: 2rem; color: #666;'>
                <p>Nog geen belastingen toegevoegd</p>
                <p style='font-size: 0.9rem;'>Gebruik de "Voeg belasting toe" knop hierboven</p>
            </div>
        """, unsafe_allow_html=True)
    
    for i in range(st.session_state.load_count):
        with st.expander(f"Belasting {i+1}", expanded=True):
            # Container voor belasting
            st.markdown("""
                <div style='background-color: white; padding: 1rem; border-radius: 4px; border: 1px solid #eaecef;'>
            """, unsafe_allow_html=True)
            
            # Type en waarde
            col1, col2 = st.columns(2)
            with col1:
                load_type = st.selectbox(
                    "Type",
                    ["Puntlast", "Gelijkmatig verdeeld"],
                    key=f"load_type_{i}",
                    help="Kies het type belasting"
                )
            with col2:
                current_force = st.session_state.loads[i][1] if i < len(st.session_state.loads) else 1000
                force = st.number_input(
                    "Waarde (N)",
                    value=float(current_force),
                    step=100.0,
                    format="%.1f",
                    key=f"load_force_{i}",
                    help="Positieve waarde voor neerwaartse kracht, negatieve voor opwaartse kracht"
                )
            
            # Positie
            current_pos = st.session_state.loads[i][0] if i < len(st.session_state.loads) else beam_length/2
            pos = st.slider(
                "Positie (mm)",
                min_value=0.0,
                max_value=float(beam_length),
                value=float(current_pos),
                step=10.0,
                format="%.1f",
                key=f"load_pos_{i}",
                help="Positie van de belasting vanaf het linkeruiteinde"
            )
            
            # Lengte voor verdeelde last
            if load_type == "Gelijkmatig verdeeld":
                current_length = st.session_state.loads[i][3] if len(st.session_state.loads[i]) > 3 else min(100.0, beam_length-pos)
                length = st.slider(
                    "Lengte (mm)",
                    min_value=0.0,
                    max_value=float(beam_length-pos),
                    value=float(current_length),
                    step=10.0,
                    format="%.1f",
                    key=f"load_length_{i}",
                    help="Lengte waarover de belasting verdeeld is"
                )
                st.session_state.loads[i] = (float(pos), float(force), load_type, float(length))
            else:
                st.session_state.loads[i] = (float(pos), float(force), load_type)

            st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Resultaten sectie
    if len(st.session_state.loads) > 0:
        st.markdown("""
            <div class="element-container">
                <div class="section-header">
                    <span>Resultaten</span>
                </div>
        """, unsafe_allow_html=True)
        
        # Resultaten container
        st.markdown("""
            <div style='background-color: #2d2d2d; padding: 1rem; border-radius: 8px; border: 1px solid #404040;'>
        """, unsafe_allow_html=True)
        
        # Bereken doorbuiging
        x = np.linspace(0, beam_length, 200)
        y = np.zeros_like(x)
        for i in range(len(st.session_state.loads)):
            pos, F, load_type, *rest = st.session_state.loads[i]
            if load_type == "Puntlast":
                for j, xi in enumerate(x):
                    if xi <= pos:
                        y[j] += 0
                    else:
                        y[j] += -F * (xi - pos)**2 * (3*pos - beam_length - 2*xi) / (6 * E * calculate_I(profile_type, height, width, wall_thickness, flange_thickness))
            elif load_type == "Gelijkmatig verdeeld":
                length = float(rest[0])
                q = F / length
                for j, xi in enumerate(x):
                    start = max(pos, xi)
                    end = min(beam_length, pos + length)
                    if start < end:
                        if xi <= start:
                            y[j] += 0
                        elif xi <= end:
                            y[j] += -q * ((xi - start)**4 / 24 - (xi - pos)**2 * (xi - start)**2 / 4) / (E * calculate_I(profile_type, height, width, wall_thickness, flange_thickness))
                        else:
                            y[j] += -q * (end - start) * ((xi - pos)**2 * (3*xi - beam_length - 2*end) / 6) / (E * calculate_I(profile_type, height, width, wall_thickness, flange_thickness))
        
        max_defl = np.max(np.abs(y))
        max_pos = x[np.argmax(np.abs(y))]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Maximale doorbuiging",
                f"{max_defl:.2f} mm",
                f"{max_defl/beam_length*100:.2f}% van lengte",
                help="De grootste vervorming van de balk"
            )
        with col2:
            st.metric(
                "Positie max. doorbuiging",
                f"{max_pos:.0f} mm",
                f"{max_pos/beam_length*100:.1f}% van lengte",
                help="Positie waar de maximale doorbuiging optreedt"
            )
        
        st.markdown("</div>", unsafe_allow_html=True)

def calculate_I(profile_type, h, b, t, tf=None):
    """Bereken traagheidsmoment"""
    if profile_type == "Koker":
        # I = (bh³)/12 - ((b-2t)(h-2t)³)/12
        return (b * h**3 - (b-2*t) * (h-2*t)**3) / 12
    elif profile_type in ["I-profiel", "H-profiel"]:
        # Voor I/H-profiel
        hw = h - 2*tf  # Hoogte van het lijf
        return (b * h**3 - (b-t) * hw**3) / 12

def calculate_beam_response(x, L, E, I, supports, loads):
    """Verbeterde berekening voor meerdere steunpunten"""
    def moment_at_x(x, load, span_start, span_end):
        pos, F, load_type, *rest = load
        if load_type == "Puntlast":
            if span_start <= pos <= span_end and span_start <= x <= span_end:
                if x <= pos:
                    return F * (x - span_start)
                else:
                    return F * (pos - span_start)
            return 0
        elif load_type == "Gelijkmatig verdeeld":
            length = float(rest[0])
            q = F / length
            start = max(span_start, pos)
            end = min(span_end, pos + length)
            if start < end and span_start <= x <= span_end:
                if x <= start:
                    return 0
                elif x <= end:
                    return q * (x - start) * (x - start) / 2
                else:
                    return q * (end - start) * (2*x - start - end) / 2
            return 0
    
    # Sorteer steunpunten op positie
    supports = sorted(supports, key=lambda s: s[0])
    
    # Bereken doorbuiging voor elk segment
    y = 0
    if len(supports) == 1:  # Enkelvoudige inklemming
        x0 = supports[0][0]
        for load in loads:
            pos, F, load_type, *rest = load
            if load_type == "Puntlast":
                if x <= x0:
                    y = 0
                elif x <= pos:
                    y += -F * (x - x0)**2 * (3*pos - x0 - 2*x) / (6 * E * calculate_I(profile_type, height, width, wall_thickness, flange_thickness))
                else:
                    y += -F * (pos - x0)**2 * (3*x - x0 - 2*pos) / (6 * E * calculate_I(profile_type, height, width, wall_thickness, flange_thickness))
            elif load_type == "Gelijkmatig verdeeld":
                length = float(rest[0])
                q = F / length
                if x <= x0:
                    y = 0
                else:
                    start = max(x0, pos)
                    end = min(L, pos + length)
                    if start < end:
                        if x <= start:
                            y += 0
                        elif x <= end:
                            y += -q * ((x - start)**4 / 24 - (x - x0)**2 * (x - start)**2 / 4) / (E * calculate_I(profile_type, height, width, wall_thickness, flange_thickness))
                        else:
                            y += -q * (end - start) * ((x - x0)**2 * (3*x - x0 - 2*end) / 6) / (E * calculate_I(profile_type, height, width, wall_thickness, flange_thickness))
    else:  # Meerdere steunpunten
        for i in range(len(supports) - 1):
            x1, type1 = supports[i]
            x2, type2 = supports[i+1]
            
            if x1 <= x <= x2:
                for load in loads:
                    M = moment_at_x(x, load, x1, x2)
                    y += M * (x2 - x) * (x - x1) / (6 * E * calculate_I(profile_type, height, width, wall_thickness, flange_thickness) * (x2 - x1))
    
    return float(y)
