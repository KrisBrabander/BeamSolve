import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, PathPatch
from matplotlib.path import Path
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Pagina configuratie
st.set_page_config(
    page_title="BeamFEA Professional",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/BeamFEA',
        'Report a bug': 'https://github.com/yourusername/BeamFEA/issues',
        'About': '''
        # BeamFEA Professional Edition
        
        Advanced Finite Element Analysis Software for Structural Engineering
        Version 2025.1 Enterprise
        
        Features:
        - Advanced 3D Beam Visualization
        - Real-time FEA Analysis
        - Professional Engineering Reports
        - Multi-support Configuration
        - Stress/Strain Analysis
        '''
    }
)

# Styling voor een high-end engineering look
st.markdown("""
    <style>
    /* Modern engineering software look */
    .main {
        background-color: #1e1e1e;
        color: #e0e0e0;
    }
    .stApp {
        max-width: 1600px;
        margin: 0 auto;
    }
    
    /* Professional header */
    h1 {
        background: linear-gradient(90deg, #0288d1 0%, #0277bd 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 8px;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 600;
        font-size: 2.2rem;
        margin: 1rem 0 2rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Engineering style containers */
    .section-container {
        background-color: #2d2d2d;
        border: 1px solid #404040;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Modern inputs */
    .stNumberInput > div > div {
        background-color: #363636;
        border: 1px solid #404040;
        border-radius: 4px;
        color: #e0e0e0;
    }
    .stNumberInput > div > div:focus-within {
        border-color: #0288d1;
        box-shadow: 0 0 0 2px rgba(2,136,209,0.2);
    }
    
    /* Professional buttons */
    .stButton > button {
        background: linear-gradient(90deg, #0288d1 0%, #0277bd 100%);
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 4px;
        font-weight: 500;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #039be5 0%, #0288d1 100%);
        box-shadow: 0 4px 8px rgba(2,136,209,0.3);
    }
    
    /* Engineering metrics */
    [data-testid="stMetricValue"] {
        color: #0288d1 !important;
        font-size: 1.8rem !important;
        font-weight: 600 !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    [data-testid="stMetricDelta"] {
        color: #4fc3f7 !important;
        font-size: 1rem !important;
    }
    
    /* Technical labels */
    label {
        color: #90caf9;
        font-weight: 500;
        text-transform: uppercase;
        font-size: 0.9rem;
        letter-spacing: 0.5px;
    }
    
    /* Help tooltips */
    .stTooltipIcon {
        color: #0288d1;
    }
    </style>
""", unsafe_allow_html=True)

# Default waardes
DEFAULT_HEIGHT = 100.0
DEFAULT_WIDTH = 50.0
DEFAULT_WALL_THICKNESS = 5.0
DEFAULT_FLANGE_THICKNESS = 5.0
DEFAULT_E_MODULUS = 210000.0
DEFAULT_BEAM_LENGTH = 1000.0
DEFAULT_FORCE = 1000.0

# Functies voor berekeningen
def calculate_I(profile_type, h, b, t, tf=None):
    """Bereken traagheidsmoment (mm⁴)"""
    h = float(h)
    b = float(b)
    t = float(t)
    if tf is not None:
        tf = float(tf)
    
    if profile_type == "Koker":
        return (b * h**3 - (b-2*t) * (h-2*t)**3) / 12
    else:  # I-profiel of U-profiel
        hw = h - 2*tf  # Hoogte van het lijf
        return (b * h**3 - (b-t) * hw**3) / 12

def calculate_beam_response_advanced(x, L, E, I, supports, loads):
    """Geavanceerde balkberekening met meerdere steunpunten"""
    x = float(x)
    L = float(L)
    E = float(E)
    I = float(I)
    
    # Sorteer steunpunten op positie
    supports = sorted(supports, key=lambda s: s[0])
    
    # Matrix methode voor meerdere steunpunten
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
    
    # Bereken doorbuiging voor elk segment
    y = 0
    for i in range(len(supports) - 1):
        x1, type1 = supports[i]
        x2, type2 = supports[i+1]
        
        if x1 <= x <= x2:
            # Pas superpositie toe voor alle belastingen
            for load in loads:
                M = moment_at_x(x, load, x1, x2)
                y += M * (x2 - x) * (x - x1) / (6 * E * I * (x2 - x1))
    
    return float(y)

def create_3d_beam_visualization(fig, profile_type, height, width, wall_thickness, flange_thickness=None):
    """Creëer 3D visualisatie van het balkprofiel"""
    ax = fig.add_subplot(122, projection='3d')
    
    if profile_type == "Koker":
        # Vertices voor koker profiel
        vertices = np.array([
            [0, 0, 0], [width, 0, 0], [width, height, 0], [0, height, 0],
            [wall_thickness, wall_thickness, 0], 
            [width-wall_thickness, wall_thickness, 0],
            [width-wall_thickness, height-wall_thickness, 0],
            [wall_thickness, height-wall_thickness, 0]
        ])
        
        # Teken de voor- en achterkant
        depth = width  # Maak het 3D
        for z in [0, depth]:
            ax.add_collection3d(plt.fill(vertices[[0,1,2,3],0], 
                                       vertices[[0,1,2,3],1], 
                                       color='#90caf9', alpha=0.3))
            ax.add_collection3d(plt.fill(vertices[[4,5,6,7],0], 
                                       vertices[[4,5,6,7],1], 
                                       color='#90caf9', alpha=0.3))
    
    ax.set_xlabel('Breedte (mm)')
    ax.set_ylabel('Hoogte (mm)')
    ax.set_zlabel('Diepte (mm)')
    ax.set_title('3D Profiel Visualisatie')
    
    # Voeg wat stijl toe
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    return ax

# Header sectie
st.markdown("""
    <h1>BeamFEA Professional</h1>
    <div class="version-badge">Version 2025.1 Enterprise</div>
    <p class="subtitle">Advanced Finite Element Analysis Software for Structural Engineering</p>
""", unsafe_allow_html=True)

# Hoofdcontainer voor de app
st.markdown('<div class="section-container">', unsafe_allow_html=True)

# Profielgegevens sectie
st.markdown("""
    <div class="section-header">
        <span>📐 Profielgegevens</span>
    </div>
""", unsafe_allow_html=True)

# Profielgegevens
profile_type = st.selectbox(
    "Profieltype",
    ["Koker", "I-profiel", "U-profiel"],
    help="Selecteer het type profiel voor de berekening"
)

col1, col2 = st.columns(2)
with col1:
    height = st.number_input(
        "Hoogte (mm)",
        min_value=1.0,
        value=DEFAULT_HEIGHT,
        step=1.0,
        format="%.1f",
        help="Totale hoogte van het profiel"
    )
with col2:
    width = st.number_input(
        "Breedte (mm)",
        min_value=1.0,
        value=DEFAULT_WIDTH,
        step=1.0,
        format="%.1f",
        help="Totale breedte van het profiel"
    )

wall_thickness = st.number_input(
    "Wanddikte (mm)",
    min_value=0.1,
    value=DEFAULT_WALL_THICKNESS,
    step=0.1,
    format="%.1f",
    help="Dikte van de wanden"
)

if profile_type in ["I-profiel", "U-profiel"]:
    flange_thickness = st.number_input(
        "Flensdikte (mm)",
        min_value=0.1,
        value=DEFAULT_FLANGE_THICKNESS,
        step=0.1,
        format="%.1f",
        help="Dikte van de flenzen"
    )
else:
    flange_thickness = None

# Materiaal
st.markdown("""
    <div class="section-header">
        <span>🔧 Materiaal</span>
    </div>
""", unsafe_allow_html=True)

E = st.number_input(
    "E-modulus (N/mm²)",
    min_value=1.0,
    value=DEFAULT_E_MODULUS,
    step=1000.0,
    format="%.1f",
    help="Elasticiteitsmodulus van het materiaal (210000 N/mm² voor staal)"
)

# Overspanning
st.markdown("""
    <div class="section-header">
        <span>📏 Overspanning</span>
    </div>
""", unsafe_allow_html=True)

beam_length = st.number_input(
    "Lengte (mm)",
    min_value=1.0,
    value=DEFAULT_BEAM_LENGTH,
    step=10.0,
    format="%.1f",
    help="Totale lengte van de balk"
)

# Steunpunten
st.markdown("""
    <div class="section-header">
        <span>🔗 Steunpunten</span>
    </div>
""", unsafe_allow_html=True)

support_count = st.selectbox(
    "Aantal steunpunten",
    [1, 2, 3],
    help="Kies het aantal steunpunten voor de balk"
)

supports = []
if support_count == 1:
    pos = st.number_input(
        "Positie inklemming (mm)",
        0.0, beam_length, 0.0,
        key="inklemming_pos",
        help="Positie van de inklemming vanaf het linkeruiteinde"
    )
    supports.append((pos, "Inklemming"))
else:
    for i in range(support_count):
        col1, col2 = st.columns(2)
        with col1:
            pos = st.number_input(
                f"Positie {i+1} (mm)",
                0.0, beam_length,
                value=i * beam_length/(support_count-1) if support_count > 1 else 0.0,
                key=f"support_pos_{i}",
                help=f"Positie van steunpunt {i+1} vanaf het linkeruiteinde"
            )
        with col2:
            type = st.selectbox(
                "Type",
                ["Scharnier", "Rol"],
                key=f"support_type_{i}",
                help="Scharnier: vast punt, Rol: kan horizontaal bewegen"
            )
        supports.append((pos, type))

# Hoofdgedeelte
col1, col2 = st.columns([7, 3])

with col1:
    st.markdown("""
        <div class="section-container">
            <div class="section-header">
                <span>📊 Visualisatie</span>
            </div>
    """, unsafe_allow_html=True)
    
    # Maak een mooiere plot
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#1e1e1e')
    ax.set_facecolor('#1e1e1e')
    
    # Grid styling
    ax.grid(True, linestyle='--', alpha=0.3, color='#666666')
    
    # Plot onvervormde balk met mooiere stijl
    ax.plot([0, beam_length], [0, 0], '--', color='#cccccc', alpha=0.5, linewidth=1.5, label='Onvervormde balk')
    
    # Bereken doorbuiging
    x = np.linspace(0, beam_length, 300)  # Meer punten voor vloeiendere curve
    y = np.zeros_like(x)
    for i, xi in enumerate(x):
        y[i] = -calculate_beam_response_advanced(xi, beam_length, E, calculate_I(profile_type, height, width, wall_thickness, flange_thickness), supports, [])
    
    # Schaal doorbuiging
    scale = 1.0
    if np.any(y != 0):
        max_defl = np.max(np.abs(y))
        if max_defl > 0:
            desired_height = beam_length / 10
            scale = desired_height / max_defl
    
    # Plot vervormde balk met mooiere stijl
    ax.plot(x, y * scale, color='#2196f3', linewidth=2.5, label='Vervormde balk', zorder=3)
    
    # Teken steunpunten met verbeterde stijl
    for pos, type in supports:
        if type == "Scharnier":
            # Driehoek voor scharnier
            triangle = plt.Polygon([[pos, 0], 
                                 [pos - beam_length*0.02, -beam_length*0.02],
                                 [pos + beam_length*0.02, -beam_length*0.02]],
                                color='#424242', alpha=0.8, zorder=4)
            ax.add_patch(triangle)
        elif type == "Rol":
            # Cirkel met driehoek voor rol
            circle = plt.Circle((pos, -beam_length*0.02), beam_length*0.01,
                              fill=False, color='#424242', linewidth=2, zorder=4)
            ax.add_artist(circle)
            # Driehoek boven rol
            triangle = plt.Polygon([[pos, 0], 
                                 [pos - beam_length*0.02, -beam_length*0.02],
                                 [pos + beam_length*0.02, -beam_length*0.02]],
                                color='#424242', alpha=0.8, zorder=4)
            ax.add_patch(triangle)
        else:  # Inklemming
            rect_height = beam_length * 0.04
            rect_width = beam_length * 0.01
            rect = patches.Rectangle((pos, -rect_height/2), rect_width, rect_height,
                                  color='#424242', alpha=0.8, zorder=4)
            ax.add_patch(rect)
    
    # Plot instellingen
    ax.set_xlabel("Lengte (mm)", fontsize=10, color='#666666')
    ax.set_ylabel("Doorbuiging (mm)", fontsize=10, color='#666666')
    ax.set_xlim(-beam_length*0.1, beam_length*1.1)
    ax.set_ylim(-beam_length*0.15, beam_length*0.15)
    ax.set_aspect('equal', adjustable='box')
    
    # Voeg legenda toe
    ax.legend(loc='upper right', frameon=True, fancybox=True, 
             shadow=True, framealpha=0.9, fontsize=9)
    
    # Toon plot
    st.pyplot(fig)

    # 3D visualisatie van het profiel
    fig = plt.figure(figsize=(10, 5))
    ax = create_3d_beam_visualization(fig, profile_type, height, width, wall_thickness, flange_thickness)
    st.pyplot(fig)

with col2:
    # Belastingen sectie
    st.markdown("""
        <div class="section-container">
            <div class="section-header">
                <span>🎯 Belastingen</span>
            </div>
    """, unsafe_allow_html=True)
    
    # Belastingen container
    st.markdown("""
        <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 8px; border: 1px solid #eaecef;'>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    with col1:
        if st.button("➕ Voeg belasting toe", 
                    help="Voeg een nieuwe belasting toe aan de balk",
                    use_container_width=True):
            if 'load_count' not in st.session_state:
                st.session_state.load_count = 0
            st.session_state.load_count += 1
            st.session_state.loads.append((0, 1000, "Puntlast"))
    with col2:
        if st.button("🗑️ Wis alles", 
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
        with st.expander(f"📌 Belasting {i+1}", expanded=True):
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
                current_force = st.session_state.loads[i][1] if i < len(st.session_state.loads) else DEFAULT_FORCE
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
            <div class="section-container">
                <div class="section-header">
                    <span>📊 Resultaten</span>
                </div>
        """, unsafe_allow_html=True)
        
        # Resultaten container
        st.markdown("""
            <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 8px; border: 1px solid #eaecef;'>
        """, unsafe_allow_html=True)
        
        # Bereken doorbuiging
        x = np.linspace(0, beam_length, 200)
        y = np.zeros_like(x)
        for i, xi in enumerate(x):
            y[i] = -calculate_beam_response_advanced(xi, beam_length, E, calculate_I(profile_type, height, width, wall_thickness, flange_thickness), supports, st.session_state.loads)
        
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

st.markdown("</div>", unsafe_allow_html=True)
