import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch

# Initialiseer session state
if 'load_count' not in st.session_state:
    st.session_state.load_count = 0
if 'loads' not in st.session_state:
    st.session_state.loads = []

# Pagina configuratie
st.set_page_config(
    page_title="Buigingsberekeningen Pro",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/buigingsberekeningen',
        'Report a bug': 'https://github.com/yourusername/buigingsberekeningen/issues',
        'About': '''
        # Buigingsberekeningen Pro
        
        Professionele software voor balkdoorbuigingsberekeningen.
        Versie 1.0.0
        
        Alle rechten voorbehouden.
        '''
    }
)

# Functies voor berekeningen
def calculate_I(profile_type, h, b, t, tf=None):
    """Bereken traagheidsmoment (mm⁴)"""
    if profile_type == "Koker":
        return (b * h**3 - (b-2*t) * (h-2*t)**3) / 12
    else:  # I-profiel of U-profiel
        hw = h - 2*tf  # Hoogte van het lijf
        return (b * h**3 - (b-t) * hw**3) / 12

def calculate_beam_response(x, L, E, I, supports, loads):
    """Bereken de mechanische respons op positie x"""
    y = 0  # Doorbuiging
    
    # Voor één inklemming
    if len(supports) == 1 and supports[0][1] == "Inklemming":
        x0 = supports[0][0]  # Positie van inklemming
        
        for load in loads:
            pos, F, load_type, *rest = load
            F = float(F)
            
            if load_type == "Puntlast":
                if pos > x0:  # Alleen als last voorbij inklemming
                    a = pos - x0
                    if x <= x0:
                        y += 0
                    elif x <= pos:
                        y += F * (x - x0)**2 * (3*a - (x - x0)) / (6 * E * I)
                    else:
                        y += F * a**2 * (3*(x - x0) - a) / (6 * E * I)
            
            elif load_type == "Gelijkmatig verdeeld":
                length = float(rest[0])
                start = max(x0, pos)
                end = pos + length
                
                if start < end:
                    q = F / length
                    if x <= x0:
                        y += 0
                    else:
                        if x <= start:
                            y += q * ((x - x0)**2 * (4*end - start - 3*x) / 24) / (E * I)
                        elif x <= end:
                            y += q * ((x - x0)**2 * (4*end - start - 3*x) / 24 + (x - start)**4 / 24) / (E * I)
                        else:
                            y += q * length * (x - (end + start)/2) * (x - x0)**2 / (6 * E * I)
    
    # Voor twee of drie steunpunten
    else:
        supports = sorted(supports, key=lambda x: x[0])
        x1, x2 = supports[0][0], supports[-1][0]
        L_eff = x2 - x1
        
        for load in loads:
            pos, F, load_type, *rest = load
            F = float(F)
            
            if load_type == "Puntlast":
                # Pas positie aan relatief tot eerste steunpunt
                a = pos - x1  # Afstand vanaf eerste steunpunt
                b = x2 - pos  # Afstand tot tweede steunpunt
                
                if x1 <= x <= x2:
                    # Doorbuiging voor puntlast
                    if x <= pos:
                        y += F * b * (x - x1) * (L_eff**2 - b**2 - (x - x1)**2) / (6 * E * I * L_eff)
                    else:
                        y += F * a * (x2 - x) * (2*L_eff*(x - x1) - (x - x1)**2 - a**2) / (6 * E * I * L_eff)
            
            elif load_type == "Gelijkmatig verdeeld":
                length = float(rest[0])
                q = F / length
                start = max(x1, pos)
                end = min(x2, pos + length)
                
                if start < end and x1 <= x <= x2:
                    # Vereenvoudigde formule voor verdeelde last
                    y += -q * (x - x1)**2 * (L_eff - (x - x1))**2 / (24 * E * I * L_eff)
    
    return y

# Styling
st.markdown("""
    <style>
    /* Algemene app styling */
    .main {
        background-color: #ffffff;
        padding: 2rem;
    }
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
    }
    
    /* Header styling */
    h1 {
        color: #1a237e;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 600;
        font-size: 2.2rem;
        padding: 1.5rem 0;
        margin-bottom: 0.5rem;
    }
    .version-badge {
        background-color: #e3f2fd;
        color: #1565c0;
        padding: 0.3rem 0.8rem;
        border-radius: 16px;
        font-size: 0.9rem;
        font-weight: 500;
        display: inline-block;
        margin-bottom: 1.5rem;
    }
    .subtitle {
        color: #546e7a;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Secties styling */
    .section-container {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
        border: 1px solid #e0e0e0;
    }
    .section-header {
        color: #1a237e;
        font-size: 1.3rem;
        font-weight: 500;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Input velden */
    .stTextInput > div > div {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.5rem;
    }
    .stTextInput > div > div:focus-within {
        border-color: #1565c0;
        box-shadow: 0 0 0 2px rgba(21,101,192,0.1);
    }
    .stNumberInput > div > div {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
    }
    
    /* Knoppen */
    .stButton > button {
        background-color: #1565c0;
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        font-weight: 500;
        letter-spacing: 0.3px;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #1976d2;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transform: translateY(-1px);
    }
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        color: #1a237e !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricDelta"] {
        color: #1565c0 !important;
        font-size: 1rem !important;
    }
    
    /* Help icons */
    .stTooltipIcon {
        color: #90a4ae;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #f8fafc;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        padding: 0.8rem !important;
    }
    .streamlit-expanderContent {
        border: 1px solid #e2e8f0;
        border-top: none;
        border-radius: 0 0 8px 8px;
        padding: 1.2rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# Header sectie
st.markdown("""
    <h1>Buigingsberekeningen Pro</h1>
    <div class="version-badge">Professional Edition v1.0.0</div>
    <p class="subtitle">Professionele balkdoorbuigingsanalyse</p>
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
    height = st.number_input("Hoogte (mm)", min_value=1.0, value=100.0,
                           help="Totale hoogte van het profiel")
with col2:
    width = st.number_input("Breedte (mm)", min_value=1.0, value=50.0,
                          help="Totale breedte van het profiel")

wall_thickness = st.number_input("Wanddikte (mm)", min_value=0.1, value=5.0,
                               help="Dikte van de wanden")

if profile_type in ["I-profiel", "U-profiel"]:
    flange_thickness = st.number_input("Flensdikte (mm)", min_value=0.1, value=5.0,
                                     help="Dikte van de flenzen")
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
    value=210000.0,
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
    value=1000.0,
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
                key=f"pos_{i}",
                help=f"Positie van steunpunt {i+1} vanaf het linkeruiteinde"
            )
        with col2:
            type = st.selectbox(
                "Type",
                ["Scharnier", "Rol"],
                key=f"type_{i}",
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
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    
    # Grid styling
    ax.grid(True, linestyle='--', alpha=0.3, color='#666666')
    
    # Plot onvervormde balk met mooiere stijl
    ax.plot([0, beam_length], [0, 0], '--', color='#cccccc', alpha=0.5, linewidth=1.5, label='Onvervormde balk')
    
    # Bereken doorbuiging
    x = np.linspace(0, beam_length, 300)  # Meer punten voor vloeiendere curve
    y = np.zeros_like(x)
    for i, xi in enumerate(x):
        y[i] = -calculate_beam_response(xi, beam_length, E, calculate_I(profile_type, height, width, wall_thickness, flange_thickness), supports, st.session_state.loads)
    
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
    
    # Plot belastingen met verbeterde stijl
    arrow_height = beam_length * 0.05
    for load in st.session_state.loads:
        pos = load[0]
        F = load[1]
        load_type = load[2]
        
        if load_type == "Puntlast":
            # Verbeterde pijlstijl voor puntlast
            if F > 0:  # Naar beneden
                arrow = FancyArrowPatch((pos, arrow_height), (pos, 0),
                                      arrowstyle='simple', color='#e53935',
                                      mutation_scale=20, linewidth=2, zorder=5)
                ax.add_patch(arrow)
                ax.text(pos, arrow_height*1.1, f'{abs(F):.0f}N',
                       ha='center', va='bottom', color='#e53935',
                       fontsize=10, fontweight='bold', zorder=5)
            else:  # Naar boven
                arrow = FancyArrowPatch((pos, -arrow_height), (pos, 0),
                                      arrowstyle='simple', color='#e53935',
                                      mutation_scale=20, linewidth=2, zorder=5)
                ax.add_patch(arrow)
                ax.text(pos, -arrow_height*1.1, f'{abs(F):.0f}N',
                       ha='center', va='top', color='#e53935',
                       fontsize=10, fontweight='bold', zorder=5)
        
        elif load_type == "Gelijkmatig verdeeld":
            length = load[3]
            q = F / length  # N/mm
            arrow_spacing = length / 10
            
            # Verbeterde pijlen voor verdeelde last
            for x in np.arange(pos, pos + length + arrow_spacing/2, arrow_spacing):
                if F > 0:  # Naar beneden
                    arrow = FancyArrowPatch((x, arrow_height/2), (x, 0),
                                          arrowstyle='simple', color='#e53935',
                                          mutation_scale=15, linewidth=1.5, zorder=5)
                    ax.add_patch(arrow)
                else:  # Naar boven
                    arrow = FancyArrowPatch((x, -arrow_height/2), (x, 0),
                                          arrowstyle='simple', color='#e53935',
                                          mutation_scale=15, linewidth=1.5, zorder=5)
                    ax.add_patch(arrow)
            
            # Waarde van de verdeelde last
            mid_pos = pos + length/2
            if F > 0:
                ax.text(mid_pos, arrow_height*0.6, f'{abs(q):.1f}N/mm',
                       ha='center', va='bottom', color='#e53935',
                       fontsize=10, fontweight='bold', zorder=5)
            else:
                ax.text(mid_pos, -arrow_height*0.6, f'{abs(q):.1f}N/mm',
                       ha='center', va='top', color='#e53935',
                       fontsize=10, fontweight='bold', zorder=5)
    
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
                force = st.number_input(
                    "Waarde (N)",
                    value=st.session_state.loads[i][1],
                    step=100.0,
                    key=f"force_{i}",
                    help="Positieve waarde voor neerwaartse kracht, negatieve voor opwaartse kracht"
                )
            
            # Positie
            pos = st.slider(
                "Positie (mm)",
                0.0, beam_length, st.session_state.loads[i][0],
                key=f"load_pos_{i}",
                help="Positie van de belasting vanaf het linkeruiteinde"
            )
            
            # Lengte voor verdeelde last
            if load_type == "Gelijkmatig verdeeld":
                length = st.slider(
                    "Lengte (mm)",
                    0.0, beam_length-pos, min(100.0, beam_length-pos),
                    key=f"load_length_{i}",
                    help="Lengte waarover de belasting verdeeld is"
                )
                st.session_state.loads[i] = (pos, force, load_type, length)
            else:
                st.session_state.loads[i] = (pos, force, load_type)
            
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
            y[i] = -calculate_beam_response(xi, beam_length, E, calculate_I(profile_type, height, width, wall_thickness, flange_thickness), supports, st.session_state.loads)
        
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
