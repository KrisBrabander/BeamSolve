import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Circle
import matplotlib.gridspec as gridspec

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

# Numerieke integratie functies
def calculate_I(profile_type, h, b, t_w, t_f=None):
    """Bereken traagheidsmoment"""
    if profile_type == "Koker":
        # I = (bh³)/12 - (b-2t)(h-2t)³/12
        I = (b * h**3)/12 - ((b-2*t_w) * (h-2*t_w)**3)/12
    else:  # I- of H-profiel
        # I = (bh³)/12 + 2*[(b*t_f*(h-t_f)²)/4]
        I = (t_w * (h-2*t_f)**3)/12 + 2*(b*t_f*(h/2-t_f/2)**2)
    return I

def calc_deflection(M, EI, dx, theta_0, v_0, start_idx, end_idx, reverse=False):
    """
    Bereken doorbuiging middels numerieke integratie
    Parameters:
        M: Array met momentwaardes
        EI: Buigstijfheid
        dx: Stapgrootte
        theta_0: Initiële rotatie
        v_0: Initiële verplaatsing
        start_idx: Start index voor integratie
        end_idx: Eind index voor integratie
        reverse: Integreer in omgekeerde richting
    """
    theta_prev = theta_0
    v_prev = v_0
    
    n = len(M)
    rotation = np.zeros(n)
    deflection = np.zeros(n)
    rotation[start_idx] = theta_0
    deflection[start_idx] = v_0
    
    if reverse:
        range_indices = range(start_idx-1, end_idx-1, -1)
    else:
        range_indices = range(start_idx+1, end_idx+1)
    
    for i in range_indices:
        # Numerieke integratie met trapeziumregel
        M_prev = M[i-1] if not reverse else M[i+1]
        M_curr = M[i]
        M_avg = 0.5 * (M_prev + M_curr)
        
        # Bereken rotatie
        theta_curr = theta_prev + (M_avg/EI) * dx
        rotation[i] = theta_curr
        
        # Bereken verplaatsing
        v_curr = v_prev + 0.5 * (theta_curr + theta_prev) * dx
        deflection[i] = v_curr
        
        # Update voor volgende iteratie
        theta_prev = theta_curr
        v_prev = v_curr
    
    return rotation, deflection

def find_initial_rotation(M, EI, dx, support_indices, init_rot=0.0, delta_rot=0.0001):
    """
    Vind de juiste initiële rotatie door iteratief te zoeken
    """
    def calc_error(rot):
        # Als er maar 1 steunpunt is (inklemming), gebruik het einde van de balk
        end_idx = support_indices[1] if len(support_indices) > 1 else len(M)-1
        _, defl = calc_deflection(M, EI, dx, rot, 0.0, 
                                support_indices[0], end_idx)
        return defl[end_idx]
    
    # Test initiële richting
    err_0 = calc_error(init_rot)
    err_pos = calc_error(init_rot + delta_rot)
    
    # Bepaal zoekrichting
    if abs(err_pos) < abs(err_0):
        search_dir = 1
    else:
        search_dir = -1
    
    # Iteratief zoeken naar nulpunt
    max_iter = 100
    iter_count = 0
    curr_rot = init_rot
    best_rot = curr_rot
    min_error = abs(err_0)
    
    while iter_count < max_iter:
        curr_rot += search_dir * delta_rot
        curr_error = abs(calc_error(curr_rot))
        
        if curr_error < min_error:
            min_error = curr_error
            best_rot = curr_rot
        elif curr_error > min_error * 1.5:  # Error wordt significant groter
            break
            
        iter_count += 1
    
    return best_rot

def calculate_moment_at_x(x, L, supports, loads):
    M = 0
    for pos, F, load_type, *rest in loads:
        if load_type == "Puntlast":
            if x >= pos:
                M += F * (x - pos)
        elif load_type == "Gelijkmatig verdeeld":
            length = float(rest[0])
            q = F / length
            start = max(pos, x)
            end = min(L, pos + length)
            if start < end:
                M += 0.5 * q * (end - start) * (end + start - 2*x)
    return M

# Update main analysis code
def analyze_beam(beam_length, supports, loads, profile_type, height, width, 
                wall_thickness, flange_thickness, E):
    """
    Hoofdfunctie voor balkanalyse
    """
    # Discretisatie parameters
    n_points = 200
    dx = beam_length / (n_points - 1)
    x = np.linspace(0, beam_length, n_points)
    
    # Bereken traagheidsmoment
    I = calculate_I(profile_type, height, width, wall_thickness, flange_thickness)
    EI = E * I
    
    # Initialiseer arrays
    M = np.zeros(n_points)  # Moment array
    
    # Bereken momentenlijn
    for i, xi in enumerate(x):
        M[i] = calculate_moment_at_x(xi, beam_length, supports, loads)
    
    # Vind support indices
    support_indices = []
    for pos, _ in supports:
        idx = np.argmin(np.abs(x - pos))
        support_indices.append(idx)
    
    # Bereken initiële rotatie
    init_rotation = find_initial_rotation(M, EI, dx, support_indices)
    
    # Bereken doorbuiging voor hoofddeel
    end_idx = support_indices[1] if len(support_indices) > 1 else len(x)-1
    rotation, deflection = calc_deflection(M, EI, dx, init_rotation, 0.0,
                                         support_indices[0], end_idx)
    
    # Bereken doorbuiging voor eventuele linker uitkraging
    if support_indices[0] > 0:
        left_rot, left_defl = calc_deflection(M, EI, dx, -init_rotation, 0.0,
                                            support_indices[0], 0, reverse=True)
        rotation[:support_indices[0]] = left_rot[:support_indices[0]]
        deflection[:support_indices[0]] = left_defl[:support_indices[0]]
    
    # Bereken doorbuiging voor eventuele rechter uitkraging
    if len(support_indices) > 1 and support_indices[1] < len(x)-1:
        right_rot, right_defl = calc_deflection(M, EI, dx, rotation[support_indices[1]], 0.0,
                                              support_indices[1], len(x)-1)
        rotation[support_indices[1]:] = right_rot[support_indices[1]:]
        deflection[support_indices[1]:] = right_defl[support_indices[1]:]
    
    return x, M, rotation, deflection

# Main content
with main:
    # Create tabs
    tab1, tab2, tab3 = st.tabs([" Analyse", " Resultaten", " Rapport"])
    
    with tab1:
        # Modern visualization container
        st.markdown('<div style="background-color: #252525; padding: 1.5rem; border-radius: 8px; border: 1px solid #333;">', unsafe_allow_html=True)
        
        # Verzamel supports
        supports = []
        if support_count == 1:
            pos = st.slider("Positie inklemming (mm)", 0.0, beam_length, 0.0, key="inklemming_pos")
            supports.append((pos, "Inklemming"))
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
        
        # Voer analyse uit
        x, M, rotation, deflection = analyze_beam(
            beam_length, supports, st.session_state.loads,
            profile_type, height, width, wall_thickness, flange_thickness, E
        )
        
        # Plot resultaten
        fig = plt.figure(figsize=(15, 10))
        fig.patch.set_facecolor('#2d2d2d')
        gs = gridspec.GridSpec(3, 3, height_ratios=[2, 1, 1])
        
        # Doorbuigingsplot
        ax_defl = fig.add_subplot(gs[0, :])
        ax_defl.set_facecolor('#2d2d2d')
        ax_defl.plot(x, deflection, '-', color='#48cae4', linewidth=2.5, label='Doorbuiging')
        ax_defl.grid(True, linestyle='--', alpha=0.3, color='#909090')
        ax_defl.set_xlabel('Positie (mm)', color='#ffffff', fontsize=10)
        ax_defl.set_ylabel('Doorbuiging (mm)', color='#ffffff', fontsize=10)
        ax_defl.set_title('Doorbuiging', color='#48cae4', pad=20, fontsize=12)
        ax_defl.tick_params(colors='#ffffff', labelsize=9)
        for spine in ax_defl.spines.values():
            spine.set_color('#909090')
        
        # Momentenplot
        ax_moment = fig.add_subplot(gs[1, :])
        ax_moment.set_facecolor('#2d2d2d')
        ax_moment.plot(x, M, '-', color='#90e0ef', linewidth=2.5, label='Moment')
        ax_moment.grid(True, linestyle='--', alpha=0.3, color='#909090')
        ax_moment.set_xlabel('Positie (mm)', color='#ffffff', fontsize=10)
        ax_moment.set_ylabel('Moment (Nmm)', color='#ffffff', fontsize=10)
        ax_moment.tick_params(colors='#ffffff', labelsize=9)
        for spine in ax_moment.spines.values():
            spine.set_color('#909090')
        
        # Rotatieplot
        ax_rot = fig.add_subplot(gs[2, :])
        ax_rot.set_facecolor('#2d2d2d')
        ax_rot.plot(x, rotation, '-', color='#00b4d8', linewidth=2.5, label='Rotatie')
        ax_rot.grid(True, linestyle='--', alpha=0.3, color='#909090')
        ax_rot.set_xlabel('Positie (mm)', color='#ffffff', fontsize=10)
        ax_rot.set_ylabel('Rotatie (rad)', color='#ffffff', fontsize=10)
        ax_rot.tick_params(colors='#ffffff', labelsize=9)
        for spine in ax_rot.spines.values():
            spine.set_color('#909090')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Toon maximale doorbuiging
        max_defl = np.max(np.abs(deflection))
        max_defl_pos = x[np.argmax(np.abs(deflection))]
        st.markdown(f"""
        <div style='background-color: #2d2d2d; padding: 1rem; border-radius: 4px; margin-top: 1rem;'>
            <h3 style='color: #48cae4; margin: 0;'>Maximale doorbuiging</h3>
            <p style='color: #ffffff; margin: 0.5rem 0;'>
                {max_defl:.2f} mm @ x = {max_defl_pos:.1f} mm
            </p>
        </div>
        """, unsafe_allow_html=True)

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
        
        def calculate_deflection():
            if st.session_state.loads:
                # Bereken doorbuiging
                x = np.linspace(0, beam_length, 200)
                y = np.zeros_like(x)
                for load in st.session_state.loads:
                    pos, F, load_type, *rest = load  
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
        
        calculate_deflection()
        
        st.markdown("</div>", unsafe_allow_html=True)
