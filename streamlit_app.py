import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Circle
import matplotlib.gridspec as gridspec
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configureer de pagina
st.set_page_config(
    page_title="BeamCAD 2025 Enterprise",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS voor professionele uitstraling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #1a1a1a;
        color: #e0e0e0;
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        background-color: #2d2d2d !important;
        color: #ffffff !important;
        border: 1px solid #404040 !important;
        border-radius: 4px;
        padding: 0.5rem;
        font-size: 0.9rem;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #0077be 0%, #48cae4 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.6rem 1.2rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.5px !important;
        text-transform: uppercase !important;
        font-size: 0.85rem !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(0,119,190,0.2) !important;
    }
    
    /* Cards */
    .css-1r6slb0 {  /* Streamlit card class */
        background-color: #2d2d2d !important;
        border: 1px solid #404040 !important;
        border-radius: 8px !important;
        padding: 1.5rem !important;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #48cae4 !important;
        font-weight: 600 !important;
        letter-spacing: -0.5px !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 600 !important;
        color: #ffffff !important;
    }
    [data-testid="stMetricDelta"] {
        color: #909090 !important;
        font-size: 0.9rem !important;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #1a1a1a !important;
        border-right: 1px solid #404040 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px !important;
        background-color: #2d2d2d !important;
        padding: 0.5rem !important;
        border-radius: 8px !important;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px !important;
        background-color: transparent !important;
        border: 1px solid #404040 !important;
        border-radius: 4px !important;
        color: #e0e0e0 !important;
        font-size: 0.9rem !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #0077be 0%, #48cae4 100%) !important;
        border: none !important;
        color: white !important;
    }
    
    /* Plot container */
    .plot-container {
        background-color: #2d2d2d !important;
        border: 1px solid #404040 !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        margin: 1rem 0 !important;
    }
    
    /* Custom grid layout */
    .grid-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    .grid-item {
        background-color: #2d2d2d;
        border: 1px solid #404040;
        border-radius: 8px;
        padding: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Header met enterprise branding
st.markdown("""
<div style='background: linear-gradient(90deg, #2d2d2d 0%, #1a1a1a 100%); padding: 2rem; border-radius: 8px; margin-bottom: 2rem; border: 1px solid #404040; box-shadow: 0 4px 12px rgba(0,0,0,0.1);'>
    <div style='display: flex; align-items: center; margin-bottom: 1rem;'>
        <span style='font-size: 2.5rem; margin-right: 1rem;'>🔧</span>
        <div>
            <h1 style='margin:0; color: #48cae4; font-size: 2.2rem; font-weight: 700;'>BeamCAD 2025 Enterprise</h1>
            <p style='color: #909090; margin: 0.3rem 0 0 0; font-size: 1rem;'>Professional Engineering Analysis Suite</p>
        </div>
    </div>
    <div style='display: flex; gap: 2rem; margin-top: 1rem;'>
        <div style='background-color: rgba(72, 202, 228, 0.1); padding: 0.5rem 1rem; border-radius: 4px; border: 1px solid rgba(72, 202, 228, 0.2);'>
            <span style='color: #48cae4; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px;'>Version</span>
            <p style='color: white; margin: 0; font-weight: 500;'>Enterprise 2025.1</p>
        </div>
        <div style='background-color: rgba(72, 202, 228, 0.1); padding: 0.5rem 1rem; border-radius: 4px; border: 1px solid rgba(72, 202, 228, 0.2);'>
            <span style='color: #48cae4; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px;'>License</span>
            <p style='color: white; margin: 0; font-weight: 500;'>Professional</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'load_count' not in st.session_state:
    st.session_state.load_count = 0
if 'loads' not in st.session_state:
    st.session_state.loads = []

# App title
# st.markdown('<p class="app-title">BeamCAE 2025 Enterprise</p>', unsafe_allow_html=True)

# Create two columns: sidebar and main content
sidebar = st.sidebar
main = st.container()

# Sidebar sections
with sidebar:
    st.markdown("""
        <div style='background: linear-gradient(180deg, #2d2d2d 0%, #1a1a1a 100%); padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border: 1px solid #404040;'>
            <h3 style='color: #48cae4; margin: 0 0 0.5rem 0; font-size: 1.1rem;'>Profielgegevens</h3>
            <div style='height: 2px; background: linear-gradient(90deg, #48cae4 0%, #0077be 100%); margin-bottom: 1rem;'></div>
        </div>
    """, unsafe_allow_html=True)
    
    profile_type = st.selectbox(
        "Profieltype",
        ["Koker", "I-profiel", "U-profiel"],
        help="Selecteer het type profiel voor de berekening"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        height = st.number_input(
            "Hoogte (mm)",
            min_value=10,
            max_value=1000,
            value=100,
            help="Hoogte van het profiel in millimeters"
        )
    with col2:
        width = st.number_input(
            "Breedte (mm)",
            min_value=10,
            max_value=1000,
            value=50,
            help="Breedte van het profiel in millimeters"
        )
    
    col3, col4 = st.columns(2)
    with col3:
        wall_thickness = st.number_input(
            "Wanddikte (mm)",
            min_value=1,
            max_value=50,
            value=5,
            help="Dikte van de wand in millimeters"
        )
    with col4:
        if profile_type in ["I-profiel", "U-profiel"]:
            flange_thickness = st.number_input(
                "Flensdikte (mm)",
                min_value=1,
                max_value=50,
                value=8,
                help="Dikte van de flenzen in millimeters"
            )
        else:
            flange_thickness = wall_thickness

    st.markdown("""
        <div style='background: linear-gradient(180deg, #2d2d2d 0%, #1a1a1a 100%); padding: 1rem; border-radius: 8px; margin: 1rem 0; border: 1px solid #404040;'>
            <h3 style='color: #48cae4; margin: 0 0 0.5rem 0; font-size: 1.1rem;'>Overspanning</h3>
            <div style='height: 2px; background: linear-gradient(90deg, #48cae4 0%, #0077be 100%); margin-bottom: 1rem;'></div>
        </div>
    """, unsafe_allow_html=True)
    
    beam_length = st.number_input(
        "Lengte (mm)",
        min_value=100,
        max_value=10000,
        value=1000,
        help="Totale lengte van de balk in millimeters"
    )

    st.markdown("""
        <div style='background: linear-gradient(180deg, #2d2d2d 0%, #1a1a1a 100%); padding: 1rem; border-radius: 8px; margin: 1rem 0; border: 1px solid #404040;'>
            <h3 style='color: #48cae4; margin: 0 0 0.5rem 0; font-size: 1.1rem;'>Materiaal</h3>
            <div style='height: 2px; background: linear-gradient(90deg, #48cae4 0%, #0077be 100%); margin-bottom: 1rem;'></div>
        </div>
    """, unsafe_allow_html=True)
    
    E = st.number_input(
        "E-modulus (N/mm²)",
        min_value=1000,
        max_value=300000,
        value=210000,
        help="Elasticiteitsmodulus van het materiaal"
    )

    st.markdown("""
        <div style='background: linear-gradient(180deg, #2d2d2d 0%, #1a1a1a 100%); padding: 1rem; border-radius: 8px; margin: 1rem 0; border: 1px solid #404040;'>
            <h3 style='color: #48cae4; margin: 0 0 0.5rem 0; font-size: 1.1rem;'>Belastingen</h3>
            <div style='height: 2px; background: linear-gradient(90deg, #48cae4 0%, #0077be 100%); margin-bottom: 1rem;'></div>
        </div>
    """, unsafe_allow_html=True)

    # Belastingen interface
    load_type = st.selectbox(
        "Type belasting",
        ["Puntlast", "Gelijkmatig verdeeld", "Moment"],
        help="Selecteer het type belasting dat toegepast moet worden"
    )

    col5, col6 = st.columns(2)
    with col5:
        if load_type == "Gelijkmatig verdeeld":
            load_value = st.number_input(
                "Belasting (N/mm)",
                min_value=0.0,
                value=1.0,
                help="Grootte van de verdeelde belasting in N/mm"
            )
        else:
            load_value = st.number_input(
                "Belasting (N)",
                min_value=0,
                value=1000,
                help="Grootte van de puntlast in N"
            )
    
    with col6:
        load_pos = st.number_input(
            "Positie (mm)",
            min_value=0,
            max_value=beam_length,
            value=beam_length//2,
            help="Positie van de belasting vanaf het linker uiteinde"
        )

    if load_type == "Gelijkmatig verdeeld":
        load_length = st.number_input(
            "Lengte (mm)",
            min_value=10,
            max_value=beam_length,
            value=beam_length,
            help="Lengte waarover de belasting verdeeld is"
        )
    else:
        load_length = 0

    col7, col8 = st.columns(2)
    with col7:
        if st.button("Toevoegen", help="Voeg deze belasting toe aan de berekening"):
            st.session_state.loads.append((load_pos, load_value, load_type, load_length))
            st.session_state.load_count += 1
    
    with col8:
        if st.button("Reset", help="Verwijder alle belastingen"):
            st.session_state.loads = []
            st.session_state.load_count = 0

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

def plot_results(x, M, rotation, deflection):
    # Maak een moderne Plotly figure met subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            "<b>Doorbuiging</b>",
            "<b>Moment</b>",
            "<b>Rotatie</b>"
        ),
        vertical_spacing=0.12,
        specs=[[{"type": "scatter"}], [{"type": "scatter"}], [{"type": "scatter"}]]
    )

    # Voeg traces toe met moderne styling
    fig.add_trace(
        go.Scatter(
            x=x, y=deflection,
            name="Doorbuiging",
            line=dict(color='#48cae4', width=3),
            hovertemplate="<b>Positie:</b> %{x:.1f} mm<br><b>Doorbuiging:</b> %{y:.2f} mm<extra></extra>"
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=x, y=M,
            name="Moment",
            line=dict(color='#90e0ef', width=3),
            hovertemplate="<b>Positie:</b> %{x:.1f} mm<br><b>Moment:</b> %{y:.0f} Nmm<extra></extra>"
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=x, y=rotation,
            name="Rotatie",
            line=dict(color='#00b4d8', width=3),
            hovertemplate="<b>Positie:</b> %{x:.1f} mm<br><b>Rotatie:</b> %{y:.6f} rad<extra></extra>"
        ),
        row=3, col=1
    )

    # Update layout voor professionele uitstraling
    fig.update_layout(
        height=800,
        showlegend=False,
        paper_bgcolor='#2d2d2d',
        plot_bgcolor='#2d2d2d',
        font=dict(
            family="Segoe UI, sans-serif",
            color='#ffffff',
            size=12
        ),
        margin=dict(l=50, r=20, t=60, b=20),
        hovermode='x unified'
    )

    # Update alle assen voor consistente styling
    fig.update_xaxes(
        showgrid=True, gridwidth=1, gridcolor='rgba(144, 144, 144, 0.1)',
        showline=True, linewidth=2, linecolor='#909090',
        zeroline=True, zerolinewidth=2, zerolinecolor='#909090',
        color='#ffffff',
        title_text="Positie (mm)",
        title_font=dict(size=12, color='#909090'),
        tickfont=dict(size=10)
    )

    fig.update_yaxes(
        showgrid=True, gridwidth=1, gridcolor='rgba(144, 144, 144, 0.1)',
        showline=True, linewidth=2, linecolor='#909090',
        zeroline=True, zerolinewidth=2, zerolinecolor='#909090',
        color='#ffffff',
        title_font=dict(size=12, color='#909090'),
        tickfont=dict(size=10)
    )

    # Update y-as labels
    fig.update_yaxes(title_text="Doorbuiging (mm)", row=1, col=1)
    fig.update_yaxes(title_text="Moment (Nmm)", row=2, col=1)
    fig.update_yaxes(title_text="Rotatie (rad)", row=3, col=1)

    # Render plot in Streamlit met custom container
    st.markdown("""
        <div class='plot-container'>
            <h3 style='color: #48cae4; margin: 0 0 1rem 0; font-size: 1.2rem;'>Analyse Resultaten</h3>
        </div>
    """, unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'beam_analysis',
            'height': 800,
            'width': 1200,
            'scale': 2
        }
    })

    # Toon resultaten in moderne cards
    max_defl = np.max(np.abs(deflection))
    max_defl_pos = x[np.argmax(np.abs(deflection))]
    max_moment = np.max(np.abs(M))
    max_moment_pos = x[np.argmax(np.abs(M))]
    max_rot = np.max(np.abs(rotation))
    max_rot_pos = x[np.argmax(np.abs(rotation))]

    st.markdown("""
    <div class='grid-container'>
        <div class='grid-item'>
            <h3 style='color: #48cae4; margin: 0; font-size: 1.1rem;'>Maximale doorbuiging</h3>
            <p style='color: #ffffff; margin: 0.5rem 0; font-size: 1.5rem; font-weight: 600;'>{:.2f} mm</p>
            <p style='color: #909090; margin: 0;'>@ x = {:.1f} mm</p>
            <div style='height: 4px; background: linear-gradient(90deg, #48cae4 0%, #0077be 100%); margin-top: 1rem; border-radius: 2px;'></div>
        </div>
        <div class='grid-item'>
            <h3 style='color: #90e0ef; margin: 0; font-size: 1.1rem;'>Maximaal moment</h3>
            <p style='color: #ffffff; margin: 0.5rem 0; font-size: 1.5rem; font-weight: 600;'>{:.0f} Nmm</p>
            <p style='color: #909090; margin: 0;'>@ x = {:.1f} mm</p>
            <div style='height: 4px; background: linear-gradient(90deg, #90e0ef 0%, #48cae4 100%); margin-top: 1rem; border-radius: 2px;'></div>
        </div>
        <div class='grid-item'>
            <h3 style='color: #00b4d8; margin: 0; font-size: 1.1rem;'>Maximale rotatie</h3>
            <p style='color: #ffffff; margin: 0.5rem 0; font-size: 1.5rem; font-weight: 600;'>{:.6f} rad</p>
            <p style='color: #909090; margin: 0;'>@ x = {:.1f} mm</p>
            <div style='height: 4px; background: linear-gradient(90deg, #00b4d8 0%, #0077be 100%); margin-top: 1rem; border-radius: 2px;'></div>
        </div>
    </div>
    """.format(max_defl, max_defl_pos, max_moment, max_moment_pos, max_rot, max_rot_pos), unsafe_allow_html=True)

# Main content
with main:
    # Create tabs
    tab1, tab2, tab3 = st.tabs([" Analyse", " Resultaten", " Rapport"])
    
    with tab1:
        # Modern visualization container
        st.markdown('<div style="background-color: #252525; padding: 1.5rem; border-radius: 8px; border: 1px solid #333;">', unsafe_allow_html=True)
        
        # Steunpunten interface
        st.markdown("""
            <div style='background: linear-gradient(180deg, #2d2d2d 0%, #1a1a1a 100%); padding: 1rem; border-radius: 8px; margin: 1rem 0; border: 1px solid #404040;'>
                <h3 style='color: #48cae4; margin: 0 0 0.5rem 0; font-size: 1.1rem;'>Steunpunten</h3>
                <div style='height: 2px; background: linear-gradient(90deg, #48cae4 0%, #0077be 100%); margin-bottom: 1rem;'></div>
            </div>
        """, unsafe_allow_html=True)

        support_count = st.selectbox(
            "Aantal steunpunten",
            [1, 2, 3],
            help="Selecteer het aantal steunpunten voor de balk"
        )

        if support_count == 1:
            pos = st.slider(
                "Positie inklemming (mm)",
                min_value=0.0,
                max_value=float(beam_length),
                value=0.0,
                help="Positie van de inklemming vanaf het linker uiteinde"
            )
            supports = [(pos, "Inklemming")]
        else:
            supports = []
            for i in range(support_count):
                col1, col2 = st.columns(2)
                with col1:
                    pos = st.slider(
                        f"Positie steunpunt {i+1} (mm)",
                        min_value=0.0,
                        max_value=float(beam_length),
                        value=i * beam_length/(support_count-1) if support_count > 1 else 0.0,
                        help=f"Positie van steunpunt {i+1} vanaf het linker uiteinde",
                        key=f"support_pos_{i}"
                    )
                with col2:
                    support_type = st.selectbox(
                        f"Type steunpunt {i+1}",
                        ["Scharnier", "Rol", "Inklemming"],
                        help="Type ondersteuning voor dit steunpunt",
                        key=f"support_type_{i}"
                    )
                supports.append((pos, support_type))

        # Voer analyse uit
        x, M, rotation, deflection = analyze_beam(
            beam_length, supports, st.session_state.loads,
            profile_type, height, width, wall_thickness, flange_thickness, E
        )
        
        # Plot resultaten
        plot_results(x, M, rotation, deflection)
        
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
