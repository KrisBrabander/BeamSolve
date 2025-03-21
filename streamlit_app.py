import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch

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

# Pagina configuratie
st.set_page_config(
    page_title="Buigingsberekeningen Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS voor een professionele look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .stTextInput>div>div>input {
        background-color: white;
    }
    h1 {
        color: #2c3e50;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 600;
    }
    .stMarkdown {
        font-family: 'Segoe UI', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

# Header met professionele styling
col1, col2 = st.columns([2, 1])
with col1:
    st.title("Buigingsberekeningen Pro")
    st.markdown("*Professionele balkdoorbuigingsanalyse*")
with col2:
    st.markdown("""
        <div style='background-color: #e3f2fd; padding: 1rem; border-radius: 4px; margin-top: 2rem;'>
            <h4 style='color: #1976d2; margin: 0;'>Professional Edition</h4>
            <p style='margin: 0.5rem 0 0 0; color: #424242;'>v1.0.0</p>
        </div>
    """, unsafe_allow_html=True)

# Sidebar met professionele styling
with st.sidebar:
    st.markdown("""
        <div style='background-color: #2c3e50; padding: 1rem; margin: -1rem -1rem 1rem -1rem; color: white;'>
            <h3 style='margin: 0;'>Configuratie</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # Profielgegevens
    st.subheader("🔧 Profielgegevens")
    profile_type = st.selectbox(
        "Profieltype",
        ["Koker", "I-profiel", "U-profiel"],
        help="Selecteer het type profiel voor de berekening"
    )
    
    with st.expander("📏 Afmetingen", expanded=True):
        height = st.number_input("Hoogte (mm)", min_value=1.0, value=100.0)
        width = st.number_input("Breedte (mm)", min_value=1.0, value=50.0)
        wall_thickness = st.number_input("Wanddikte (mm)", min_value=0.1, value=5.0)
        
        if profile_type in ["I-profiel", "U-profiel"]:
            flange_thickness = st.number_input("Flensdikte (mm)", min_value=0.1, value=5.0)
        else:
            flange_thickness = None
    
    with st.expander("🔨 Materiaal", expanded=True):
        E = st.number_input("E-modulus (N/mm²)", min_value=1.0, value=210000.0,
                          help="Elasticiteitsmodulus van het materiaal")
    
    # Overspanning
    st.subheader("📏 Overspanning")
    beam_length = st.number_input("Lengte (mm)", min_value=1.0, value=1000.0)
    
    # Steunpunten
    support_count = st.selectbox("Aantal steunpunten", [1, 2, 3])
    
    supports = []
    if support_count == 1:
        st.markdown("##### 🔒 Inklemming")
        pos = st.number_input("Positie inklemming (mm)", 0.0, beam_length, 0.0)
        supports.append((pos, "Inklemming"))
    else:
        st.markdown("##### 🔗 Steunpunten")
        for i in range(support_count):
            col1, col2 = st.columns(2)
            with col1:
                pos = st.number_input(f"Positie {i+1} (mm)", 0.0, beam_length, 
                                    value=i * beam_length/(support_count-1) if support_count > 1 else 0.0,
                                    key=f"pos_{i}")
            with col2:
                type = st.selectbox("Type", ["Scharnier", "Rol"], key=f"type_{i}")
            supports.append((pos, type))

# Hoofdgedeelte
col1, col2 = st.columns([7, 3])

with col1:
    st.markdown("### 📊 Visualisatie")
    
    # Maak een mooiere plot
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#ffffff')
    
    # Grid styling
    ax.grid(True, linestyle='--', alpha=0.7, color='#dcdcdc')
    
    # Plot onvervormde balk met mooiere stijl
    ax.plot([0, beam_length], [0, 0], 'k--', alpha=0.3, linewidth=1.5)
    
    # Bereken doorbuiging
    x = np.linspace(0, beam_length, 200)  # Meer punten voor vloeiendere curve
    y = np.zeros_like(x)
    for i, xi in enumerate(x):
        y[i] = -calculate_beam_response(xi, beam_length, E, calculate_I(profile_type, height, width, wall_thickness, flange_thickness), supports, [])  # Negeer de y voor correcte richting
    
    # Schaal doorbuiging
    scale = 1.0
    if np.any(y != 0):
        max_defl = np.max(np.abs(y))
        if max_defl > 0:
            desired_height = beam_length / 10
            scale = desired_height / max_defl
    
    # Plot vervormde balk met mooiere stijl
    ax.plot(x, y * scale, color='#2196f3', linewidth=2.5, label='Vervormde balk')
    
    # Teken steunpunten met verbeterde stijl
    for pos, type in supports:
        if type == "Scharnier":
            triangle_height = beam_length * 0.02
            ax.plot([pos, pos - triangle_height, pos + triangle_height, pos],
                    [0, -triangle_height, -triangle_height, 0],
                    color='#424242', linewidth=2)
        elif type == "Rol":
            circle = plt.Circle((pos, -beam_length * 0.02), beam_length * 0.01,
                              fill=False, color='#424242', linewidth=2)
            ax.add_artist(circle)
            triangle_height = beam_length * 0.02
            ax.plot([pos, pos - triangle_height, pos + triangle_height, pos],
                    [0, -triangle_height, -triangle_height, 0],
                    color='#424242', linewidth=2)
        else:  # Inklemming
            rect_height = beam_length * 0.04
            rect_width = beam_length * 0.01
            rect = patches.Rectangle((pos, -rect_height/2), rect_width, rect_height,
                                  color='#424242', alpha=0.8)
            ax.add_patch(rect)
    
    # Plot instellingen
    ax.set_xlabel("Lengte (mm)", fontsize=10, color='#424242')
    ax.set_ylabel("Doorbuiging (mm)", fontsize=10, color='#424242')
    ax.set_xlim(-beam_length*0.1, beam_length*1.1)
    ax.set_ylim(-beam_length*0.15, beam_length*0.15)
    ax.set_aspect('equal', adjustable='box')
    
    # Toon plot
    st.pyplot(fig)

with col2:
    st.markdown("### 🎯 Belastingen")
    
    # Belastingen sectie met verbeterde styling
    st.markdown("""
        <div style='background-color: #fff; padding: 1rem; border-radius: 4px; border: 1px solid #e0e0e0;'>
    """, unsafe_allow_html=True)
    
    if st.button("➕ Voeg belasting toe", key="add_load"):
        if 'load_count' not in st.session_state:
            st.session_state.load_count = 0
        st.session_state.load_count += 1
    
    # Toon bestaande belastingen
    if 'load_count' not in st.session_state:
        st.session_state.load_count = 0
    
    loads = []
    for i in range(st.session_state.load_count):
        with st.expander(f"📌 Belasting {i+1}", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                load_type = st.selectbox("Type", ["Puntlast", "Gelijkmatig verdeeld"],
                                       key=f"load_type_{i}")
            with col2:
                force = st.number_input("Waarde (N)", value=1000.0, key=f"force_{i}")
            
            pos = st.number_input("Positie (mm)", 0.0, beam_length,
                                value=beam_length/2, key=f"load_pos_{i}")
            
            if load_type == "Gelijkmatig verdeeld":
                length = st.number_input("Lengte (mm)", 0.0, beam_length-pos,
                                       value=100.0, key=f"load_length_{i}")
                loads.append((pos, force, load_type, length))
            else:
                loads.append((pos, force, load_type))
    
    if st.button("🗑️ Wis alle belastingen", key="clear_loads"):
        st.session_state.load_count = 0
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Resultaten sectie
    if len(loads) > 0:
        st.markdown("### 📊 Resultaten")
        st.markdown("""
            <div style='background-color: #fff; padding: 1rem; border-radius: 4px; border: 1px solid #e0e0e0;'>
        """, unsafe_allow_html=True)
        
        x = np.linspace(0, beam_length, 200)  # Meer punten voor vloeiendere curve
        y = np.zeros_like(x)
        for i, xi in enumerate(x):
            y[i] = -calculate_beam_response(xi, beam_length, E, calculate_I(profile_type, height, width, wall_thickness, flange_thickness), supports, loads)  # Negeer de y voor correcte richting
        
        max_defl = np.max(np.abs(y))
        st.metric(
            label="Maximale doorbuiging",
            value=f"{max_defl:.2f} mm",
            delta=f"{max_defl/beam_length*100:.2f}% van de lengte"
        )
        
        st.markdown("</div>", unsafe_allow_html=True)

