import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

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
st.set_page_config(page_title="Buigingsberekeningen", layout="wide")
st.title("Buigingsberekeningen")

# Sidebar voor invoer
with st.sidebar:
    st.header("Profielgegevens")
    
    # Profieltype
    profile_type = st.selectbox(
        "Profieltype",
        ["Koker", "I-profiel", "U-profiel"]
    )
    
    # Basis afmetingen
    height = st.number_input("Hoogte (mm)", min_value=1.0, value=100.0)
    width = st.number_input("Breedte (mm)", min_value=1.0, value=50.0)
    wall_thickness = st.number_input("Wanddikte (mm)", min_value=0.1, value=5.0)
    
    # Flensdikte voor I- en U-profielen
    flange_thickness = None
    if profile_type in ["I-profiel", "U-profiel"]:
        flange_thickness = st.number_input("Flensdikte (mm)", min_value=0.1, value=5.0)
    
    # E-modulus
    E = st.number_input("E-modulus (N/mm²)", min_value=1.0, value=210000.0)
    
    st.header("Overspanning")
    beam_length = st.number_input("Lengte (mm)", min_value=1.0, value=1000.0)
    
    # Aantal steunpunten
    support_count = st.selectbox("Aantal steunpunten", [1, 2, 3])
    
    # Steunpunten configuratie
    supports = []
    if support_count == 1:
        st.subheader("Steunpunt (inklemming)")
        pos = st.number_input("Positie inklemming (mm)", 0.0, beam_length, 0.0)
        supports.append((pos, "Inklemming"))
    else:
        st.subheader("Steunpunten")
        for i in range(support_count):
            col1, col2 = st.columns(2)
            with col1:
                pos = st.number_input(f"Positie {i+1} (mm)", 0.0, beam_length, 
                                    value=i * beam_length/(support_count-1) if support_count > 1 else 0.0,
                                    key=f"pos_{i}")
            with col2:
                type = st.selectbox("Type", ["Scharnier", "Rol"], key=f"type_{i}")
            supports.append((pos, type))

    # Belastingen sectie
    st.header("Belastingen")
    if st.button("Voeg belasting toe"):
        if 'load_count' not in st.session_state:
            st.session_state.load_count = 0
        st.session_state.load_count += 1

    # Toon bestaande belastingen
    if 'load_count' not in st.session_state:
        st.session_state.load_count = 0
        
    loads = []
    for i in range(st.session_state.load_count):
        st.subheader(f"Belasting {i+1}")
        col1, col2 = st.columns(2)
        with col1:
            load_type = st.selectbox("Type", ["Puntlast", "Gelijkmatig verdeeld"], key=f"load_type_{i}")
        with col2:
            force = st.number_input("Waarde (N)", value=1000.0, key=f"force_{i}")
        
        pos = st.number_input("Positie (mm)", 0.0, beam_length, value=beam_length/2, key=f"load_pos_{i}")
        
        if load_type == "Gelijkmatig verdeeld":
            length = st.number_input("Lengte (mm)", 0.0, beam_length-pos, value=100.0, key=f"load_length_{i}")
            loads.append((pos, force, load_type, length))
        else:
            loads.append((pos, force, load_type))
            
    if st.button("Wis alle belastingen"):
        st.session_state.load_count = 0

# Bereken traagheidsmoment
I = calculate_I(profile_type, height, width, wall_thickness, flange_thickness)

# Hoofdgedeelte voor de plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot onvervormde balk
ax.plot([0, beam_length], [0, 0], 'k--', alpha=0.3)

# Bereken en plot vervormde balk
x = np.linspace(0, beam_length, 100)
y = np.zeros_like(x)
for i, xi in enumerate(x):
    y[i] = calculate_beam_response(xi, beam_length, E, I, supports, loads)

# Schaal doorbuiging voor visualisatie
scale = 1.0
if np.any(y != 0):
    max_defl = np.max(np.abs(y))
    if max_defl > 0:
        desired_height = beam_length / 10
        scale = desired_height / max_defl

# Plot vervormde balk
ax.plot(x, y * scale, 'b-', linewidth=2)

# Teken steunpunten
for pos, type in supports:
    if type == "Scharnier":
        # Driehoek voor scharnier
        triangle_height = beam_length * 0.02
        ax.plot([pos, pos - triangle_height, pos + triangle_height, pos],
                [0, -triangle_height, -triangle_height, 0], 'k-')
    elif type == "Rol":
        # Cirkel met driehoek voor rol
        circle = plt.Circle((pos, -beam_length * 0.02), beam_length * 0.01, fill=False, color='k')
        ax.add_artist(circle)
        triangle_height = beam_length * 0.02
        ax.plot([pos, pos - triangle_height, pos + triangle_height, pos],
                [0, -triangle_height, -triangle_height, 0], 'k-')
    else:  # Inklemming
        # Rechthoek voor inklemming
        rect_height = beam_length * 0.04
        rect_width = beam_length * 0.01
        ax.add_patch(plt.Rectangle((pos, -rect_height/2), rect_width, rect_height,
                                 color='k', alpha=0.3))

# Plot belastingen
arrow_height = beam_length * 0.05
for load in loads:
    pos = load[0]
    F = load[1]
    load_type = load[2]
    
    if load_type == "Puntlast":
        # Pijl voor puntlast
        if F > 0:  # Naar beneden
            ax.arrow(pos, arrow_height, 0, -arrow_height/2,
                    head_width=beam_length/50, head_length=arrow_height/4,
                    fc='r', ec='r', linewidth=2)
            ax.text(pos, arrow_height*1.1, f'{F:.0f}N', ha='center', va='bottom')
        else:  # Naar boven
            ax.arrow(pos, -arrow_height, 0, arrow_height/2,
                    head_width=beam_length/50, head_length=arrow_height/4,
                    fc='r', ec='r', linewidth=2)
            ax.text(pos, -arrow_height*1.1, f'{F:.0f}N', ha='center', va='top')
    
    elif load_type == "Gelijkmatig verdeeld":
        length = load[3]
        q = F / length  # N/mm
        arrow_spacing = length / 10
        
        # Teken pijlen voor verdeelde last
        for x in np.arange(pos, pos + length + arrow_spacing/2, arrow_spacing):
            if F > 0:  # Naar beneden
                ax.arrow(x, arrow_height/2, 0, -arrow_height/4,
                        head_width=beam_length/100, head_length=arrow_height/8,
                        fc='r', ec='r', linewidth=1)
            else:  # Naar boven
                ax.arrow(x, -arrow_height/2, 0, arrow_height/4,
                        head_width=beam_length/100, head_length=arrow_height/8,
                        fc='r', ec='r', linewidth=1)
        
        # Waarde van de verdeelde last
        mid_pos = pos + length/2
        if F > 0:
            ax.text(mid_pos, arrow_height*0.6, f'{q:.1f}N/mm', ha='center', va='bottom')
        else:
            ax.text(mid_pos, -arrow_height*0.6, f'{q:.1f}N/mm', ha='center', va='top')

# Plot instellingen
ax.grid(True)
ax.set_xlabel("Lengte (mm)")
ax.set_ylabel("Doorbuiging (mm)")
ax.set_xlim(-beam_length*0.1, beam_length*1.1)
ax.set_ylim(-beam_length*0.15, beam_length*0.15)
ax.set_aspect('equal', adjustable='box')

# Toon plot
st.pyplot(fig)

# Toon maximale doorbuiging
if len(loads) > 0:
    max_defl = np.max(np.abs(y))
    st.write(f"Maximale doorbuiging: {max_defl:.2f} mm")
