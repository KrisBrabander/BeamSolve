import streamlit as st
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import os
import base64

# Profiel bibliotheken
# HEA profielen (h, b, tw, tf)
HEA_PROFILES = {
    "HEA 100": (96, 100, 5.0, 8.0),
    "HEA 120": (114, 120, 5.0, 8.0),
    "HEA 140": (133, 140, 5.5, 8.5),
    "HEA 160": (152, 160, 6.0, 9.0),
    "HEA 180": (171, 180, 6.0, 9.5),
    "HEA 200": (190, 200, 6.5, 10.0),
    "HEA 220": (210, 220, 7.0, 11.0),
    "HEA 240": (230, 240, 7.5, 12.0),
    "HEA 260": (250, 260, 7.5, 12.5),
    "HEA 280": (270, 280, 8.0, 13.0),
    "HEA 300": (290, 300, 8.5, 14.0),
}

# HEB profielen (h, b, tw, tf)
HEB_PROFILES = {
    "HEB 100": (100, 100, 6.0, 10.0),
    "HEB 120": (120, 120, 6.5, 11.0),
    "HEB 140": (140, 140, 7.0, 12.0),
    "HEB 160": (160, 160, 8.0, 13.0),
    "HEB 180": (180, 180, 8.5, 14.0),
    "HEB 200": (200, 200, 9.0, 15.0),
    "HEB 220": (220, 220, 9.5, 16.0),
    "HEB 240": (240, 240, 10.0, 17.0),
    "HEB 260": (260, 260, 10.0, 17.5),
    "HEB 280": (280, 280, 10.5, 18.0),
    "HEB 300": (300, 300, 11.0, 19.0),
}

# IPE profielen (h, b, tw, tf)
IPE_PROFILES = {
    "IPE 80": (80, 46, 3.8, 5.2),
    "IPE 100": (100, 55, 4.1, 5.7),
    "IPE 120": (120, 64, 4.4, 6.3),
    "IPE 140": (140, 73, 4.7, 6.9),
    "IPE 160": (160, 82, 5.0, 7.4),
    "IPE 180": (180, 91, 5.3, 8.0),
    "IPE 200": (200, 100, 5.6, 8.5),
    "IPE 220": (220, 110, 5.9, 9.2),
    "IPE 240": (240, 120, 6.2, 9.8),
    "IPE 270": (270, 135, 6.6, 10.2),
    "IPE 300": (300, 150, 7.1, 10.7),
}

# UNP profielen (h, b, tw, tf)
UNP_PROFILES = {
    "UNP 80": (80, 45, 6.0, 8.0),
    "UNP 100": (100, 50, 6.0, 8.5),
    "UNP 120": (120, 55, 7.0, 9.0),
    "UNP 140": (140, 60, 7.0, 10.0),
    "UNP 160": (160, 65, 7.5, 10.5),
    "UNP 180": (180, 70, 8.0, 11.0),
    "UNP 200": (200, 75, 8.5, 11.5),
    "UNP 220": (220, 80, 9.0, 12.5),
    "UNP 240": (240, 85, 9.5, 13.0),
}

# Koker profielen (h, b, t)
KOKER_PROFILES = {
    "Koker 40x40x3": (40, 40, 3.0),
    "Koker 50x50x3": (50, 50, 3.0),
    "Koker 60x60x3": (60, 60, 3.0),
    "Koker 60x60x4": (60, 60, 4.0),
    "Koker 70x70x3": (70, 70, 3.0),
    "Koker 70x70x4": (70, 70, 4.0),
    "Koker 80x80x3": (80, 80, 3.0),
    "Koker 80x80x4": (80, 80, 4.0),
    "Koker 80x80x5": (80, 80, 5.0),
    "Koker 90x90x3": (90, 90, 3.0),
    "Koker 90x90x4": (90, 90, 4.0),
}

def get_profile_dimensions(profile_type, profile_name):
    """Haal de dimensies op voor een specifiek profiel"""
    if profile_type == "HEA":
        return HEA_PROFILES.get(profile_name)
    elif profile_type == "HEB":
        return HEB_PROFILES.get(profile_name)
    elif profile_type == "IPE":
        return IPE_PROFILES.get(profile_name)
    elif profile_type == "UNP":
        return UNP_PROFILES.get(profile_name)
    elif profile_type == "Koker":
        return KOKER_PROFILES.get(profile_name)
    return None

def get_profile_list(profile_type):
    """Krijg een lijst van alle profielen van een bepaald type"""
    if profile_type == "HEA":
        return list(HEA_PROFILES.keys())
    elif profile_type == "HEB":
        return list(HEB_PROFILES.keys())
    elif profile_type == "IPE":
        return list(IPE_PROFILES.keys())
    elif profile_type == "UNP":
        return list(UNP_PROFILES.keys())
    elif profile_type == "Koker":
        return list(KOKER_PROFILES.keys())
    return []

# Initialize session state
if 'loads' not in st.session_state:
    st.session_state.loads = []
if 'load_count' not in st.session_state:
    st.session_state.load_count = 0
if 'supports' not in st.session_state:
    st.session_state.supports = []
if 'calculations' not in st.session_state:
    st.session_state.calculations = []
if 'units' not in st.session_state:
    st.session_state.units = {
        'length': 'mm',
        'force': 'N',
        'stress': 'MPa'
    }

def main():
    st.set_page_config(page_title="BeamSolve Professional", layout="wide")
    
    # Sidebar voor profiel selectie
    with st.sidebar:
        st.title("BeamSolve Professional")
        st.markdown("---")
        
        # Profiel selectie
        profile_type = st.selectbox("Profieltype", ["HEA", "HEB", "IPE", "UNP", "Koker"])
        profile_name = st.selectbox("Profiel", get_profile_list(profile_type))
        
        # Haal profiel dimensies op
        dimensions = get_profile_dimensions(profile_type, profile_name)
        if dimensions:
            if profile_type == "Koker":
                height, width, wall_thickness = dimensions
                flange_thickness = wall_thickness
            else:
                height, width, wall_thickness, flange_thickness = dimensions
        
        # Toon dimensies
        st.markdown("### Profiel Dimensies")
        st.write(f"Hoogte: {height} mm")
        st.write(f"Breedte: {width} mm")
        st.write(f"Wanddikte: {wall_thickness} mm")
        if profile_type != "Koker":
            st.write(f"Flensdikte: {flange_thickness} mm")
        
        # E-modulus
        E = st.number_input("E-modulus (N/mm²)", value=210000.0, step=1000.0)
        
        st.markdown("---")
        st.markdown(" BeamSolve Professional")

    # Hoofdgedeelte
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Balk Configuratie")
        
        # Overspanning
        beam_length = st.number_input("Overspanning (mm)", value=3000.0, step=100.0)
        
        # Steunpunten
        st.subheader("Steunpunten")
        supports = []
        num_supports = st.number_input("Aantal steunpunten", min_value=2, max_value=4, value=2)
        
        for i in range(num_supports):
            col_pos, col_type = st.columns(2)
            with col_pos:
                pos = st.number_input(f"Positie {i+1} (mm)", value=0.0 if i == 0 else beam_length if i == 1 else beam_length/2)
            with col_type:
                type = st.selectbox(f"Type {i+1}", ["Vast", "Rol"], index=0 if i == 0 else 1)
            supports.append((pos, type))
        
        # Belastingen
        st.subheader("Belastingen")
        loads = []
        num_loads = st.number_input("Aantal belastingen", min_value=1, max_value=5, value=1)
        
        for i in range(num_loads):
            col_type, col_val, col_pos = st.columns(3)
            with col_type:
                load_type = st.selectbox(f"Type {i+1}", ["Puntlast", "Verdeelde last", "Moment"], key=f"load_type_{i}")
            with col_val:
                value = st.number_input(f"Waarde {i+1} (N)", value=1000.0, step=100.0, key=f"load_value_{i}")
            with col_pos:
                position = st.number_input(f"Positie {i+1} (mm)", value=beam_length/2, step=100.0, key=f"load_pos_{i}")
            
            if load_type == "Verdeelde last":
                length = st.number_input(f"Lengte {i+1} (mm)", value=1000.0, step=100.0, key=f"load_length_{i}")
                loads.append((position, value, load_type, length))
            else:
                loads.append((position, value, load_type))
    
    with col2:
        st.header("Analyse")
        if st.button("Bereken"):
            # Verzamel alle gegevens
            beam_data = {
                "profile_type": profile_type,
                "profile_name": profile_name,
                "height": height,
                "width": width,
                "wall_thickness": wall_thickness,
                "flange_thickness": flange_thickness,
                "length": beam_length,
                "E": E,
                "supports": supports,
                "loads": loads
            }
            
            # Voer analyse uit
            x, M, rotation, deflection = analyze_beam(beam_length, supports, loads, profile_type, height, width, wall_thickness, flange_thickness, E)
            
            # Toon resultaten
            st.subheader("Resultaten")
            max_moment = max(abs(np.min(M)), abs(np.max(M)))
            max_deflection = max(abs(np.min(deflection)), abs(np.max(deflection)))
            max_rotation = max(abs(np.min(rotation)), abs(np.max(rotation)))
            st.write(f"Maximaal moment: {max_moment:.2f} Nmm")
            st.write(f"Maximale doorbuiging: {max_deflection:.2f} mm")
            st.write(f"Maximale rotatie: {max_rotation:.6f} rad")
            
            # Maak plots
            beam_plot = plot_beam_diagram(beam_length, supports, loads, x, deflection)
            analysis_plot = plot_results(x, M, rotation, deflection)
            
            st.plotly_chart(beam_plot, use_container_width=True)
            st.plotly_chart(analysis_plot, use_container_width=True)
            
            # Genereer rapport
            if st.button("Genereer Rapport"):
                html_content = generate_report_html(beam_data, {
                    "max_moment": max_moment,
                    "max_deflection": max_deflection,
                    "max_rotation": max_rotation
                }, [beam_plot, analysis_plot])
                output_dir = "reports"
                os.makedirs(output_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(output_dir, f"beamsolve_report_{timestamp}.html")
                save_report(html_content, output_path)
                st.success(f"Rapport opgeslagen als: {output_path}")
                
if __name__ == "__main__":
    main()

def calculate_I(profile_type, h, b, t_w, t_f=None):
    """Bereken traagheidsmoment voor verschillende profieltypes"""
    if profile_type == "Koker":
        h_i = h - 2*t_w
        b_i = b - 2*t_w
        return (b*h**3)/12 - (b_i*h_i**3)/12
    elif profile_type in ["I-profiel", "U-profiel"]:
        # Flens bijdrage
        I_f = 2 * (b*t_f**3/12 + b*t_f*(h/2 - t_f/2)**2)
        # Lijf bijdrage
        h_w = h - 2*t_f
        I_w = t_w*h_w**3/12
        return I_f + I_w
    return 0

def plot_results(x, M, rotation, deflection):
    """Plot resultaten met Plotly"""
    fig = go.Figure(data=[
        go.Scatter(x=x, y=deflection, name="Doorbuiging"),
        go.Scatter(x=x, y=M, name="Moment"),
        go.Scatter(x=x, y=rotation, name="Rotatie")
    ])
    
    fig.update_layout(
        title="Analyse Resultaten",
        xaxis_title="Positie (mm)",
        yaxis_title="Waarde",
        legend_title="Groep"
    )
    
    return fig

def plot_beam_diagram(beam_length, supports, loads, x=None, deflection=None):
    """Plot een schematische weergave van de balk met steunpunten en belastingen"""
    fig = go.Figure()
    
    # Teken de balk
    if x is not None and deflection is not None:
        # Teken vervormde balk
        scale_factor = beam_length / (20 * max(abs(np.max(deflection)), abs(np.min(deflection))) if np.any(deflection != 0) else 1)
        fig.add_trace(go.Scatter(
            x=x,
            y=deflection * scale_factor,
            mode='lines',
            name='Vervormde balk',
            line=dict(color='#2c3e50', width=8),
            hovertemplate="Doorbuiging: %{y:.2f} mm<br>x = %{x:.0f} mm"
        ))
        # Teken onvervormde balk gestippeld
        fig.add_trace(go.Scatter(
            x=[0, beam_length],
            y=[0, 0],
            mode='lines',
            name='Onvervormde balk',
            line=dict(color='#95a5a6', width=3, dash='dash'),
            hoverinfo='skip'
        ))
    else:
        # Teken alleen onvervormde balk
        fig.add_trace(go.Scatter(
            x=[0, beam_length],
            y=[0, 0],
            mode='lines',
            name='Balk',
            line=dict(color='#2c3e50', width=8),
            hoverinfo='skip'
        ))
    
    # Teken steunpunten
    for pos, support_type in supports:
        y_pos = deflection[np.abs(x - pos).argmin()] * scale_factor if x is not None and deflection is not None else 0
        
        if support_type == "Inklemming":
            # Teken rechthoek voor inklemming
            fig.add_trace(go.Scatter(
                x=[pos-30, pos-30, pos+30, pos+30],
                y=[y_pos-60, y_pos+60, y_pos+60, y_pos-60],
                fill="toself",
                mode='lines',
                name='Inklemming',
                line=dict(color='#2ecc71', width=3),
                hovertemplate=f"Inklemming<br>x = {pos} mm"
            ))
            # Voeg arcering toe
            for i in range(-50, 51, 20):
                fig.add_trace(go.Scatter(
                    x=[pos-30, pos+30],
                    y=[y_pos+i, y_pos+i],
                    mode='lines',
                    line=dict(color='#2ecc71', width=2),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        elif support_type == "Scharnier":
            # Teken driehoek voor scharnier
            fig.add_trace(go.Scatter(
                x=[pos-30, pos, pos+30],
                y=[y_pos-60, y_pos, y_pos-60],
                fill="toself",
                mode='lines',
                name='Scharnier',
                line=dict(color='#3498db', width=3),
                hovertemplate=f"Scharnier<br>x = {pos} mm"
            ))
            # Voeg cirkels toe voor scharnier
            theta = np.linspace(0, 2*np.pi, 50)
            r = 8
            x_circle = r * np.cos(theta) + pos
            y_circle = r * np.sin(theta) + y_pos-60
            fig.add_trace(go.Scatter(
                x=x_circle,
                y=y_circle,
                mode='lines',
                line=dict(color='#3498db', width=2),
                fill='toself',
                showlegend=False,
                hoverinfo='skip'
            ))
        else:  # Rol
            # Teken driehoek voor rol
            fig.add_trace(go.Scatter(
                x=[pos-30, pos, pos+30],
                y=[y_pos-60, y_pos, y_pos-60],
                fill="toself",
                mode='lines',
                name='Rol',
                line=dict(color='#e74c3c', width=3),
                hovertemplate=f"Rol<br>x = {pos} mm"
            ))
            # Teken cirkels voor rol
            for offset in [-10, 0, 10]:
                theta = np.linspace(0, 2*np.pi, 50)
                r = 8
                x_circle = r * np.cos(theta) + pos + offset
                y_circle = r * np.sin(theta) + y_pos-70
                fig.add_trace(go.Scatter(
                    x=x_circle,
                    y=y_circle,
                    mode='lines',
                    line=dict(color='#e74c3c', width=2),
                    fill='toself',
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    # Teken belastingen
    for pos, F, load_type, *rest in loads:
        y_pos = deflection[np.abs(x - pos).argmin()] * scale_factor if x is not None and deflection is not None else 0
        
        if load_type == "Puntlast":
            # Teken pijl voor puntlast
            arrow_length = 80 if F > 0 else -80
            fig.add_trace(go.Scatter(
                x=[pos, pos],
                y=[y_pos, y_pos + arrow_length],
                mode='lines',
                name=f'Puntlast {F}N',
                line=dict(color='#e67e22', width=3),
                hovertemplate=f"Puntlast<br>F = {F} N<br>x = {pos} mm"
            ))
            # Teken pijlpunt
            if F > 0:
                fig.add_trace(go.Scatter(
                    x=[pos-10, pos, pos+10],
                    y=[y_pos + arrow_length+10, y_pos + arrow_length, y_pos + arrow_length+10],
                    mode='lines',
                    line=dict(color='#e67e22', width=3),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=[pos-10, pos, pos+10],
                    y=[y_pos + arrow_length-10, y_pos + arrow_length, y_pos + arrow_length-10],
                    mode='lines',
                    line=dict(color='#e67e22', width=3),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        elif load_type == "Gelijkmatig verdeeld":
            # Teken meerdere pijlen voor verdeelde belasting
            length = float(rest[0])
            q = F / length  # Last per lengte-eenheid
            # Bereken reactiekrachten voor verdeelde belasting
            start = pos
            end = pos + length
            if end > beam_length:  # Begrens tot einde van de balk
                end = beam_length
            if start < 0:  # Begin vanaf begin van de balk
                start = 0
                
            load_length = end - start
            load_center = start + load_length/2
            total_load = q * load_length
            
            n_arrows = min(int(load_length/50) + 1, 15)
            dx = load_length / (n_arrows - 1)
            arrow_length = 60 if F > 0 else -60
            
            # Teken lijn boven de pijlen
            x_load = np.linspace(start, end, n_arrows)
            y_load = np.array([deflection[np.abs(x - xi).argmin()] * scale_factor for xi in x_load])
            
            fig.add_trace(go.Scatter(
                x=x_load,
                y=y_load + arrow_length,
                mode='lines',
                line=dict(color='#9b59b6', width=3),
                name=f'q = {q:.1f} N/mm',
                hovertemplate=f"Verdeelde last<br>q = {q:.1f} N/mm"
            ))
            
            # Teken pijlen
            for i in range(n_arrows):
                x_pos = start + i * dx
                if x_pos <= beam_length:
                    y_pos = deflection[np.abs(x - x_pos).argmin()] * scale_factor
                    fig.add_trace(go.Scatter(
                        x=[x_pos, x_pos],
                        y=[y_pos, y_pos + arrow_length],
                        mode='lines',
                        line=dict(color='#9b59b6', width=2),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    # Teken pijlpunt
                    if F > 0:
                        fig.add_trace(go.Scatter(
                            x=[x_pos-8, x_pos, x_pos+8],
                            y=[y_pos + arrow_length+8, y_pos + arrow_length, y_pos + arrow_length+8],
                            mode='lines',
                            line=dict(color='#9b59b6', width=2),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                    else:
                        fig.add_trace(go.Scatter(
                            x=[x_pos-8, x_pos, x_pos+8],
                            y=[y_pos + arrow_length-8, y_pos + arrow_length, y_pos + arrow_length-8],
                            mode='lines',
                            line=dict(color='#9b59b6', width=2),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
    
    # Update layout
    fig.update_layout(
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        margin=dict(l=20, r=20, t=50, b=20),
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white',
        title=dict(
            text="Balkschema",
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top',
            font=dict(size=20, color='#2c3e50')
        ),
        xaxis=dict(
            title="Positie (mm)",
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0,0,0,0.1)',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='#2c3e50',
            showline=True,
            linewidth=2,
            linecolor='#2c3e50',
            mirror=True,
            range=[-beam_length*0.1, beam_length*1.1]
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            showticklabels=False,
            range=[-150, 150]
        )
    )
    
    return fig

def analyze_beam(beam_length, supports, loads, profile_type, height, width, wall_thickness, flange_thickness, E):
    """Analyseer de balk en bereken momenten, dwarskrachten, rotaties en doorbuigingen"""
    n_points = 200
    x = np.linspace(0, beam_length, n_points)
    dx = x[1] - x[0]
    
    # Initialiseer arrays
    M = np.zeros_like(x)  # Moment
    V = np.zeros_like(x)  # Dwarskracht
    rotation = np.zeros_like(x)  # Rotatie
    deflection = np.zeros_like(x)  # Doorbuiging
    
    # Bereken traagheidsmoment
    I = calculate_I(profile_type, height, width, wall_thickness, flange_thickness)
    
    # Sorteer steunpunten op positie
    supports = sorted(supports, key=lambda s: s[0])
    
    # Bereken reactiekrachten
    R = np.zeros(len(supports))  # Reactiekrachten bij steunpunten
    
    # Matrix voor oplegreacties
    A = np.zeros((len(supports), len(supports)))
    b = np.zeros(len(supports))
    
    # Vul matrix A en vector b voor momentenvergelijkingen
    for i, (pos_i, type_i) in enumerate(supports):
        for j, (pos_j, type_j) in enumerate(supports):
            # Coëfficiënt voor reactiekracht j in momentenvergelijking om steunpunt i
            A[i,j] = pos_j - pos_i
        
        # Bereken moment van externe krachten om steunpunt i
        for load in loads:
            if len(load) == 3:  # Puntlast of moment
                pos, value, load_type = load
                if load_type == "Puntlast":
                    b[i] -= value * (pos - pos_i)
                elif load_type == "Moment":
                    b[i] -= value
            else:  # Verdeelde last
                pos, value, load_type, length = load
                # Resultante van verdeelde last werkt aan in midden van belaste lengte
                center = pos + length/2
                total_force = value * length
                b[i] -= total_force * (center - pos_i)
    
    # Los reactiekrachten op
    R = np.linalg.solve(A, b)
    
    # Bereken dwarskracht en moment
    for i, xi in enumerate(x):
        # Bijdrage van reactiekrachten
        for j, (pos, _) in enumerate(supports):
            if xi >= pos:
                V[i] += R[j]
                M[i] += R[j] * (xi - pos)
        
        # Bijdrage van belastingen
        for load in loads:
            if len(load) == 3:  # Puntlast of moment
                pos, value, load_type = load
                if xi >= pos:
                    if load_type == "Puntlast":
                        V[i] -= value
                        M[i] -= value * (xi - pos)
                    elif load_type == "Moment":
                        M[i] -= value
            else:  # Verdeelde last
                pos, value, load_type, length = load
                if xi >= pos:
                    # Bereken totale last tot dit punt
                    overlap = min(xi - pos, length)
                    if overlap > 0:
                        force = value * overlap
                        center = pos + overlap/2
                        V[i] -= force
                        M[i] -= force * (xi - center)
    
    # Bereken rotatie en doorbuiging door dubbele integratie
    for i in range(1, len(x)):
        # Integreer M/EI voor rotatie
        rotation[i] = rotation[i-1] + (M[i-1] / (E * I)) * dx
        # Integreer rotatie voor doorbuiging
        deflection[i] = deflection[i-1] + rotation[i-1] * dx
    
    # Corrigeer voor randvoorwaarden
    # Vind vaste steunpunten
    fixed_supports = [(i, pos) for i, (pos, type) in enumerate(supports) if type == "Vast"]
    
    if fixed_supports:
        # Pas rotatie en doorbuiging aan voor vaste steunpunten
        for idx, pos in fixed_supports:
            # Vind dichtstbijzijnde x-waarde
            support_idx = np.argmin(np.abs(x - pos))
            rotation_correction = rotation[support_idx]
            deflection_correction = deflection[support_idx]
            
            # Corrigeer alle waarden
            rotation -= rotation_correction
            deflection -= deflection_correction
    
    return x, M, rotation, deflection

def generate_report_html(beam_data, results, plots):
    """Genereer een HTML rapport"""
    
    # Converteer plots naar base64 images
    plot_images = []
    for plot in plots:
        img_bytes = pio.to_image(plot, format="png")
        img_base64 = base64.b64encode(img_bytes).decode()
        plot_images.append(f"data:image/png;base64,{img_base64}")
    
    # HTML template
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            .header {{
                text-align: center;
                padding: 20px;
                background: #f8f9fa;
                margin-bottom: 30px;
                border-radius: 8px;
            }}
            .section {{
                margin-bottom: 30px;
                padding: 20px;
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #f8f9fa;
            }}
            img {{
                max-width: 100%;
                height: auto;
                margin: 20px 0;
            }}
            .footer {{
                text-align: center;
                padding: 20px;
                color: #666;
                font-size: 0.9em;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>BeamSolve Professional</h1>
            <h2>Technisch Rapport</h2>
            <p>Gegenereerd op: {datetime.now().strftime('%d-%m-%Y %H:%M')}</p>
        </div>

        <div class="section">
            <h3>1. Invoergegevens</h3>
            <table>
                <tr><th>Parameter</th><th>Waarde</th></tr>
                <tr><td>Profieltype</td><td>{beam_data['profile_type']}</td></tr>
                <tr><td>Hoogte</td><td>{beam_data['height']} mm</td></tr>
                <tr><td>Breedte</td><td>{beam_data['width']} mm</td></tr>
                <tr><td>Wanddikte</td><td>{beam_data['wall_thickness']} mm</td></tr>
                <tr><td>Overspanning</td><td>{beam_data['length']} mm</td></tr>
                <tr><td>E-modulus</td><td>{beam_data['E']} N/mm²</td></tr>
            </table>
        </div>

        <div class="section">
            <h3>2. Steunpunten</h3>
            <table>
                <tr><th>Positie</th><th>Type</th></tr>
                {chr(10).join([f'<tr><td>{pos} mm</td><td>{type}</td></tr>' for pos, type in beam_data['supports']])}
            </table>
        </div>

        <div class="section">
            <h3>3. Belastingen</h3>
            <table>
                <tr><th>Type</th><th>Waarde</th><th>Positie</th><th>Lengte</th></tr>
                {chr(10).join([f'<tr><td>{load[2]}</td><td>{load[1]} N</td><td>{load[0]} mm</td><td>{load[3] if len(load) > 3 else "-"} mm</td></tr>' for load in beam_data['loads']])}
            </table>
        </div>

        <div class="section">
            <h3>4. Resultaten</h3>
            <table>
                <tr><th>Parameter</th><th>Waarde</th></tr>
                <tr><td>Maximaal moment</td><td>{results['max_moment']:.2f} Nmm</td></tr>
                <tr><td>Maximale doorbuiging</td><td>{results['max_deflection']:.2f} mm</td></tr>
                <tr><td>Maximale rotatie</td><td>{results['max_rotation']:.6f} rad</td></tr>
            </table>
        </div>

        <div class="section">
            <h3>5. Grafieken</h3>
            <h4>5.1 Balkschema</h4>
            <img src="{plot_images[0]}" alt="Balkschema">
            
            <h4>5.2 Analyse Grafieken</h4>
            <img src="{plot_images[1]}" alt="Analyse">
        </div>

        <div class="footer">
            <p>BeamSolve Professional {datetime.now().year}</p>
        </div>
    </body>
    </html>
    """
    
    return html

def save_report(html_content, output_path):
    """Sla het rapport op als HTML bestand"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    return output_path
