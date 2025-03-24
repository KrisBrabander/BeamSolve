import streamlit as st
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import os
import base64
from plotly.subplots import make_subplots

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

def calculate_A(profile_type, h, b, t_w, t_f=None):
    """Bereken oppervlakte voor verschillende profieltypes"""
    if profile_type == "Koker":
        return (h * b) - ((h - 2*t_w) * (b - 2*t_w))
    elif profile_type in ["I-profiel", "U-profiel"]:
        # Flens oppervlakte
        A_f = 2 * (b * t_f)
        # Lijf oppervlakte
        h_w = h - 2*t_f
        A_w = t_w * h_w
        return A_f + A_w
    return 0

def plot_beam_diagram(beam_length, supports, loads):
    """Teken professioneel balkschema"""
    fig = go.Figure()
    
    # Teken balk - dikker en duidelijker
    fig.add_trace(go.Scatter(
        x=[0, beam_length/1000],
        y=[0, 0],
        mode='lines',
        line=dict(color='black', width=4),
        name='Balk'
    ))
    
    # Teken steunpunten
    for pos, type in supports:
        x_pos = pos/1000  # Convert to meters
        triangle_size = beam_length/50
        type = type.lower()  # Convert to lowercase for consistent comparison
        if type == "inklemming":
            # Inklemming (driehoek met arcering)
            # Onderste deel
            fig.add_trace(go.Scatter(
                x=[x_pos-triangle_size/1000, x_pos+triangle_size/1000, x_pos, x_pos-triangle_size/1000],
                y=[-triangle_size/1000, -triangle_size/1000, 0, -triangle_size/1000],
                fill="toself",
                mode='lines',
                line=dict(color='black', width=2),
                fillcolor='lightgray',
                name='Inklemming'
            ))
            # Arcering lijnen onder
            for offset in np.linspace(-triangle_size/1000, triangle_size/1000, 5):
                fig.add_trace(go.Scatter(
                    x=[x_pos+offset-triangle_size/2000, x_pos+offset+triangle_size/2000],
                    y=[-triangle_size/1000, -triangle_size/2000],
                    mode='lines',
                    line=dict(color='black', width=1),
                    showlegend=False
                ))
            # Bovenste deel
            fig.add_trace(go.Scatter(
                x=[x_pos-triangle_size/1000, x_pos+triangle_size/1000, x_pos, x_pos-triangle_size/1000],
                y=[triangle_size/1000, triangle_size/1000, 0, triangle_size/1000],
                fill="toself",
                mode='lines',
                line=dict(color='black', width=2),
                fillcolor='lightgray',
                showlegend=False
            ))
            # Arcering lijnen boven
            for offset in np.linspace(-triangle_size/1000, triangle_size/1000, 5):
                fig.add_trace(go.Scatter(
                    x=[x_pos+offset-triangle_size/2000, x_pos+offset+triangle_size/2000],
                    y=[triangle_size/1000, triangle_size/2000],
                    mode='lines',
                    line=dict(color='black', width=1),
                    showlegend=False
                ))
        elif type == "scharnier":
            # Scharnier (driehoek)
            fig.add_trace(go.Scatter(
                x=[x_pos-triangle_size/1000, x_pos+triangle_size/1000, x_pos, x_pos-triangle_size/1000],
                y=[-triangle_size/1000, -triangle_size/1000, 0, -triangle_size/1000],
                fill="toself",
                mode='lines',
                line=dict(color='black', width=2),
                fillcolor='white',
                name='Scharnier'
            ))
        elif type == "rol":
            # Scharnier (driehoek)
            fig.add_trace(go.Scatter(
                x=[x_pos-triangle_size/1000, x_pos+triangle_size/1000, x_pos, x_pos-triangle_size/1000],
                y=[-triangle_size/1000, -triangle_size/1000, 0, -triangle_size/1000],
                fill="toself",
                mode='lines',
                line=dict(color='black', width=2),
                fillcolor='white',
                name='Rol'
            ))
            # Rol (cirkel)
            circle_radius = triangle_size/2000
            theta = np.linspace(0, 2*np.pi, 50)
            circle_x = x_pos + circle_radius * np.cos(theta)
            circle_y = -triangle_size/1000 - circle_radius + circle_radius * np.sin(theta)
            fig.add_trace(go.Scatter(
                x=circle_x,
                y=circle_y,
                mode='lines',
                line=dict(color='black', width=1),
                fill='toself',
                fillcolor='white',
                showlegend=False
            ))
    
    # Teken belastingen
    for load in loads:
        pos, value, load_type, *rest = load
        x_pos = pos/1000  # Convert to meters
        if load_type == "Puntlast":
            # Puntlast pijl
            arrow_height = beam_length/30
            fig.add_trace(go.Scatter(
                x=[x_pos, x_pos],
                y=[arrow_height/1000, 0],
                mode='lines+text',
                text=[f'{value/1000:.1f} kN', ''],
                textposition='top center',
                line=dict(color='blue', width=2),
                name='Puntlast'
            ))
            # Pijlpunt
            fig.add_trace(go.Scatter(
                x=[x_pos-arrow_height/2000, x_pos, x_pos+arrow_height/2000],
                y=[arrow_height/2000, 0, arrow_height/2000],
                mode='lines',
                line=dict(color='blue', width=2),
                showlegend=False
            ))
        elif load_type == "Verdeelde last":
            length = rest[0]/1000  # Convert to meters
            arrow_height = beam_length/40
            # Pijlen voor verdeelde last
            num_arrows = min(max(int(length*5), 3), 10)  # Minimaal 3, maximaal 10 pijlen
            for i in range(num_arrows):
                arrow_x = x_pos + (i * length/(num_arrows-1))
                fig.add_trace(go.Scatter(
                    x=[arrow_x, arrow_x],
                    y=[arrow_height/1000, 0],
                    mode='lines',
                    line=dict(color='blue', width=1),
                    showlegend=(i==0),
                    name=f'{value/1000:.1f} kN/m'
                ))
                # Pijlpunt
                fig.add_trace(go.Scatter(
                    x=[arrow_x-arrow_height/4000, arrow_x, arrow_x+arrow_height/4000],
                    y=[arrow_height/2000, 0, arrow_height/2000],
                    mode='lines',
                    line=dict(color='blue', width=1),
                    showlegend=False
                ))
            # Lijn boven pijlen
            fig.add_trace(go.Scatter(
                x=[x_pos, x_pos+length],
                y=[arrow_height/1000, arrow_height/1000],
                mode='lines',
                line=dict(color='blue', width=2),
                showlegend=False
            ))
    
    # Update layout voor professionele uitstraling
    fig.update_layout(
        title="Balkschema en Belastingen",
        height=300,  # Kleiner voor betere overzichtelijkheid
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white',
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1,
            range=[-beam_length/20/1000, beam_length/20/1000],  # Beter zichtbare schaal
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='black',
            showgrid=False
        ),
        xaxis=dict(
            range=[-beam_length/20/1000, beam_length*1.1/1000],
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='black',
            showgrid=False
        ),
        margin=dict(t=50, b=50),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor='rgba(255,255,255,0.8)'
        )
    )
    
    return fig

def plot_results(x, V, M, rotation, deflection):
    """Plot alle resultaten in één figuur met subplots"""
    # Maak subplot layout
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=('Doorbuiging (mm)', 'Rotatie (rad)', 'Dwarskracht (kN)', 'Moment (kNm)'),
        vertical_spacing=0.08,
        shared_xaxes=True
    )
    
    # Doorbuiging (bovenste plot)
    fig.add_trace(
        go.Scatter(
            x=x/1000,  # Convert to meters
            y=deflection,  # Already in mm
            mode='lines',
            name='Doorbuiging',
            line=dict(color='#2980b9', width=2),
            fill='tozeroy',
            fillcolor='rgba(41, 128, 185, 0.3)'
        ),
        row=1, col=1
    )
    
    # Rotatie (tweede plot)
    fig.add_trace(
        go.Scatter(
            x=x/1000,  # Convert to meters
            y=rotation,  # In radians
            mode='lines',
            name='Rotatie',
            line=dict(color='#e67e22', width=2),
            fill='tozeroy',
            fillcolor='rgba(230, 126, 34, 0.3)'
        ),
        row=2, col=1
    )
    
    # Dwarskrachtenlijn (derde plot)
    fig.add_trace(
        go.Scatter(
            x=x/1000,  # Convert to meters
            y=[v/1000 for v in V],  # Convert to kN
            mode='lines',
            name='Dwarskracht',
            line=dict(color='#27ae60', width=2),
            fill='tozeroy',
            fillcolor='rgba(46, 204, 113, 0.3)'
        ),
        row=3, col=1
    )
    
    # Momentenlijn (onderste plot)
    fig.add_trace(
        go.Scatter(
            x=x/1000,  # Convert to meters
            y=[m/1000000 for m in M],  # Convert to kNm
            mode='lines',
            name='Moment',
            line=dict(color='#8e44ad', width=2),
            fill='tozeroy',
            fillcolor='rgba(142, 68, 173, 0.3)'
        ),
        row=4, col=1
    )
    
    # Update layout voor professionele uitstraling
    fig.update_layout(
        height=900,
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor='rgba(255, 255, 255, 0.8)'
        )
    )
    
    # Update x-assen
    for i in range(1, 5):
        fig.update_xaxes(
            row=i, col=1,
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0,0,0,0.1)',
            zeroline=True,
            zerolinecolor='rgba(0,0,0,0.2)',
            zerolinewidth=2,
            dtick=1  # 1m intervallen
        )
    
    # Update y-assen
    for i in range(1, 5):
        fig.update_yaxes(
            row=i, col=1,
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0,0,0,0.1)',
            zeroline=True,
            zerolinecolor='rgba(0,0,0,0.2)',
            zerolinewidth=2
        )
    
    # Voeg waarden toe bij belangrijke punten
    max_defl_idx = np.argmax(np.abs(deflection))
    max_rot_idx = np.argmax(np.abs(rotation))
    max_v_idx = np.argmax(np.abs(V))
    max_m_idx = np.argmax(np.abs(M))
    
    # Annotaties voor maximale waarden
    fig.add_annotation(
        x=x[max_defl_idx]/1000,
        y=deflection[max_defl_idx],
        text=f"{deflection[max_defl_idx]:.2f} mm",
        showarrow=True,
        arrowhead=2,
        row=1, col=1
    )
    
    fig.add_annotation(
        x=x[max_rot_idx]/1000,
        y=rotation[max_rot_idx],
        text=f"{rotation[max_rot_idx]:.4f} rad",
        showarrow=True,
        arrowhead=2,
        row=2, col=1
    )
    
    fig.add_annotation(
        x=x[max_v_idx]/1000,
        y=V[max_v_idx]/1000,
        text=f"{V[max_v_idx]/1000:.1f} kN",
        showarrow=True,
        arrowhead=2,
        row=3, col=1
    )
    
    fig.add_annotation(
        x=x[max_m_idx]/1000,
        y=M[max_m_idx]/1000000,
        text=f"{M[max_m_idx]/1000000:.1f} kNm",
        showarrow=True,
        arrowhead=2,
        row=4, col=1
    )
    
    # Update x-as label alleen op onderste plot
    fig.update_xaxes(title_text="Positie (m)", row=4, col=1)
    
    return fig

def calculate_reactions(beam_length, supports, loads):
    """Bereken reactiekrachten voor alle belastingen met moment evenwicht"""
    # Sorteer steunpunten op positie
    supports = sorted(supports, key=lambda x: x[0])
    
    # We hebben minimaal 1 steunpunt nodig
    if len(supports) < 1:
        return {}
    
    # Tel het aantal onbekenden (krachten en momenten)
    n_unknowns = len(supports)  # Verticale reacties
    n_fixed = sum(1 for _, type in supports if type.lower() == "inklemming")
    n_unknowns += n_fixed  # Extra onbekenden voor inklemming (momenten)
    
    # Bepaal aantal vergelijkingen
    n_equations = min(3, n_unknowns)  # Max 3: ΣF=0, ΣM=0, en extra voor hyperstatisch
    
    # Maak matrices voor het stelsel
    A = np.zeros((n_equations, n_unknowns))
    b = np.zeros(n_equations)
    
    # Vul matrix voor som krachten (eerste rij)
    for i in range(len(supports)):
        A[0, i] = 1.0  # Verticale reacties
    
    # Vul matrix voor momentevenwicht (tweede rij)
    if n_equations > 1:
        for i, (pos, _) in enumerate(supports):
            A[1, i] = pos  # Moment van verticale reacties
        
        # Voeg momenten toe voor inklemmingen
        j = len(supports)
        for i, (_, type) in enumerate(supports):
            if type.lower() == "inklemming":
                A[1, j] = 1.0  # Directe momenten
                j += 1
    
    # Extra vergelijking voor hyperstatisch systeem
    if n_equations > 2:
        # Gebruik compatibiliteit voor extra vergelijking
        # Voor nu: verdeel de last gelijk over de steunpunten
        for i in range(len(supports)):
            A[2, i] = 1.0 / len(supports)
    
    # Bereken belastingstermen
    for load in loads:
        pos = load[0]
        value = load[1]
        load_type = load[2]
        
        if load_type == "Puntlast":
            b[0] += value  # Som krachten
            if n_equations > 1:
                b[1] += value * pos  # Moment
            
        elif load_type == "Verdeelde last":
            length = load[3]
            total_force = value * length
            force_pos = pos + length/2
            b[0] += total_force
            if n_equations > 1:
                b[1] += total_force * force_pos
            
        elif load_type == "Driehoekslast":
            length = load[3]
            total_force = 0.5 * value * length
            force_pos = pos + 2*length/3
            b[0] += total_force
            if n_equations > 1:
                b[1] += total_force * force_pos
            
        elif load_type == "Moment":
            if n_equations > 1:
                b[1] += value
    
    try:
        # Los het stelsel op
        x = np.linalg.solve(A, b)
        
        # Maak dictionary met reacties
        reactions = {}
        for i, (pos, type) in enumerate(supports):
            reactions[pos] = {"force": x[i], "moment": 0.0}
        
        # Voeg momenten toe voor inklemmingen
        j = len(supports)
        for i, (pos, type) in enumerate(supports):
            if type.lower() == "inklemming" and j < len(x):
                reactions[pos]["moment"] = x[j]
                j += 1
        
        return reactions
    except np.linalg.LinAlgError:
        return {pos: {"force": 0.0, "moment": 0.0} for pos, _ in supports}

def calculate_internal_forces(x, beam_length, supports, loads, reactions):
    """Bereken dwarskracht en moment op elke positie x"""
    V = np.zeros_like(x)  # Dwarskracht array
    M = np.zeros_like(x)  # Moment array
    
    # Verwerk reactiekrachten
    for pos, reaction in reactions.items():
        # Dwarskracht van reactiekracht
        V += reaction["force"] * (x >= pos)
        # Moment van reactiekracht
        M += reaction["force"] * np.maximum(0, x - pos)
        # Direct moment van inklemming
        M += reaction["moment"] * (x >= pos)
    
    # Verwerk belastingen
    for load in loads:
        pos = load[0]
        value = load[1]
        load_type = load[2]
        
        if load_type == "Puntlast":
            V -= value * (x >= pos)
            M -= value * np.maximum(0, x - pos)
            
        elif load_type == "Verdeelde last":
            length = load[3]
            end_pos = pos + length
            # Belast gebied
            mask = (x >= pos) & (x <= end_pos)
            # Dwarskracht
            V[mask] -= value * (x[mask] - pos)
            V[x > end_pos] -= value * length
            # Moment
            M[mask] -= value * (x[mask] - pos)**2 / 2
            M[x > end_pos] -= value * length * (x[x > end_pos] - (pos + length/2))
            
        elif load_type == "Driehoekslast":
            length = load[3]
            end_pos = pos + length
            # Belast gebied
            mask = (x >= pos) & (x <= end_pos)
            # Relatieve x-positie
            rel_x = (x[mask] - pos) / length
            # Dwarskracht
            V[mask] -= value * length * rel_x**2 / 2
            V[x > end_pos] -= value * length / 2
            # Moment
            M[mask] -= value * length * rel_x**3 / 6
            M[x > end_pos] -= value * length * (x[x > end_pos] - (pos + 2*length/3)) / 2
            
        elif load_type == "Moment":
            M -= value * (x >= pos)
    
    return V, M

def analyze_beam(beam_length, supports, loads, profile_type, height, width, wall_thickness, flange_thickness, E):
    """Analyseer de balk en bereken dwarskrachten, momenten, rotatie en doorbuiging"""
    # Aantal punten voor berekening
    n_points = 2001
    x = np.linspace(0, beam_length, n_points)
    dx = x[1] - x[0]
    
    # Bereken profiel eigenschappen
    A, I, W = calculate_profile_properties(profile_type, height, width, wall_thickness, flange_thickness)
    EI = E * I
    
    # Bereken reactiekrachten
    reactions = calculate_reactions(beam_length, supports, loads)
    
    # Bereken interne krachten
    V, M = calculate_internal_forces(x, beam_length, supports, loads, reactions)
    
    # Initialiseer arrays
    rotation = np.zeros_like(x)
    deflection = np.zeros_like(x)
    
    # Sorteer steunpunten op positie
    supports = sorted(supports, key=lambda x: x[0])
    support_indices = [np.abs(x - pos).argmin() for pos, _ in supports]
    
    # Eerste integratie: M/EI -> θ
    for i in range(1, len(x)):
        rotation[i] = rotation[i-1] + (M[i-1] + M[i])/(2 * EI) * dx
    
    # Tweede integratie: θ -> v
    for i in range(1, len(x)):
        deflection[i] = deflection[i-1] + (rotation[i-1] + rotation[i])/2 * dx
    
    # Pas randvoorwaarden toe
    if len(support_indices) > 0:
        # Matrix voor randvoorwaarden
        n_equations = len(support_indices)
        if any(type.lower() == "inklemming" for _, type in supports):
            n_equations += 1  # Extra vergelijking voor rotatie bij inklemming
        
        A = np.zeros((n_equations, 2))  # 2 onbekenden: C1 en C2
        b = np.zeros(n_equations)
        
        # Vergelijkingen voor doorbuiging bij steunpunten
        for i, idx in enumerate(support_indices):
            A[i, 0] = 1  # C1 coefficient
            A[i, 1] = x[idx]  # C2 coefficient
            b[i] = -deflection[idx]  # Tegengestelde van huidige doorbuiging
        
        # Extra vergelijking voor inklemming indien aanwezig
        for pos, type in supports:
            if type.lower() == "inklemming":
                idx = np.abs(x - pos).argmin()
                if idx == 0:  # Inklemming aan begin
                    A[-1, 1] = 1  # C2 is de rotatie
                    b[-1] = -rotation[0]
                    break
                elif idx == len(x)-1:  # Inklemming aan eind
                    A[-1, 1] = 1
                    b[-1] = -rotation[-1]
                    break
        
        try:
            # Los het stelsel op
            C = np.linalg.solve(A, b)
            # Pas correcties toe
            deflection += C[0] + C[1] * x  # C1 + C2*x
            rotation += C[1]  # C2 is de rotatie correctie
        except np.linalg.LinAlgError:
            # Fallback: forceer nul bij eerste steunpunt
            deflection -= deflection[support_indices[0]]
    
    # Forceer exacte waarden bij steunpunten
    for pos, type in supports:
        idx = np.abs(x - pos).argmin()
        deflection[idx] = 0.0  # Altijd nul doorbuiging bij steunpunt
        if type.lower() == "inklemming":
            rotation[idx] = 0.0  # Nul rotatie bij inklemming
    
    return x, V, M, rotation, deflection

@st.cache_data
def calculate_profile_properties(profile_type, height, width, wall_thickness, flange_thickness):
    """Cache profiel eigenschappen voor snellere berekeningen"""
    # Map profile types to calculation types
    calc_type = "Koker" if profile_type == "Koker" else ("U-profiel" if profile_type == "UNP" else "I-profiel")
    
    A = calculate_A(calc_type, height, width, wall_thickness, flange_thickness)
    I = calculate_I(calc_type, height, width, wall_thickness, flange_thickness)
    W = I / (height/2) if height > 0 else 0
    return A, I, W

def generate_pdf_report(beam_data, results_plot):
    """Genereer een PDF rapport"""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from io import BytesIO
    import base64
    
    # Converteer plot naar afbeelding
    img_bytes = results_plot.to_image(format="png", width=800, height=600)
    plot_img = BytesIO(img_bytes)
    
    # Maak PDF document
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=20*mm, leftMargin=20*mm, topMargin=20*mm, bottomMargin=20*mm)
    
    # Opmaakstijlen
    styles = getSampleStyleSheet()
    title_style = styles['Heading1']
    heading_style = styles['Heading2']
    normal_style = styles['Normal']
    
    # Document elementen
    elements = []
    
    # Titel
    elements.append(Paragraph("BeamSolve Professional - Berekeningsrapport", title_style))
    elements.append(Spacer(1, 12))
    
    # Profiel informatie
    elements.append(Paragraph("Profiel Gegevens", heading_style))
    profile_data = [
        ["Profiel Type", beam_data["profile_type"]],
        ["Hoogte", f"{beam_data['dimensions']['height']} mm"],
        ["Breedte", f"{beam_data['dimensions']['width']} mm"],
        ["Wanddikte", f"{beam_data['dimensions']['wall_thickness']} mm"]
    ]
    if "flange_thickness" in beam_data["dimensions"]:
        profile_data.append(["Flensdikte", f"{beam_data['dimensions']['flange_thickness']} mm"])
    
    profile_table = Table(profile_data, colWidths=[100, 200])
    profile_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.25, colors.black),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('PADDING', (0, 0), (-1, -1), 6)
    ]))
    elements.append(profile_table)
    elements.append(Spacer(1, 12))
    
    # Profiel eigenschappen
    elements.append(Paragraph("Profiel Eigenschappen", heading_style))
    properties_data = [
        ["Oppervlakte", f"{beam_data['properties']['A']:.0f} mm²"],
        ["Traagheidsmoment", f"{beam_data['properties']['I']:.0f} mm⁴"],
        ["Weerstandsmoment", f"{beam_data['properties']['W']:.0f} mm³"]
    ]
    properties_table = Table(properties_data, colWidths=[100, 200])
    properties_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.25, colors.black),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('PADDING', (0, 0), (-1, -1), 6)
    ]))
    elements.append(properties_table)
    elements.append(Spacer(1, 12))
    
    # Algemene gegevens
    elements.append(Paragraph("Algemene Gegevens", heading_style))
    general_data = [
        ["Lengte", f"{beam_data['length']:.0f} mm"],
        ["E-modulus", f"{beam_data['E']:.0f} N/mm²"]
    ]
    general_table = Table(general_data, colWidths=[100, 200])
    general_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.25, colors.black),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('PADDING', (0, 0), (-1, -1), 6)
    ]))
    elements.append(general_table)
    elements.append(Spacer(1, 12))
    
    # Resultaten
    elements.append(Paragraph("Resultaten", heading_style))
    results_data = [
        ["Max. Dwarskracht", f"{beam_data['results']['max_V']:.1f} kN"],
        ["Max. Moment", f"{beam_data['results']['max_M']:.1f} kNm"],
        ["Max. Doorbuiging", f"{beam_data['results']['max_deflection']:.2f} mm"],
        ["Max. Rotatie", f"{beam_data['results']['max_rotation']:.4f} rad"],
        ["Max. Spanning", f"{beam_data['results']['max_stress']:.1f} N/mm²"],
        ["Unity Check", f"{beam_data['results']['unity_check']:.2f}"]
    ]
    results_table = Table(results_data, colWidths=[100, 200])
    results_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.25, colors.black),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('PADDING', (0, 0), (-1, -1), 6)
    ]))
    elements.append(results_table)
    elements.append(Spacer(1, 12))
    
    # Grafieken
    elements.append(Paragraph("Grafieken", heading_style))
    img = Image(plot_img, width=160*mm, height=120*mm)
    elements.append(img)
    
    # Genereer PDF
    doc.build(elements)
    return buffer.getvalue()

def save_report(report_content, output_path):
    """Sla het rapport op"""
    with open(output_path, 'wb') as f:
        f.write(report_content)

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
    
    # Test voorbeeld (zoals in de afbeelding)
    if st.sidebar.button("Laad Testvoorbeeld", type="secondary"):
        st.session_state.test_example = True
        # Balk van 18m met 3 steunpunten
        st.session_state.beam_length = 18000  # 18m in mm
        st.session_state.supports = [
            (3000, "Scharnier"),   # C op 3m
            (9000, "Scharnier"),   # D op 9m
            (15000, "Scharnier"),  # B op 15m
        ]
        st.session_state.loads = [
            # Driehoekslast van 50 kN/m over 4m
            (3000, 50, "Driehoekslast", 4000),
            # Verdeelde last van 20 kN/m over rest
            (9000, 20, "Verdeelde last", 6000),
            # Puntlast van 100 kN
            (9000, 100, "Puntlast")
        ]
        st.session_state.profile_type = "HEA"
        st.session_state.profile_name = "HEA 300"
    
    # Sidebar voor invoer
    with st.sidebar:
        st.title("BeamSolve Professional")
        st.markdown("---")
        
        # Profiel selectie
        st.subheader("1. Profiel")
        col1, col2 = st.columns(2)
        with col1:
            profile_type = st.selectbox("Type", ["HEA", "HEB", "IPE", "UNP", "Koker"])
        with col2:
            profile_name = st.selectbox("Naam", get_profile_list(profile_type))
        
        # Haal profiel dimensies op
        dimensions = get_profile_dimensions(profile_type, profile_name)
        if dimensions:
            if profile_type == "Koker":
                height, width, wall_thickness = dimensions
                flange_thickness = wall_thickness
            else:
                height, width, wall_thickness, flange_thickness = dimensions
        
        # Toon dimensies
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Hoogte", f"{height} mm")
            st.metric("Breedte", f"{width} mm")
        with col2:
            st.metric("Wanddikte", f"{wall_thickness} mm")
            if profile_type != "Koker":
                st.metric("Flensdikte", f"{flange_thickness} mm")
        
        # E-modulus
        E = st.number_input("E-modulus", value=210000.0, step=1000.0, format="%.0f", help="N/mm²")
        
        st.markdown("---")
        
        # Overspanning
        st.subheader("2. Overspanning")
        beam_length = st.number_input("Lengte", value=3000.0, step=100.0, format="%.0f", help="mm")
        
        # Steunpunten
        st.subheader("3. Steunpunten")
        num_supports = st.number_input("Aantal", min_value=1, max_value=4, value=1)
        
        supports = []
        for i in range(num_supports):
            col1, col2 = st.columns(2)
            with col1:
                pos = st.number_input(
                    f"Positie {i+1}",
                    value=0.0 if i == 0 else beam_length if i == 1 else beam_length/2,
                    min_value=0.0,
                    max_value=beam_length,
                    step=100.0,
                    format="%.0f",
                    help="mm"
                )
            with col2:
                type = st.selectbox(
                    f"Type {i+1}",
                    ["Inklemming", "Scharnier", "Rol"],
                    index=0 if i == 0 else 1
                )
            supports.append((pos, type))
        
        # Belastingen
        st.subheader("4. Belastingen")
        num_loads = st.number_input("Aantal", min_value=0, max_value=5, value=1)
        
        loads = []
        for i in range(num_loads):
            st.markdown(f"**Belasting {i+1}**")
            
            col1, col2 = st.columns(2)
            with col1:
                load_type = st.selectbox(
                    "Type",
                    ["Puntlast", "Verdeelde last", "Moment", "Driehoekslast"],
                    key=f"load_type_{i}"
                )
            with col2:
                if load_type == "Moment":
                    unit = "Nmm"
                elif load_type in ["Verdeelde last", "Driehoekslast"]:
                    unit = "N/mm"
                else:
                    unit = "N"
                    
                value = st.number_input(
                    f"Waarde ({unit})",
                    value=1000.0,
                    step=100.0,
                    format="%.1f",
                    key=f"load_value_{i}"
                )
            
            col1, col2 = st.columns(2)
            with col1:
                position = st.number_input(
                    "Positie",
                    value=beam_length/2,
                    min_value=0.0,
                    max_value=beam_length,
                    step=100.0,
                    format="%.0f",
                    help="mm",
                    key=f"load_pos_{i}"
                )
            
            if load_type in ["Verdeelde last", "Driehoekslast"]:
                with col2:
                    length = st.number_input(
                        "Lengte",
                        value=1000.0,
                        min_value=0.0,
                        max_value=beam_length - position,
                        step=100.0,
                        format="%.0f",
                        help="mm",
                        key=f"load_length_{i}"
                    )
                loads.append((position, value, load_type, length))
            else:
                loads.append((position, value, load_type))
    
    # Hoofdgedeelte
    if st.sidebar.button("Bereken", type="primary", use_container_width=True):
        # Voer analyse uit
        x, V, M, rotation, deflection = analyze_beam(beam_length, supports, loads, profile_type, height, width, wall_thickness, flange_thickness, E)
        
        # Teken balkschema
        beam_fig = plot_beam_diagram(beam_length, supports, loads)
        st.plotly_chart(beam_fig, use_container_width=True)
        
        # Toon resultaten
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Analyse Resultaten")
            
            # Plot alle resultaten
            results_plot = plot_results(x, V, M, rotation, deflection)
            st.plotly_chart(results_plot, use_container_width=True)
            
            # Maximale waarden
            max_vals = {
                "Dwarskracht": f"{max(abs(np.min(V)), abs(np.max(V))):.0f} N",
                "Moment": f"{max(abs(np.min(M)), abs(np.max(M))):.0f} Nmm",
                "Rotatie": f"{max(abs(np.min(rotation)), abs(np.max(rotation))):.6f} rad",
                "Doorbuiging": f"{max(abs(np.min(deflection)), abs(np.max(deflection))):.2f} mm"
            }
            
            st.subheader("Maximale Waarden")
            cols = st.columns(4)
            for i, (key, val) in enumerate(max_vals.items()):
                cols[i].metric(key, val)
        
        with col2:
            st.subheader("Profiel Details")
            A, I, W = calculate_profile_properties(profile_type, height, width, wall_thickness, flange_thickness)
            
            st.metric("Oppervlakte", f"{A:.0f} mm²")
            st.metric("Traagheidsmoment", f"{I:.0f} mm⁴")
            st.metric("Weerstandsmoment", f"{W:.0f} mm³")
            
            # Spanningen
            st.subheader("Spanningen")
            max_moment = max(abs(min(M)), abs(max(M)))
            sigma = max_moment / W
            st.metric("Max. buigspanning", f"{sigma:.1f} N/mm²")
            
            # Toetsing
            st.subheader("Toetsing")
            f_y = 235  # Vloeigrens S235
            UC = sigma / f_y
            st.metric("Unity Check", f"{UC:.2f}", help="UC ≤ 1.0")
            
            if UC > 1.0:
                st.error("Profiel voldoet niet! Kies een zwaarder profiel.")
            elif UC > 0.9:
                st.warning("Profiel zwaar belast. Overweeg een zwaarder profiel.")
            else:
                st.success("Profiel voldoet ruim.")
            
            # Download rapport
            st.markdown("---")
            if st.button("Genereer Rapport", type="primary"):
                # Genereer rapport
                try:
                    beam_data = {
                        "profile_type": profile_type,
                        "profile_name": profile_name,
                        "dimensions": {
                            "height": height,
                            "width": width,
                            "wall_thickness": wall_thickness,
                            "flange_thickness": flange_thickness
                        },
                        "properties": {
                            "A": A,
                            "I": I,
                            "W": W
                        },
                        "length": beam_length,
                        "E": E,
                        "supports": supports,
                        "loads": loads,
                        "results": {
                            "max_V": max(abs(np.min(V)), abs(np.max(V))),
                            "max_M": max(abs(np.min(M)), abs(np.max(M))),
                            "max_deflection": max(abs(np.min(deflection)), abs(np.max(deflection))),
                            "max_rotation": max(abs(np.min(rotation)), abs(np.max(rotation))),
                            "max_stress": sigma,
                            "unity_check": UC
                        }
                    }
                    
                    # Genereer PDF
                    pdf_content = generate_pdf_report(beam_data, results_plot)
                    
                    # Sla op en toon download knop
                    st.download_button(
                        label="⬇️ Download Rapport (PDF)",
                        data=pdf_content,
                        file_name="beamsolve_report.pdf",
                        mime="application/pdf",
                        key="download_report"
                    )
                except Exception as e:
                    st.error(f"Fout bij genereren rapport: {str(e)}")

if __name__ == "__main__":
    main()
