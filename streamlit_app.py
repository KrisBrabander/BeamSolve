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
        if type == "vast":
            # Vaste oplegging (driehoek met arcering)
            triangle_size = beam_length/50
            fig.add_trace(go.Scatter(
                x=[x_pos-triangle_size/1000, x_pos+triangle_size/1000, x_pos, x_pos-triangle_size/1000],
                y=[-triangle_size/1000, -triangle_size/1000, 0, -triangle_size/1000],
                fill="toself",
                mode='lines',
                line=dict(color='black', width=2),
                fillcolor='rgb(200,200,200)',
                name='Vast steunpunt'
            ))
            # Arcering
            for i in range(3):
                offset = (i-1) * triangle_size/3000
                fig.add_trace(go.Scatter(
                    x=[x_pos+offset-triangle_size/2000, x_pos+offset+triangle_size/2000],
                    y=[-triangle_size/1000, -triangle_size/2000],
                    mode='lines',
                    line=dict(color='black', width=1),
                    showlegend=False
                ))
        else:
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
            # Rol (cirkel)
            fig.add_trace(go.Scatter(
                x=[x_pos-triangle_size/2000, x_pos+triangle_size/2000],
                y=[-triangle_size/1000-triangle_size/2000]*2,
                mode='lines',
                line=dict(color='black', width=1),
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
    
    # Layout
    margin = beam_length/20
    fig.update_layout(
        showlegend=True,
        legend=dict(
            x=0,
            y=1.1,
            orientation='h'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            range=[-margin/1000, (beam_length+margin)/1000],
            title='Positie [m]',
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='black',
            dtick=1  # 1m intervallen
        ),
        yaxis=dict(
            range=[-beam_length/15000, beam_length/15000],
            scaleanchor='x',
            scaleratio=1,
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        height=300,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

def plot_results(x, V, M, rotation, deflection):
    """Plot alle resultaten in één figuur met subplots, vergelijkbaar met professionele software"""
    # Maak een figuur met subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            "Balkschema en Belastingen",
            "Dwarskrachtenlijn (kN)",
            "Momentenlijn (kNm)"
        ),
        vertical_spacing=0.12,
        row_heights=[0.4, 0.3, 0.3]
    )
    
    # Balkschema (bovenste plot)
    # Teken de balk zelf
    fig.add_trace(
        go.Scatter(
            x=x/1000,  # Convert to meters
            y=[0]*len(x),
            mode='lines',
            name='Balk',
            line=dict(color='#3498db', width=6),
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Dwarskrachtenlijn (middelste plot)
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
        row=2, col=1
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
        row=3, col=1
    )
    
    # Update layout voor professionele uitstraling
    fig.update_layout(
        height=900,
        showlegend=True,
        plot_bgcolor='rgba(240,240,240,0.8)',
        paper_bgcolor='white',
        font=dict(size=12),
        margin=dict(t=100),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255, 255, 255, 0.8)'
        )
    )
    
    # Update x-assen
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(255,255,255,0.9)',
        zeroline=True,
        zerolinecolor='rgba(0,0,0,0.2)',
        title_text="Positie (m)",
        dtick=1  # Markering elke meter
    )
    
    # Update y-assen
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(255,255,255,0.9)',
        zeroline=True,
        zerolinecolor='rgba(0,0,0,0.5)',
        row=1, col=1
    )
    
    # Specifieke y-as labels
    fig.update_yaxes(title_text="", row=1, col=1)  # Geen y-as label voor balkschema
    fig.update_yaxes(title_text="V (kN)", row=2, col=1)
    fig.update_yaxes(title_text="M (kNm)", row=3, col=1)
    
    # Voeg waarden toe bij belangrijke punten
    max_shear = max(abs(min(V)), abs(max(V))) / 1000
    max_moment = max(abs(min(M)), abs(max(M))) / 1000000
    
    # Voeg annotaties toe voor maximale waarden
    max_v_idx = np.argmax(np.abs(V))
    max_m_idx = np.argmax(np.abs(M))
    
    fig.add_annotation(
        x=x[max_v_idx]/1000,  # Convert to meters
        y=V[max_v_idx]/1000,
        text=f"{V[max_v_idx]/1000:.1f} kN",
        showarrow=True,
        arrowhead=2,
        row=2, col=1
    )
    
    fig.add_annotation(
        x=x[max_m_idx]/1000,  # Convert to meters
        y=M[max_m_idx]/1000000,
        text=f"{M[max_m_idx]/1000000:.1f} kNm",
        showarrow=True,
        arrowhead=2,
        row=3, col=1
    )
    
    return fig

def calculate_reactions(beam_length, supports, loads):
    """Bereken reactiekrachten voor alle belastingen"""
    # Sorteer steunpunten op positie
    supports = sorted(supports, key=lambda x: x[0])
    
    # We hebben minimaal 2 steunpunten nodig
    if len(supports) < 2:
        return {}
    
    # Maak matrix voor reactiekrachten
    n_supports = len(supports)
    A = np.zeros((2, n_supports))  # 2 vergelijkingen: som krachten en som momenten
    b = np.zeros(2)
    
    # Vul matrix voor som krachten (eerste rij)
    A[0, :] = 1.0
    
    # Vul matrix voor som momenten t.o.v. eerste steunpunt (tweede rij)
    for i in range(n_supports):
        A[1, i] = supports[i][0]  # Afstand tot eerste steunpunt
    
    # Bereken belastingstermen
    for load in loads:
        pos = load[0]
        value = load[1]
        load_type = load[2]
        
        if load_type == "Puntlast":
            b[0] += value  # Som krachten
            b[1] += value * pos  # Som momenten
            
        elif load_type == "Moment":
            b[1] += value  # Alleen effect op momenten
            
        elif load_type in ["Verdeelde last", "Driehoekslast"]:
            length = load[3]
            if load_type == "Verdeelde last":
                total_force = value * length
                force_pos = pos + length/2
            else:  # Driehoekslast
                total_force = 0.5 * value * length
                force_pos = pos + 2*length/3
                
            b[0] += total_force
            b[1] += total_force * force_pos
    
    try:
        # Los reactiekrachten op
        reactions = np.linalg.solve(A, b)
        
        # Maak dictionary met reactiekrachten
        reaction_dict = {}
        for i, support in enumerate(supports):
            reaction_dict[support[0]] = reactions[i]
        
        return reaction_dict
    except np.linalg.LinAlgError:
        # Als het stelsel niet oplosbaar is
        return {support[0]: 0.0 for support in supports}

def calculate_internal_forces(x, beam_length, supports, loads, reactions):
    """Bereken dwarskracht en moment op elke positie x"""
    V = np.zeros_like(x)  # Dwarskracht array
    M = np.zeros_like(x)  # Moment array
    dx = x[1] - x[0]
    
    # Verwerk alle krachten in één keer met numpy operaties
    for pos, force in reactions.items():
        V += force * (x >= pos)
    
    for load in loads:
        pos, value, load_type = load[:3]
        
        if load_type == "Puntlast":
            V -= value * (x >= pos)
            
        elif load_type == "Moment":
            M -= value * (x >= pos)
            
        elif load_type in ["Verdeelde last", "Driehoekslast"]:
            length = load[3]
            end_pos = pos + length
            load_region = (x >= pos) & (x <= end_pos)
            beyond_region = x > end_pos
            
            if load_type == "Verdeelde last":
                # Vectorized calculation
                V[load_region] -= value * (x[load_region] - pos)
                V[beyond_region] -= value * length
            else:  # Driehoekslast
                # Vectorized triangle load
                local_x = np.where(load_region, x - pos, 0)
                local_q = value * (local_x / length)
                V[load_region] -= 0.5 * local_q[load_region] * local_x[load_region]
                V[beyond_region] -= 0.5 * value * length
    
    # Bereken moment met cumsum
    M += np.cumsum(V) * dx
    
    return V, M

def analyze_beam(beam_length, supports, loads, profile_type, height, width, wall_thickness, flange_thickness, E):
    """Analyseer de balk en bereken dwarskrachten, momenten, rotatie en doorbuiging"""
    # Optimaal aantal punten voor goede nauwkeurigheid en snelheid
    num_points = 201
    x = np.linspace(0, beam_length, num_points)
    dx = x[1] - x[0]
    
    # Bereken reactiekrachten
    reactions = calculate_reactions(beam_length, supports, loads)
    
    # Bereken interne krachten
    V, M = calculate_internal_forces(x, beam_length, supports, loads, reactions)
    
    # Bereken doorbuiging en rotatie
    I = calculate_I(profile_type, height, width, wall_thickness, flange_thickness)
    EI = E * I
    
    if EI > 0:
        # Vectorized berekeningen voor rotatie en doorbuiging
        theta = np.cumsum(M / EI) * dx
        w = np.cumsum(theta) * dx
        
        # Pas randvoorwaarden toe
        support_positions = np.array([s[0] for s in supports])
        support_indices = np.array([np.abs(x - pos).argmin() for pos in support_positions])
        
        if len(support_indices) >= 2:
            # Matrix oplossing voor randvoorwaarden
            idx1, idx2 = support_indices[:2]
            A = np.array([[1, 1], [idx2 - idx1, 1]])
            b = np.array([-w[idx1], -w[idx2]])
            
            try:
                c1, c2 = np.linalg.solve(A, b)
                # Vectorized correctie
                w += c1 * np.arange(len(x)) + c2
                theta += c1
            except np.linalg.LinAlgError:
                w.fill(0)
                theta.fill(0)
    else:
        w = np.zeros_like(x)
        theta = np.zeros_like(x)
    
    return x, V, M, theta, w

@st.cache_data
def calculate_profile_properties(profile_type, height, width, wall_thickness, flange_thickness):
    """Cache profiel eigenschappen voor snellere berekeningen"""
    I = calculate_I(profile_type, height, width, wall_thickness, flange_thickness)
    A = calculate_A(profile_type, height, width, wall_thickness, flange_thickness)
    W = I / (height/2) if height > 0 else 0
    return I, A, W

def generate_report_html(beam_data, results_plot):
    """Genereer een HTML rapport"""
    
    # Converteer plots naar base64 images
    img_bytes = results_plot.to_image(format="png")
    img_base64 = base64.b64encode(img_bytes).decode()
    plot_image = f"data:image/png;base64,{img_base64}"
    
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
                <tr><td>Hoogte</td><td>{beam_data['dimensions']['height']} mm</td></tr>
                <tr><td>Breedte</td><td>{beam_data['dimensions']['width']} mm</td></tr>
                <tr><td>Wanddikte</td><td>{beam_data['dimensions']['wall_thickness']} mm</td></tr>
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
                <tr><td>Maximaal moment</td><td>{beam_data['results']['max_M']:.2f} Nmm</td></tr>
                <tr><td>Maximale doorbuiging</td><td>{beam_data['results']['max_deflection']:.2f} mm</td></tr>
                <tr><td>Maximale rotatie</td><td>{beam_data['results']['max_rotation']:.6f} rad</td></tr>
            </table>
        </div>

        <div class="section">
            <h3>5. Grafieken</h3>
            <h4>5.1 Analyse Resultaten</h4>
            <img src="{plot_image}" alt="Analyse">
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
        num_supports = st.number_input("Aantal", min_value=2, max_value=4, value=2)
        
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
                    ["Vast", "Scharnier"],
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
            I, A, W = calculate_profile_properties(profile_type, height, width, wall_thickness, flange_thickness)
            
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
            if st.button("Download Rapport", type="secondary", use_container_width=True):
                # Genereer rapport
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
                    "results": {
                        "max_V": max(abs(np.min(V)), abs(np.max(V))),
                        "max_M": max_moment,
                        "max_deflection": max(abs(np.min(deflection)), abs(np.max(deflection))),
                        "max_rotation": max(abs(np.min(rotation)), abs(np.max(rotation))),
                        "max_stress": sigma,
                        "unity_check": UC
                    }
                }
                
                # Genereer rapport
                html_content = generate_report_html(beam_data, results_plot)
                output_dir = "reports"
                os.makedirs(output_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(output_dir, f"beamsolve_report_{timestamp}.html")
                save_report(html_content, output_path)
                st.success(f"Rapport opgeslagen als: {output_path}")

if __name__ == "__main__":
    main()
