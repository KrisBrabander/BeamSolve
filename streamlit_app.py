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
    
    # Bereken schalingsfactor voor doorbuiging
    if x is not None and deflection is not None and np.any(deflection != 0):
        max_defl = max(abs(np.max(deflection)), abs(np.min(deflection)))
        scale_factor = beam_length / (10 * max_defl)  # Maak doorbuiging duidelijker zichtbaar
    else:
        scale_factor = 1
        deflection = np.zeros_like(x) if x is not None else None

    # Teken de balk
    if x is not None and deflection is not None:
        # Teken vervormde balk (dikke lijn)
        scaled_deflection = deflection * scale_factor
        fig.add_trace(go.Scatter(
            x=x,
            y=scaled_deflection,
            mode='lines',
            name='Vervormde balk',
            line=dict(color='#2c3e50', width=6),
            hovertemplate='Positie: %{x:.0f} mm<br>Doorbuiging: %{text:.2f} mm',
            text=deflection
        ))
        
        # Teken onvervormde balk (gestippelde lijn)
        fig.add_trace(go.Scatter(
            x=[0, beam_length],
            y=[0, 0],
            mode='lines',
            name='Onvervormde balk',
            line=dict(color='#95a5a6', width=2, dash='dash')
        ))
    else:
        # Teken alleen onvervormde balk
        fig.add_trace(go.Scatter(
            x=[0, beam_length],
            y=[0, 0],
            mode='lines',
            name='Balk',
            line=dict(color='#2c3e50', width=6)
        ))
    
    # Teken steunpunten
    for pos, type in supports:
        y_pos = deflection[np.abs(x - pos).argmin()] * scale_factor if x is not None and deflection is not None else 0
        
        if type == "Vast":
            # Driehoek voor vaste oplegging
            triangle_x = [pos-40, pos, pos+40]
            triangle_y = [y_pos-50, y_pos, y_pos-50]
            fig.add_trace(go.Scatter(
                x=triangle_x,
                y=triangle_y,
                fill="toself",
                mode='lines',
                name='Vaste oplegging',
                line=dict(color='#e74c3c', width=2),
                fillcolor='rgba(231, 76, 60, 0.3)',
                hoverinfo='name+text',
                text=f'<br>Positie: {pos:.0f} mm'
            ))
            
            # Arcering voor inklemming
            for i in range(-35, 36, 10):
                fig.add_trace(go.Scatter(
                    x=[pos-40, pos+40],
                    y=[y_pos-50+i, y_pos-50+i],
                    mode='lines',
                    line=dict(color='#e74c3c', width=1),
                    showlegend=False
                ))
        else:
            # Driehoek voor scharnier
            triangle_x = [pos-40, pos, pos+40]
            triangle_y = [y_pos-50, y_pos, y_pos-50]
            fig.add_trace(go.Scatter(
                x=triangle_x,
                y=triangle_y,
                fill="toself",
                mode='lines',
                name='Scharnieroplegging',
                line=dict(color='#3498db', width=2),
                fillcolor='rgba(52, 152, 219, 0.3)',
                hoverinfo='name+text',
                text=f'<br>Positie: {pos:.0f} mm'
            ))
            
            # Cirkels voor rol
            for dx in [-20, 0, 20]:
                theta = np.linspace(0, 2*np.pi, 50)
                circle_x = pos + dx + 8 * np.cos(theta)
                circle_y = y_pos-58 + 8 * np.sin(theta)
                fig.add_trace(go.Scatter(
                    x=circle_x,
                    y=circle_y,
                    mode='lines',
                    line=dict(color='#3498db', width=1),
                    fill='toself',
                    fillcolor='rgba(52, 152, 219, 0.3)',
                    showlegend=False
                ))
    
    # Teken belastingen
    for load in loads:
        pos, value, type = load[:3]
        y_pos = deflection[np.abs(x - pos).argmin()] * scale_factor if x is not None and deflection is not None else 0
        
        if type == "Puntlast":
            # Pijl voor puntlast
            arrow_height = 80 if value >= 0 else -80
            fig.add_trace(go.Scatter(
                x=[pos, pos],
                y=[y_pos + arrow_height, y_pos],
                mode='lines',
                name=f'Puntlast {abs(value):.0f} N',
                line=dict(color='#2ecc71', width=3),
                hovertemplate='Puntlast<br>Waarde: %{text:.0f} N<br>Positie: %{x:.0f} mm',
                text=[abs(value), abs(value)]
            ))
            
            # Pijlpunt
            head_size = 20
            if value >= 0:
                fig.add_trace(go.Scatter(
                    x=[pos-head_size/2, pos, pos+head_size/2],
                    y=[y_pos + arrow_height - head_size, y_pos + arrow_height, y_pos + arrow_height - head_size],
                    mode='lines',
                    line=dict(color='#2ecc71', width=3),
                    showlegend=False
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=[pos-head_size/2, pos, pos+head_size/2],
                    y=[y_pos + arrow_height + head_size, y_pos + arrow_height, y_pos + arrow_height + head_size],
                    mode='lines',
                    line=dict(color='#2ecc71', width=3),
                    showlegend=False
                ))
        
        elif type == "Verdeelde last":
            # Verdeelde last met meerdere pijlen
            length = load[3]
            num_arrows = int(length / 200) + 2  # Aantal pijlen afhankelijk van lengte
            positions = np.linspace(pos, pos + length, num_arrows)
            arrow_height = 60 if value >= 0 else -60
            
            # Lijn boven pijlen
            y_loads = [deflection[np.abs(x - p).argmin()] * scale_factor if x is not None and deflection is not None else 0 for p in positions]
            fig.add_trace(go.Scatter(
                x=[pos, pos + length],
                y=[y_pos + arrow_height, y_pos + arrow_height],
                mode='lines',
                name=f'q = {abs(value):.1f} N/mm',
                line=dict(color='#f1c40f', width=3),
                hovertemplate='Verdeelde last<br>q = %{text:.1f} N/mm<br>Lengte: ' + f'{length:.0f} mm',
                text=[abs(value), abs(value)]
            ))
            
            # Pijlen
            for p in positions:
                y_load = deflection[np.abs(x - p).argmin()] * scale_factor if x is not None and deflection is not None else 0
                fig.add_trace(go.Scatter(
                    x=[p, p],
                    y=[y_load + arrow_height, y_load],
                    mode='lines',
                    line=dict(color='#f1c40f', width=2),
                    showlegend=False
                ))
                
                # Pijlpunt
                head_size = 15
                if value >= 0:
                    fig.add_trace(go.Scatter(
                        x=[p-head_size/2, p, p+head_size/2],
                        y=[y_load + head_size, y_load, y_load + head_size],
                        mode='lines',
                        line=dict(color='#f1c40f', width=2),
                        showlegend=False
                    ))
                else:
                    fig.add_trace(go.Scatter(
                        x=[p-head_size/2, p, p+head_size/2],
                        y=[y_load - head_size, y_load, y_load - head_size],
                        mode='lines',
                        line=dict(color='#f1c40f', width=2),
                        showlegend=False
                    ))
        
        elif type == "Moment":
            # Gebogen pijl voor moment
            radius = 40
            theta = np.linspace(-np.pi, np.pi, 50)
            x_circle = pos + radius * np.cos(theta)
            y_circle = y_pos + radius * np.sin(theta)
            
            fig.add_trace(go.Scatter(
                x=x_circle,
                y=y_circle,
                mode='lines',
                name=f'Moment {abs(value):.0f} Nmm',
                line=dict(color='#9b59b6', width=3),
                hovertemplate='Moment<br>Waarde: %{text:.0f} Nmm<br>Positie: ' + f'{pos:.0f} mm',
                text=[abs(value)]
            ))
            
            # Pijlpunt
            arrow_angle = 0 if value >= 0 else np.pi
            arrow_x = [
                pos + radius * np.cos(arrow_angle),
                pos + (radius + 15) * np.cos(arrow_angle),
                pos + radius * np.cos(arrow_angle + np.pi/6)
            ]
            arrow_y = [
                y_pos + radius * np.sin(arrow_angle),
                y_pos + (radius + 15) * np.sin(arrow_angle),
                y_pos + radius * np.sin(arrow_angle + np.pi/6)
            ]
            fig.add_trace(go.Scatter(
                x=arrow_x,
                y=arrow_y,
                mode='lines',
                line=dict(color='#9b59b6', width=3),
                showlegend=False
            ))
    
    # Update layout
    margin = 150  # Grotere marge voor belastingen en steunpunten
    y_range = [-150, 150]
    if x is not None and deflection is not None:
        y_min = min(-150, np.min(scaled_deflection) * 1.2)
        y_max = max(150, np.max(scaled_deflection) * 1.2)
        y_range = [y_min, y_max]
    
    fig.update_layout(
        showlegend=True,
        legend=dict(
            x=0,
            y=1,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.3)',
            borderwidth=1
        ),
        xaxis=dict(
            range=[-margin, beam_length + margin],
            title="Positie (mm)",
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0, 0, 0, 0.1)',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='rgba(0, 0, 0, 0.5)'
        ),
        yaxis=dict(
            range=y_range,
            title="Doorbuiging (mm)",
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0, 0, 0, 0.1)',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='rgba(0, 0, 0, 0.5)',
            scaleanchor="x",
            scaleratio=1
        ),
        title=dict(
            text="Balkschema met Vervormingen",
            x=0.5,
            y=0.95
        ),
        plot_bgcolor='white',
        height=600,  # Grotere hoogte voor betere visualisatie
        margin=dict(t=100, b=100)
    )
    
    return fig

def analyze_beam(beam_length, supports, loads, profile_type, height, width, wall_thickness, flange_thickness, E):
    """Analyseer de balk en bereken momenten, dwarskrachten, rotaties en doorbuigingen"""
    # Discretisatie
    n_points = 200
    x = np.linspace(0, beam_length, n_points)
    dx = x[1] - x[0]
    
    # Initialiseer arrays
    M = np.zeros_like(x)  # Moment
    V = np.zeros_like(x)  # Dwarskracht
    w = np.zeros_like(x)  # Doorbuiging
    theta = np.zeros_like(x)  # Rotatie
    
    # Bereken traagheidsmoment
    I = calculate_I(profile_type, height, width, wall_thickness, flange_thickness)
    
    # Sorteer steunpunten op positie
    supports = sorted(supports, key=lambda s: s[0])
    n = len(supports)
    
    # Matrix voor oplegreacties
    # Voor n steunpunten hebben we n vergelijkingen nodig:
    # 1 verticaal evenwicht
    # n-1 momentevenwichten
    A = np.zeros((n, n))
    b = np.zeros(n)
    
    # Verticaal evenwicht (eerste vergelijking)
    A[0,:] = 1.0  # Som van alle reactiekrachten
    
    # Som van alle externe krachten (negatief, want reactiekrachten moeten tegengesteld zijn)
    for load in loads:
        pos, value, type = load[:3]
        if type == "Puntlast":
            b[0] -= value
        elif type == "Verdeelde last":
            length = load[3]
            b[0] -= value * length
    
    # Momentevenwichten (overige n-1 vergelijkingen)
    # Neem momenten om eerste steunpunt
    ref_pos = supports[0][0]  # Referentiepunt voor momenten
    
    for i in range(1, n):
        # Momentarmen voor reactiekrachten
        for j in range(n):
            A[i,j] = supports[j][0] - ref_pos
        
        # Momenten van externe krachten
        for load in loads:
            pos, value, type = load[:3]
            if type == "Puntlast":
                b[i] -= value * (pos - ref_pos)
            elif type == "Verdeelde last":
                length = load[3]
                q = value
                center = pos + length/2
                total_force = q * length
                b[i] -= total_force * (center - ref_pos)
            elif type == "Moment":
                b[i] -= value
    
    # Los reactiekrachten op
    try:
        R = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Als matrix singulier is, gebruik pseudo-inverse
        R = np.linalg.lstsq(A, b, rcond=None)[0]
    
    # Bereken interne krachten
    for i, xi in enumerate(x):
        # Bijdrage van reactiekrachten
        for j, (pos, _) in enumerate(supports):
            if xi >= pos:
                V[i] += R[j]
                M[i] += R[j] * (xi - pos)
        
        # Bijdrage van belastingen
        for load in loads:
            pos, value, type = load[:3]
            if type == "Puntlast":
                if xi >= pos:
                    V[i] -= value
                    M[i] -= value * (xi - pos)
            elif type == "Verdeelde last":
                length = load[3]
                q = value
                if xi >= pos:
                    if xi <= pos + length:
                        # Binnen belaste gebied
                        load_length = xi - pos
                        V[i] -= q * load_length
                        M[i] -= q * load_length * (xi - (pos + load_length/2))
                    else:
                        # Voorbij belaste gebied
                        V[i] -= q * length
                        M[i] -= q * length * (xi - (pos + length/2))
            elif type == "Moment":
                if xi >= pos:
                    M[i] -= value
    
    # Bereken rotatie en doorbuiging door integratie
    # θ = ∫(M/EI)dx
    # w = ∫θdx
    for i in range(1, len(x)):
        theta[i] = theta[i-1] + (M[i-1] / (E * I)) * dx
        w[i] = w[i-1] + theta[i-1] * dx
    
    # Pas randvoorwaarden toe
    # Voor vaste oplegging: w = 0, θ = 0
    # Voor scharnier: w = 0
    for pos, type in supports:
        idx = np.argmin(abs(x - pos))
        if type == "Vast":
            # Bij vaste oplegging: geen rotatie en geen verplaatsing
            theta_correction = theta[idx]
            w_correction = w[idx]
            theta -= theta_correction
            w -= w_correction + theta_correction * (x - x[idx])
        else:  # Scharnier/rol
            # Bij scharnier: wel rotatie, geen verplaatsing
            w_correction = w[idx]
            w -= w_correction
    
    return x, M, theta, w

def generate_report_html(beam_data, results, plots):
    """Genereer een HTML rapport"""
    
    # Converteer plots naar base64 images
    plot_images = []
    for plot in plots:
        img_bytes = plot.to_image(format="png")
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
        
        # Invoer sectie
        st.header("Invoer")
        
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
    
    # Hoofdgedeelte - Visualisaties
    if st.button("Bereken", type="primary"):
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
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Balkschema en vervormingen
            st.subheader("Balkschema en Vervormingen")
            beam_plot = plot_beam_diagram(beam_length, supports, loads, x, deflection)
            st.plotly_chart(beam_plot, use_container_width=True, height=600)
            
            # Analyse resultaten
            st.subheader("Analyse Resultaten")
            analysis_plot = plot_results(x, M, rotation, deflection)
            st.plotly_chart(analysis_plot, use_container_width=True, height=600)
        
        with col2:
            # Maximale waarden
            st.subheader("Maximale Waarden")
            max_moment = max(abs(np.min(M)), abs(np.max(M)))
            max_deflection = max(abs(np.min(deflection)), abs(np.max(deflection)))
            max_rotation = max(abs(np.min(rotation)), abs(np.max(rotation)))
            
            st.metric("Maximaal moment", f"{max_moment:.2f} Nmm")
            st.metric("Maximale doorbuiging", f"{max_deflection:.2f} mm")
            st.metric("Maximale rotatie", f"{max_rotation:.6f} rad")
            
            # Rapport genereren
            st.markdown("---")
            if st.button("Genereer Rapport", type="secondary"):
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
