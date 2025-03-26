import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os
import base64
from plotly.subplots import make_subplots

def calculate_reactions(beam_length, supports, loads):
    """Bereken reactiekrachten voor verschillende steunpuntconfiguraties.
    Tekenconventies:
    - Krachten omhoog positief
    - Momenten rechtsom positief
    - Belastingen omlaag positief"""
    
    if not supports or not loads:
        return {}
        
    # Sorteer steunpunten
    supports = sorted(supports, key=lambda x: x[0])
    n = len(supports)
    
    if n == 0:
        return {}
    
    reactions = {}
    
    try:
        if n == 1:
            # Enkel steunpunt (moet inklemming zijn)
            pos, type = supports[0]
            if type.lower() != "inklemming":
                st.error("❌ Systeem met één steunpunt moet een inklemming zijn")
                return None
                
            # Bereken totale verticale kracht en moment t.o.v. inklemming
            V_total = 0
            M_total = 0
            
            for load in loads:
                pos_load, value, load_type, *rest = load
                if load_type.lower() == "puntlast":
                    V_total += value  # Last omlaag positief
                    M_total += value * (pos_load - pos)  # Rechtsom positief
                elif load_type.lower() == "verdeelde last":
                    length = rest[0]
                    q = value  # Last per lengte-eenheid
                    V_total += q * length  # Totale last
                    x_c = pos_load + length/2  # Zwaartepunt
                    M_total += q * length * (x_c - pos)  # Moment t.o.v. inklemming
                elif load_type.lower() == "moment":
                    M_total += value  # Extern moment (rechtsom positief)
            
            reactions[pos] = -V_total  # Reactiekracht tegengesteld aan belasting
            reactions[f"M_{pos}"] = -M_total  # Reactiemoment tegengesteld aan belastingmoment
            
        elif n == 2:
            # Twee steunpunten - statisch bepaald systeem
            x1, type1 = supports[0]
            x2, type2 = supports[1]
            L = x2 - x1
            
            if L == 0:
                st.error("❌ Steunpunten mogen niet op dezelfde positie liggen")
                return None
            
            # Bereken totale verticale kracht en moment t.o.v. linker steunpunt
            V_total = 0  # Totale verticale belasting
            M_1 = 0     # Moment t.o.v. steunpunt 1
            
            for load in loads:
                pos, value, load_type, *rest = load
                if load_type.lower() == "puntlast":
                    V_total += value  # Last omlaag positief
                    M_1 += value * (pos - x1)  # Moment t.o.v. steunpunt 1
                elif load_type.lower() == "verdeelde last":
                    length = rest[0]
                    q = value
                    Q = q * length  # Totale last
                    x_c = pos + length/2  # Zwaartepunt
                    V_total += Q
                    M_1 += Q * (x_c - x1)  # Moment t.o.v. steunpunt 1
                elif load_type.lower() == "moment":
                    M_1 += value  # Extern moment (rechtsom positief)
            
            # Los reactiekrachten op met momentevenwicht
            R2 = M_1 / L  # Reactie in steunpunt 2 (omhoog positief)
            R1 = V_total - R2  # Reactie in steunpunt 1 (omhoog positief)
            
            reactions[x1] = -R1  # Reactiekracht tegengesteld aan belasting
            reactions[x2] = -R2
            
            # Voeg inklemming momenten toe indien nodig
            if type1.lower() == "inklemming" and type2.lower() == "inklemming":
                # Beide ingeklemd: symmetrische verdeling
                reactions[f"M_{x1}"] = -M_1/2
                reactions[f"M_{x2}"] = M_1/2
            elif type1.lower() == "inklemming":
                # Alleen links ingeklemd
                reactions[f"M_{x1}"] = -M_1
            elif type2.lower() == "inklemming":
                # Alleen rechts ingeklemd
                reactions[f"M_{x2}"] = M_1
                
        else:
            # Drie of meer steunpunten
            # Bereken totale belasting en zwaartepunt
            V_total = 0
            M_total = 0
            x_ref = supports[0][0]  # Referentiepunt voor momenten
            
            for load in loads:
                pos, value, load_type, *rest = load
                if load_type.lower() == "puntlast":
                    V_total += value
                    M_total += value * (pos - x_ref)
                elif load_type.lower() == "verdeelde last":
                    length = rest[0]
                    q = value
                    Q = q * length
                    x_c = pos + length/2
                    V_total += Q
                    M_total += Q * (x_c - x_ref)
                elif load_type.lower() == "moment":
                    M_total += value
            
            # Voor 3 steunpunten: middelste draagt meer
            if n == 3:
                x1, _ = supports[0]
                x2, _ = supports[1]
                x3, _ = supports[2]
                L1 = x2 - x1
                L2 = x3 - x2
                L = x3 - x1
                
                # Los op met momentevenwicht en verticaal evenwicht
                # Neem aan dat middelste steunpunt 40-60% draagt
                R2 = V_total * 0.5  # Middelste steunpunt
                # Verdeel rest over buitenste steunpunten o.b.v. momentevenwicht
                R3 = (M_total - R2*(x2 - x_ref)) / (x3 - x_ref)
                R1 = V_total - R2 - R3
                
                reactions[x1] = -R1
                reactions[x2] = -R2
                reactions[x3] = -R3
            else:
                # Voor 4+ steunpunten: gelijke verdeling
                R = V_total / n
                for pos, _ in supports:
                    reactions[pos] = -R
                
    except Exception as e:
        st.error(f"❌ Fout bij berekenen reactiekrachten: {str(e)}")
        return None
            
    return reactions

def calculate_internal_forces(x, beam_length, supports, loads, reactions):
    """Bereken dwarskracht en moment in de balk.
    Tekenconventies:
    - Dwarskracht positief omhoog
    - Moment positief rechtsom
    - Belastingen positief omlaag"""
    
    # Initialiseer arrays
    V = np.zeros_like(x)
    M = np.zeros_like(x)
    
    # Sorteer steunpunten
    supports = sorted(supports, key=lambda x: x[0])
    
    try:
        # 1. Verwerk reactiekrachten (positief omhoog)
        for pos, value in reactions.items():
            if not isinstance(pos, str):  # Skip moment reacties (M_x)
                idx = np.abs(x - pos).argmin()
                V[idx:] += value  # Dwarskrachtsprong
                M[idx:] += value * (x[idx:] - pos)  # Momentbijdrage
        
        # 2. Verwerk uitwendige belasting
        for load in loads:
            pos, value, load_type, *rest = load
            
            if load_type.lower() == "puntlast":
                # Puntlast (positief omlaag)
                idx = np.abs(x - pos).argmin()
                V[idx:] -= value  # Dwarskrachtsprong
                M[idx:] -= value * (x[idx:] - pos)  # Momentbijdrage
                
            elif load_type.lower() == "verdeelde last":
                # Verdeelde last (positief omlaag)
                length = rest[0]
                q = value  # Last per lengte-eenheid
                
                # Begin- en eindpunt van de last
                start_idx = np.abs(x - pos).argmin()
                end_idx = np.abs(x - (pos + length)).argmin()
                
                # Dwarskracht: lineair afnemend
                V[start_idx:end_idx] -= q * (x[start_idx:end_idx] - pos)
                V[end_idx:] -= q * length
                
                # Moment: kwadratisch
                for i in range(start_idx, len(x)):
                    if i < end_idx:
                        # Onder de last
                        xi = x[i] - pos
                        M[i] -= q * xi**2 / 2
                    else:
                        # Voorbij de last
                        M[i] -= q * length * (x[i] - (pos + length/2))
                        
            elif load_type.lower() == "moment":
                # Extern moment (positief rechtsom)
                idx = np.abs(x - pos).argmin()
                M[idx:] += value
                
    except Exception as e:
        st.error(f"❌ Fout bij berekenen inwendige krachten: {str(e)}")
        return None, None
        
    return V, M

def calculate_deflection(x, beam_length, supports, loads, reactions, EI):
    """Bereken doorbuiging met de momentenvlakken methode."""
    
    try:
        # 1. Bereken momentenlijn
        _, M = calculate_internal_forces(x, beam_length, supports, loads, reactions)
        if M is None:
            return None, None
            
        # 2. Dubbele integratie
        dx = x[1] - x[0]
        kappa = M / EI  # Kromming κ = M/EI
        
        # Eerste integratie voor helling
        theta = np.zeros_like(x)
        theta[1:] = np.cumsum(kappa[:-1]) * dx
        
        # Tweede integratie voor doorbuiging
        y = np.zeros_like(x)
        y[1:] = np.cumsum(theta[:-1]) * dx
        
        # 3. Pas randvoorwaarden toe voor 3 steunpunten
        if len(supports) == 3:
            # Vind steunpunt indices
            support_indices = []
            for pos, _ in supports:
                idx = np.abs(x - pos).argmin()
                support_indices.append(idx)
                
            i1, i2, i3 = support_indices
            
            # Bereken correctie parameters
            L1 = x[i2] - x[i1]  # Lengte eerste overspanning
            L2 = x[i3] - x[i2]  # Lengte tweede overspanning
            
            # Correctie voor eerste overspanning (0 tot L1)
            mask1 = (x >= x[i1]) & (x <= x[i2])
            xi1 = (x[mask1] - x[i1]) / L1
            y[mask1] = y[i1] + (y[i2] - y[i1]) * (3*xi1**2 - 2*xi1**3)
            
            # Correctie voor tweede overspanning (L1 tot L1+L2)
            mask2 = (x > x[i2]) & (x <= x[i3])
            xi2 = (x[mask2] - x[i2]) / L2
            y[mask2] = y[i2] + (y[i3] - y[i2]) * (3*xi2**2 - 2*xi2**3)
            
            # Update helling na correcties
            theta[1:-1] = (y[2:] - y[:-2]) / (2*dx)
            theta[0] = (y[1] - y[0]) / dx
            theta[-1] = (y[-1] - y[-2]) / dx
        
        return y, theta
        
    except Exception as e:
        st.error(f"❌ Fout bij berekenen doorbuiging: {str(e)}")
        return None, None

def analyze_beam(beam_length, supports, loads, profile_type, height, width, wall_thickness, flange_thickness, E):
    """Analyseer de balk met de elementaire methode"""
    try:
        # Controleer invoer
        if not all([beam_length, supports, loads, profile_type, height, width, wall_thickness, E]):
            st.error("❌ Vul alle velden in")
            return None, None, None, None, None
            
        if E <= 0:
            st.error("❌ E-modulus moet positief zijn")
            return None, None, None, None, None
            
        # Bereken profieleigenschappen
        I = calculate_I(profile_type, height, width, wall_thickness, flange_thickness)
        if I is None or I <= 0:
            st.error("❌ Ongeldige profielafmetingen")
            return None, None, None, None, None
            
        # Bereken buigstijfheid
        EI = E * I  # N·mm²
        
        # Maak x-array voor berekeningen (100 punten)
        x = np.linspace(0, beam_length, 100)
        
        # Bereken reactiekrachten
        reactions = calculate_reactions(beam_length, supports, loads)
        if reactions is None:
            return None, None, None, None, None
            
        # Bereken inwendige krachten
        V, M = calculate_internal_forces(x, beam_length, supports, loads, reactions)
        if V is None or M is None:
            return None, None, None, None, None
            
        # Bereken doorbuiging en rotatie
        y, theta = calculate_deflection(x, beam_length, supports, loads, reactions, EI)
        if y is None or theta is None:
            return None, None, None, None, None
            
        return x, V, M, theta, y
        
    except Exception as e:
        st.error(f"❌ Fout bij analyse: {str(e)}")
        return None, None, None, None, None

def analyze_beam_matrix(beam_length, supports, loads, EI, x):
    """Analyseer de balk met de stijfheidsmethode"""
    # 1. Bereken eerst de reactiekrachten
    reactions = calculate_reactions(beam_length, supports, loads)
    if not reactions:
        return None, None, None, None, None
    
    # 2. Bereken interne krachten
    V, M = calculate_internal_forces(x, beam_length, supports, loads, reactions)
    
    # 3. Bereken doorbuiging en rotatie
    deflection, rotation = calculate_deflection(x, beam_length, supports, loads, reactions, EI)
    
    return x, V, M, rotation, deflection

def plot_beam_diagram(beam_length, supports, loads):
    """Teken professioneel balkschema"""
    fig = go.Figure()
    
    # Moderne kleuren
    colors = {
        'beam': '#2c3e50',  # Donkerblauw-grijs
        'support': '#3498db',  # Helder blauw
        'load': '#e74c3c',  # Rood
        'background': '#ffffff',  # Wit
        'grid': '#ecf0f1'  # Lichtgrijs
    }
    
    # Teken balk - modern en strak
    fig.add_trace(go.Scatter(
        x=[0, beam_length/1000],
        y=[0, 0],
        mode='lines',
        line=dict(color=colors['beam'], width=6),
        name='Balk'
    ))
    
    # Teken steunpunten
    for pos, type in supports:
        x_pos = pos/1000  # Convert to meters
        triangle_size = beam_length/50
        type = type.lower()
        
        if type == "inklemming":
            # Moderne inklemming met gevulde rechthoek en arcering
            fig.add_trace(go.Scatter(
                x=[x_pos, x_pos, x_pos+triangle_size/1000, x_pos+triangle_size/1000, x_pos],
                y=[-triangle_size/1000, triangle_size/1000, triangle_size/1000, -triangle_size/1000, -triangle_size/1000],
                fill="toself",
                mode='lines',
                line=dict(color=colors['support'], width=2),
                fillcolor=colors['support'],
                opacity=0.3,
                name='Inklemming',
                showlegend=True if type == "inklemming" else False
            ))
            # Moderne arcering met dunnere lijnen
            for i in range(5):
                offset = -triangle_size/1000 + i * triangle_size/500
                fig.add_trace(go.Scatter(
                    x=[x_pos, x_pos+triangle_size/1000],
                    y=[offset, offset],
                    mode='lines',
                    line=dict(color=colors['support'], width=1),
                    showlegend=False
                ))
                
        elif type == "scharnier":
            # Modern driehoekig support met vulling
            fig.add_trace(go.Scatter(
                x=[x_pos-triangle_size/1000, x_pos+triangle_size/1000, x_pos, x_pos-triangle_size/1000],
                y=[-triangle_size/1000, -triangle_size/1000, 0, -triangle_size/1000],
                fill="toself",
                mode='lines',
                line=dict(color=colors['support'], width=2),
                fillcolor=colors['support'],
                opacity=0.3,
                name='Scharnier',
                showlegend=True if type == "scharnier" else False
            ))
            
        elif type == "rol":
            # Moderne rol met cirkels
            fig.add_trace(go.Scatter(
                x=[x_pos-triangle_size/1000, x_pos+triangle_size/1000, x_pos, x_pos-triangle_size/1000],
                y=[-triangle_size/1000, -triangle_size/1000, 0, -triangle_size/1000],
                fill="toself",
                mode='lines',
                line=dict(color=colors['support'], width=2),
                fillcolor=colors['support'],
                opacity=0.3,
                name='Rol',
                showlegend=True if type == "rol" else False
            ))
            # Voeg cirkels toe voor rol effect
            circle_size = triangle_size/2000
            for i in [-1, 0, 1]:
                fig.add_trace(go.Scatter(
                    x=[x_pos + i*circle_size*2],
                    y=[-triangle_size/1000 - circle_size],
                    mode='markers',
                    marker=dict(size=6, color=colors['support']),
                    showlegend=False
                ))
    
    # Teken belastingen
    for load in loads:
        x_pos = load[0]/1000
        value = load[1]
        load_type = load[2]
        
        if load_type.lower() == "puntlast":
            # Maak puntlast pijlen 1.5x langer dan verdeelde last pijlen
            arrow_height = beam_length/25  # Was /40, nu langer
            # Label boven de pijl
            fig.add_trace(go.Scatter(
                x=[x_pos],
                y=[arrow_height/1000 + arrow_height/4000],
                mode='text',
                text=[f'{value/1000:.1f} kN'],
                textposition='top center',
                textfont=dict(size=14, color=colors['load']),
                showlegend=False
            ))
            # Pijl
            fig.add_trace(go.Scatter(
                x=[x_pos, x_pos],
                y=[arrow_height/1000, 0],
                mode='lines',
                line=dict(color=colors['load'], width=3),
                showlegend=True,
                name='Puntlast'
            ))
            # Pijlpunt (driehoek)
            fig.add_shape(
                type="path",
                path=f"M {x_pos-arrow_height/3000} {arrow_height/3000} L {x_pos} 0 L {x_pos+arrow_height/3000} {arrow_height/3000} Z",
                fillcolor=colors['load'],
                line=dict(color=colors['load'], width=0),
            )
            
        elif load_type.lower() == "verdeelde last":
            # Standaard hoogte voor verdeelde last
            arrow_height = beam_length/40
            length = load[3]/1000 if len(load) > 3 else (beam_length - load[0])/1000
            # Label boven de verdeelde last
            fig.add_trace(go.Scatter(
                x=[x_pos + length/2],
                y=[arrow_height/1000 + arrow_height/4000],
                mode='text',
                text=[f'{value/1000:.1f} kN/m'],
                textposition='top center',
                textfont=dict(size=14, color=colors['load']),
                showlegend=False
            ))
            
            # Verbindingslijn bovenaan
            fig.add_trace(go.Scatter(
                x=[x_pos, x_pos+length],
                y=[arrow_height/1000, arrow_height/1000],
                mode='lines',
                line=dict(color=colors['load'], width=3),
                showlegend=True,
                name='Verdeelde last'
            ))
            
            # Pijlen
            num_arrows = min(max(int(length*8), 4), 15)  # Meer pijlen voor vloeiender uiterlijk
            for i in range(num_arrows):
                arrow_x = x_pos + (i * length/(num_arrows-1))
                # Pijlsteel
                fig.add_trace(go.Scatter(
                    x=[arrow_x, arrow_x],
                    y=[arrow_height/1000, 0],
                    mode='lines',
                    line=dict(color=colors['load'], width=2),
                    showlegend=False
                ))
                # Pijlpunt (driehoek)
                fig.add_shape(
                    type="path",
                    path=f"M {arrow_x-arrow_height/4000} {arrow_height/4000} L {arrow_x} 0 L {arrow_x+arrow_height/4000} {arrow_height/4000} Z",
                    fillcolor=colors['load'],
                    line=dict(color=colors['load'], width=0),
                )
            
        elif load_type.lower() == "driehoekslast":
            length = load[3]/1000
            # Label boven het hoogste punt
            fig.add_trace(go.Scatter(
                x=[x_pos + length],
                y=[beam_length/40/1000 + beam_length/40/4000],
                mode='text',
                text=[f'{value/1000:.1f} kN/m'],
                textposition='top center',
                textfont=dict(size=14, color=colors['load']),
                showlegend=False
            ))
            
            # Schuine lijn bovenaan
            fig.add_trace(go.Scatter(
                x=[x_pos, x_pos+length],
                y=[0, beam_length/40/1000],
                mode='lines',
                line=dict(color=colors['load'], width=3),
                showlegend=True,
                name='Driehoekslast'
            ))
            
            # Pijlen met variabele lengte
            num_arrows = min(max(int(length*8), 4), 15)  # Meer pijlen voor vloeiender uiterlijk
            for i in range(num_arrows):
                rel_pos = i/(num_arrows-1)
                arrow_x = x_pos + length * rel_pos
                current_height = (beam_length/40/1000) * rel_pos  # Hoogte op basis van positie
                
                # Pijlsteel
                fig.add_trace(go.Scatter(
                    x=[arrow_x, arrow_x],
                    y=[current_height, 0],
                    mode='lines',
                    line=dict(color=colors['load'], width=2),
                    showlegend=False
                ))
                # Pijlpunt (driehoek)
                arrow_size = (beam_length/40/4000) * rel_pos  # Pijlgrootte schaalt mee
                if rel_pos > 0:  # Alleen pijlpunten tekenen als er een steel is
                    fig.add_shape(
                        type="path",
                        path=f"M {arrow_x-arrow_size} {arrow_size} L {arrow_x} 0 L {arrow_x+arrow_size} {arrow_size} Z",
                        fillcolor=colors['load'],
                        line=dict(color=colors['load'], width=0),
                    )
            
        elif load_type.lower() == "moment":
            # Label bij het moment
            fig.add_trace(go.Scatter(
                x=[x_pos],
                y=[beam_length/40/1000 + beam_length/40/4000],
                mode='text',
                text=[f'{value/1e6:.1f} kNm'],
                textposition='top center',
                textfont=dict(size=14, color=colors['load']),
                showlegend=False
            ))
            
            # Moment cirkel met pijl
            radius = beam_length/40/2000
            theta = np.linspace(-np.pi/2, 3*np.pi/2, 50)
            fig.add_trace(go.Scatter(
                x=x_pos + radius*np.cos(theta),
                y=radius*np.sin(theta),
                mode='lines',
                line=dict(color=colors['load'], width=3),
                showlegend=True,
                name='Moment'
            ))
            # Pijlpunt op cirkel
            arrow_angle = 3*np.pi/2
            arrow_size = radius/2
            fig.add_shape(
                type="path",
                path=f"M {x_pos + radius*np.cos(arrow_angle-0.2)} {radius*np.sin(arrow_angle-0.2)} L {x_pos + radius*np.cos(arrow_angle)} {radius*np.sin(arrow_angle)} L {x_pos + radius*np.cos(arrow_angle-0.2)} {radius*np.sin(arrow_angle+0.2)}",
                fillcolor=colors['load'],
                line=dict(color=colors['load'], width=3),
            )
    
    # Update layout voor moderne uitstraling
    fig.update_layout(
        title=dict(
            text="Balkschema en Belastingen",
            font=dict(size=24, color=colors['beam'])
        ),
        height=300,
        showlegend=True,
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1,
            range=[-beam_length/20/1000, beam_length/20/1000],
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor=colors['beam'],
            showgrid=True,
            gridcolor=colors['grid'],
            gridwidth=1
        ),
        xaxis=dict(
            range=[-beam_length/20/1000, beam_length*1.1/1000],
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor=colors['beam'],
            showgrid=True,
            gridcolor=colors['grid'],
            gridwidth=1
        ),
        margin=dict(t=50, b=50, l=50, r=50),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor=colors['beam'],
            borderwidth=1
        )
    )
    
    return fig

def plot_results(x, V, M, theta, y, beam_length, supports, loads):
    """Plot resultaten in één figuur met subplots"""
    
    # Maak figuur met subplots
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            "Dwarskracht [kN]",
            "Moment [kNm]",
            "Rotatie [rad]",
            "Doorbuiging [mm]"
        ),
        vertical_spacing=0.08,
        row_heights=[0.25, 0.25, 0.25, 0.25]
    )
    
    # Plot dwarskracht (bovenaan)
    fig.add_trace(
        go.Scatter(
            x=x/1000, y=V/1000,
            mode='lines',
            name='V',
            line=dict(color='green', width=2)
        ),
        row=1, col=1
    )
    
    # Plot moment
    fig.add_trace(
        go.Scatter(
            x=x/1000, y=M/1e6,
            mode='lines',
            name='M',
            line=dict(color='red', width=2)
        ),
        row=2, col=1
    )
    
    # Plot rotatie
    fig.add_trace(
        go.Scatter(
            x=x/1000, y=theta,
            mode='lines',
            name='θ',
            line=dict(color='orange', width=2)
        ),
        row=3, col=1
    )
    
    # Plot doorbuiging
    fig.add_trace(
        go.Scatter(
            x=x/1000, y=y,
            mode='lines',
            name='y',
            line=dict(color='blue', width=2)
        ),
        row=4, col=1
    )
    
    # Voeg steunpunten toe aan doorbuigingsgrafiek
    for pos, type in supports:
        marker = "triangle-up" if type.lower() != "inklemming" else "square"
        fig.add_trace(
            go.Scatter(
                x=[pos/1000],
                y=[0],
                mode='markers',
                name=type,
                marker=dict(
                    symbol=marker,
                    size=12,
                    color='black'
                ),
                showlegend=False
            ),
            row=4, col=1
        )
    
    # Voeg belastingen toe aan doorbuigingsgrafiek
    for load in loads:
        pos, value, load_type, *rest = load
        if load_type.lower() == "puntlast":
            # Puntlast pijl omlaag
            fig.add_trace(
                go.Scatter(
                    x=[pos/1000],
                    y=[2],  # Iets boven de balk
                    mode='markers',
                    name=f'{value/1000:.1f} kN',
                    marker=dict(
                        symbol='arrow-down',
                        size=12,
                        color='red'
                    ),
                    showlegend=False
                ),
                row=4, col=1
            )
        elif load_type.lower() == "verdeelde last":
            # Verdeelde last als lijn met pijlen
            length = rest[0]
            start = pos/1000
            end = (pos + length)/1000
            fig.add_trace(
                go.Scatter(
                    x=[start, end],
                    y=[2, 2],  # Iets boven de balk
                    mode='lines',
                    name=f'{value/1000:.1f} kN/m',
                    line=dict(color='red', width=2),
                    showlegend=False
                ),
                row=4, col=1
            )
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=False,
        margin=dict(t=60, b=20),
        plot_bgcolor='white'
    )
    
    # Update assen
    for i in range(1, 5):
        fig.update_xaxes(
            title="Positie [m]" if i == 4 else None,
            gridcolor='lightgray',
            zerolinecolor='black',
            row=i, col=1
        )
        fig.update_yaxes(
            gridcolor='lightgray',
            zerolinecolor='black',
            row=i, col=1
        )
    
    return fig

def generate_pdf_report(beam_data, results_plot):
    """Genereer een professioneel PDF rapport"""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from io import BytesIO
    import plotly.io as pio
    from datetime import datetime
    
    # Converteer plotly figuur naar afbeelding
    img_bytes = pio.to_image(results_plot, format="png", width=800, height=600, scale=2)
    
    # Maak PDF buffer
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=20*mm,
        leftMargin=20*mm,
        topMargin=20*mm,
        bottomMargin=20*mm
    )
    
    # Definieer stijlen
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        textColor=colors.HexColor('#2c3e50')
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceBefore=20,
        spaceAfter=10,
        textColor=colors.HexColor('#34495e')
    )
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#2c3e50')
    )
    
    # Start document opbouw
    elements = []
    
    # Header met logo en titel
    elements.append(Paragraph("BeamSolve Pro", title_style))
    elements.append(Paragraph(f"Rapport gegenereerd op {datetime.now().strftime('%d-%m-%Y %H:%M')}", body_style))
    elements.append(Spacer(1, 20))
    
    # Profiel informatie
    elements.append(Paragraph("1. Profiel Specificaties", heading_style))
    profile_data = [
        ["Type", beam_data["profile_type"]],
        ["Hoogte", f"{beam_data['height']} mm"],
        ["Breedte", f"{beam_data['width']} mm"],
        ["Wanddikte", f"{beam_data['wall_thickness']} mm"]
    ]
    if "flange_thickness" in beam_data and beam_data["flange_thickness"]:
        profile_data.append(["Flensdikte", f"{beam_data['flange_thickness']} mm"])
    
    profile_table = Table(profile_data, colWidths=[100, 200])
    profile_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f8f9fa')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#2c3e50')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6'))
    ]))
    elements.append(profile_table)
    elements.append(Spacer(1, 20))
    
    # Profiel eigenschappen
    elements.append(Paragraph("2. Profiel Eigenschappen", heading_style))
    properties_data = [
        ["Parameter", "Waarde", "Eenheid"],
        ["Oppervlakte", f"{beam_data['area']:.0f}", "mm²"],
        ["Traagheidsmoment", f"{beam_data['moment_of_inertia']:.0f}", "mm⁴"],
        ["Weerstandsmoment", f"{beam_data['section_modulus']:.0f}", "mm³"],
        ["Max. buigspanning", f"{beam_data['max_stress']:.1f}", "N/mm²"]
    ]
    
    properties_table = Table(properties_data, colWidths=[100, 100, 100])
    properties_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f8f9fa')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#2c3e50')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6'))
    ]))
    elements.append(properties_table)
    elements.append(Spacer(1, 20))
    
    # Steunpunten
    elements.append(Paragraph("3. Steunpunten", heading_style))
    support_data = [["#", "Type", "Positie"]]
    for i, (pos, type) in enumerate(beam_data['supports'], 1):
        support_data.append([str(i), type, f"{pos} mm"])
    
    support_table = Table(support_data, colWidths=[50, 150, 100])
    support_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f8f9fa')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#2c3e50')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6'))
    ]))
    elements.append(support_table)
    elements.append(Spacer(1, 20))
    
    # Belastingen
    elements.append(Paragraph("4. Belastingen", heading_style))
    load_data = [["#", "Type", "Waarde", "Positie", "Lengte"]]
    for i, load in enumerate(beam_data['loads'], 1):
        if len(load) == 4:  # Verdeelde of driehoekslast
            pos, val, type, length = load
            if type == "Verdeelde last":
                load_data.append([str(i), type, f"{val/1000:.1f} kN/m", f"{pos} mm", f"{length} mm"])
            elif type == "Driehoekslast":
                load_data.append([str(i), type, f"{val/1000:.1f} kN/m", f"{pos} mm", f"{length} mm"])
        else:  # Puntlast of moment
            pos, val, type = load
            if type == "Moment":
                load_data.append([str(i), type, f"{val/1e6:.1f} kNm", f"{pos} mm", "-"])
            else:
                load_data.append([str(i), type, f"{val/1000:.1f} kN", f"{pos} mm", "-"])
    
    load_table = Table(load_data, colWidths=[30, 100, 80, 80, 80])
    load_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f8f9fa')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#2c3e50')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6'))
    ]))
    elements.append(load_table)
    elements.append(Spacer(1, 20))
    
    # Resultaten
    elements.append(Paragraph("5. Resultaten", heading_style))
    
    # Grafieken
    img_stream = BytesIO(img_bytes)
    img = Image(img_stream, width=160*mm, height=120*mm)
    elements.append(img)
    elements.append(Spacer(1, 10))
    
    # Maximale waarden
    elements.append(Paragraph("Maximale Waarden:", heading_style))
    max_data = [
        ["Parameter", "Waarde", "Eenheid"],
        ["Max. Doorbuiging", f"{beam_data['max_deflection']:.2f}", "mm"],
        ["Max. Rotatie", f"{beam_data['max_rotation']:.4f}", "rad"],
        ["Max. Dwarskracht", f"{beam_data['max_shear']/1000:.1f}", "kN"],
        ["Max. Moment", f"{beam_data['max_moment']/1000000:.1f}", "kNm"]
    ]
    
    max_table = Table(max_data, colWidths=[100, 100, 100])
    max_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f8f9fa')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#2c3e50')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6'))
    ]))
    elements.append(max_table)
    
    # Footer
    elements.append(Spacer(1, 30))
    footer_text = "Berekend met BeamSolve Pro 2025"
    elements.append(Paragraph(footer_text, body_style))
    
    # Build PDF
    doc.build(elements)
    pdf = buffer.getvalue()
    buffer.close()
    
    return pdf

def save_report(report_content, output_path):
    """Sla het rapport op"""
    with open(output_path, 'wb') as f:
        f.write(report_content)

def setup_stiffness_system(beam_length, supports, loads, EI):
    """Stel stijfheidsmatrix en belastingsvector op voor het complete systeem"""
    # Tel aantal vrijheidsgraden (DOFs)
    n_dofs = 0
    dof_map = {}  # Maps support position to DOF indices
    
    for pos, type in sorted(supports, key=lambda s: s[0]):
        if type.lower() == "inklemming":
            # Inklemming heeft 2 DOFs: verplaatsing en rotatie
            dof_map[pos] = (n_dofs, n_dofs + 1)
            n_dofs += 2
        else:
            # Scharnier of rol heeft 1 DOF: alleen verplaatsing
            dof_map[pos] = (n_dofs,)
            n_dofs += 1
    
    # Initialiseer stijfheidsmatrix en belastingsvector
    K = np.zeros((n_dofs, n_dofs))
    F = np.zeros(n_dofs)
    
    # Vul stijfheidsmatrix
    for i, (pos_i, type_i) in enumerate(sorted(supports, key=lambda s: s[0])):
        for j, (pos_j, type_j) in enumerate(sorted(supports, key=lambda s: s[0])):
            # Bepaal welke elementen we moeten vullen op basis van steunpunttype
            dofs_i = dof_map[pos_i]
            dofs_j = dof_map[pos_j]
            
            # Bereken stijfheidscoëfficiënten
            if pos_i <= pos_j:
                x1, x2 = pos_i, pos_j
            else:
                x1, x2 = pos_j, pos_i
            
            L = beam_length
            
            # Basiscoëfficiënten voor verplaatsing-verplaatsing interactie
            k_vv = EI * (3*L - 3*x1 - 3*(L-x2)) / L**3
            
            # Als een van beide punten een inklemming is
            if type_i.lower() == "inklemming" or type_j.lower() == "inklemming":
                k_vr = EI * (2 - 3*(x2-x1)/L) / L**2  # verplaatsing-rotatie
                k_rr = EI * (2*L - 3*x1 - 3*(L-x2)) / L  # rotatie-rotatie
                
                # Vul de juiste elementen in de matrix
                if type_i.lower() == "inklemming":
                    K[dofs_i[0], dofs_j[0]] = k_vv
                    if len(dofs_j) > 1:  # als j ook een inklemming is
                        K[dofs_i[0], dofs_j[1]] = k_vr
                        K[dofs_i[1], dofs_j[0]] = k_vr
                        K[dofs_i[1], dofs_j[1]] = k_rr
                
                if type_j.lower() == "inklemming":
                    K[dofs_j[0], dofs_i[0]] = k_vv
                    if len(dofs_i) > 1:  # als i ook een inklemming is
                        K[dofs_j[0], dofs_i[1]] = k_vr
                        K[dofs_j[1], dofs_i[0]] = k_vr
                        K[dofs_j[1], dofs_i[1]] = k_rr
            else:
                # Beide punten zijn scharnieren of rollen
                K[dofs_i[0], dofs_j[0]] = k_vv
    
    # Vul belastingsvector
    for load in loads:
        pos, value, load_type, *rest = load
        
        # Voor elk steunpunt, bereken de bijdrage van deze last
        for support_pos, support_type in supports:
            dofs = dof_map[support_pos]
            
            if load_type.lower() == "puntlast":
                # Puntlast: F = P * influence_function(x)
                if support_pos <= pos:
                    x = support_pos
                    a = pos
                    F[dofs[0]] -= value * x*(L-a)*(2*L-x-a)/(6*EI*L)
                    if len(dofs) > 1:  # inklemming
                        F[dofs[1]] -= value * x*(L-a)/(2*EI*L)
                else:
                    x = pos
                    a = support_pos
                    F[dofs[0]] -= value * x*(L-a)*(2*L-x-a)/(6*EI*L)
                    if len(dofs) > 1:  # inklemming
                        F[dofs[1]] -= value * x*(L-a)/(2*EI*L)
            
            elif load_type.lower() == "q":
                # Verdeelde last: integreer over de belaste lengte
                length = rest[0] if rest else 0
                end_pos = pos + length
                
                n_steps = 50
                dx = length / n_steps
                for x_load in np.linspace(pos, end_pos, n_steps):
                    if support_pos <= x_load:
                        x = support_pos
                        a = x_load
                    else:
                        x = x_load
                        a = support_pos
                    
                    F[dofs[0]] -= value * dx * x*(L-a)*(2*L-x-a)/(6*EI*L)
                    if len(dofs) > 1:  # inklemming
                        F[dofs[1]] -= value * dx * x*(L-a)/(2*EI*L)
            
            elif load_type.lower() == "moment":
                # Moment: gebruik momenteninvloedslijn
                if support_pos <= pos:
                    x = support_pos
                    a = pos
                    if len(dofs) > 1:  # inklemming
                        F[dofs[1]] -= value * x*(L-a)/(EI*L)
                else:
                    x = pos
                    a = support_pos
                    if len(dofs) > 1:  # inklemming
                        F[dofs[1]] -= value * x*(L-a)/(EI*L)
    
    return K, F

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
if 'export_count' not in st.session_state:
    st.session_state.export_count = 0

def calculate_profile_properties(profile_type, height, width, wall_thickness, flange_thickness):
    """Cache profiel eigenschappen voor snellere berekeningen"""
    # Map profile types to calculation types
    calc_type = "Koker" if profile_type == "Koker" else ("U-profiel" if profile_type == "UNP" else "I-profiel")
    
    A = calculate_A(calc_type, height, width, wall_thickness, flange_thickness)
    I = calculate_I(calc_type, height, width, wall_thickness, flange_thickness)
    W = I / (height/2) if height > 0 else 0
    return A, I, W

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
    try:
        # Controleer invoer
        if h <= 0 or b <= 0 or t_w <= 0:
            st.error("❌ Profielafmetingen moeten positief zijn")
            return None
            
        if t_w >= min(h, b)/2:
            st.error("❌ Wanddikte te groot voor profiel")
            return None
            
        # Bereken traagheidsmoment
        if profile_type.lower() == "koker":
            # Koker: I = (BH³ - bh³)/12
            H = h  # Uitwendige hoogte
            B = b   # Uitwendige breedte
            t = t_w
            
            h_i = H - 2*t  # Inwendige hoogte
            b_i = B - 2*t  # Inwendige breedte
            
            if h_i <= 0 or b_i <= 0:
                st.error("❌ Wanddikte te groot voor profiel")
                return None
                
            I = (B*H**3 - b_i*h_i**3)/12
            
        elif profile_type.lower() in ["i-profiel", "u-profiel"]:
            if t_f is None or t_f <= 0:
                st.error("❌ Flensdikte moet positief zijn")
                return None
                
            H = h  # Totale hoogte
            B = b   # Flensbreedte
            tw = t_w    # Lijfdikte
            tf = t_f  # Flensdikte
            
            if tf >= H/2:
                st.error("❌ Flensdikte te groot voor profiel")
                return None
                
            # I-profiel: I = (bh³)/12 + 2×(BH³)/12
            hw = H - 2*tf  # Lijfhoogte
            if hw <= 0:
                st.error("❌ Flensdikte te groot voor profiel")
                return None
                
            I_web = tw * hw**3 / 12  # Lijf
            I_flange = B * tf**3 / 12 + B*tf * (H-tf)**2 / 2  # Flenzen
            I = I_web + 2*I_flange
            
        else:
            st.error("❌ Ongeldig profieltype")
            return None
            
        return I  # mm⁴
        
    except Exception as e:
        st.error(f"❌ Fout bij berekenen traagheidsmoment: {str(e)}")
        return None{{ ... }}

def calculate_A(profile_type, h, b, t_w, t_f=None):
    """Bereken oppervlakte voor verschillende profieltypes"""
    if profile_type.lower() == "koker":
        return (h * b) - ((h - 2*t_w) * (b - 2*t_w))
    elif profile_type.lower() in ["i-profiel", "u-profiel"]:
        # Flens oppervlakte
        A_f = 2 * (b * t_f)
        # Lijf oppervlakte
        h_w = h - 2*t_f
        A_w = t_w * h_w
        return A_f + A_w
    return 0

def main():
    st.set_page_config(
        page_title="BeamSolve Pro",
        page_icon="",
        layout="wide"
    )
    
    # Header
    col1, col2 = st.columns([3,1])
    with col1:
        st.title("BeamSolve Pro")
        st.markdown("Geavanceerde balkberekeningen voor constructeurs")
    with col2:
        st.markdown("### Versie")
        st.markdown("""
        <div style='background-color: #f8f9fa; padding: 10px; border-radius: 5px;'>
        <small>
        Free Edition<br>
        <span style='color: #6c757d;'>2/2 Exports Beschikbaar</span>
        </small>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
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
        st.title("BeamSolve Pro")
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
                    value=0.0,
                    min_value=0.0,
                    max_value=beam_length,
                    step=100.0,
                    format="%.0f",
                    help="mm",
                    key=f"load_pos_{i}"
                )
            with col2:
                if load_type in ["Verdeelde last", "Driehoekslast"]:
                    length = st.number_input(
                        "Lengte",
                        value=1000.0,
                        min_value=0.0,
                        max_value=beam_length,
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
        x, V, M, theta, y = analyze_beam(beam_length, supports, loads, profile_type, height, width, wall_thickness, flange_thickness, E)
        
        # Teken balkschema
        beam_fig = plot_beam_diagram(beam_length, supports, loads)
        st.plotly_chart(beam_fig, use_container_width=True)
        
        # Resultaten header
        st.markdown("""
        <style>
        .results-header {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .results-section {
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            border: 1px solid #e9ecef;
            margin-bottom: 20px;
        }
        </style>
        <div class="results-header">
        <h2> Berekeningsresultaten</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Grafieken sectie
        st.markdown("""
        <div class="results-section">
        <h3> Grafieken</h3>
        </div>
        """, unsafe_allow_html=True)
        
        results_plot = plot_results(x, V, M, theta, y, beam_length, supports, loads)
        st.plotly_chart(results_plot, use_container_width=True)
        
        # Maximale waarden en profiel eigenschappen
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="results-section">
            <h3> Maximale Waarden</h3>
            </div>
            """, unsafe_allow_html=True)
            
            max_vals = {
                "Dwarskracht": [f"{max(abs(np.min(V)), abs(np.max(V)))/1000:.1f}", "kN"],
                "Moment": [f"{max(abs(np.min(M)), abs(np.max(M)))/1e6:.1f}", "kNm"],
                "Rotatie": [f"{max(abs(np.min(theta)), abs(np.max(theta))):.4f}", "rad"],
                "Doorbuiging": [f"{max(abs(np.min(y)), abs(np.max(y))):.2f}", "mm"]
            }
            
            for key, (val, unit) in max_vals.items():
                st.metric(key, f"{val} {unit}")
        
        with col2:
            st.markdown("""
            <div class="results-section">
            <h3> Profiel Eigenschappen</h3>
            </div>
            """, unsafe_allow_html=True)
            
            A, I, W = calculate_profile_properties(profile_type, height, width, wall_thickness, flange_thickness)
            properties = {
                "Oppervlakte": [f"{A:.0f}", "mm²"],
                "Traagheidsmoment": [f"{I:.0f}", "mm⁴"],
                "Weerstandsmoment": [f"{W:.0f}", "mm³"],
                "Max. buigspanning": [f"{max(abs(np.min(M)), abs(np.max(M)))/W:.1f}", "N/mm²"]
            }
            
            for key, (val, unit) in properties.items():
                st.metric(key, f"{val} {unit}")
            
        # PDF Export sectie
        st.markdown("""
        <div class="results-section">
        <h3> Rapport Exporteren</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Maak een dictionary met alle beam data
        beam_data = {
            "profile_type": profile_type,
            "height": height,
            "width": width,
            "wall_thickness": wall_thickness,
            "flange_thickness": flange_thickness,
            "beam_length": beam_length,
            "supports": supports,
            "loads": loads,
            "max_deflection": max(abs(np.min(y)), abs(np.max(y))),
            "max_rotation": max(abs(np.min(theta)), abs(np.max(theta))),
            "max_shear": max(abs(np.min(V)), abs(np.max(V))),
            "max_moment": max(abs(np.min(M)), abs(np.max(M)))
        }
        
        # Demo export knop (beperkt tot 2 exports)
        remaining_exports = 2 - st.session_state.export_count
        if remaining_exports > 0:
            try:
                pdf_content = generate_pdf_report(beam_data, results_plot)
                col1, col2 = st.columns([3,1])
                with col1:
                    if st.download_button(
                        label=f" Download Rapport (PDF) - {remaining_exports} export(s) over",
                        data=pdf_content,
                        file_name="beamsolve_report.pdf",
                        mime="application/pdf",
                        key="download_report"
                    ):
                        st.session_state.export_count += 1
                with col2:
                    st.markdown(f"<small> Tip: Sla dit rapport op voor later gebruik</small>", unsafe_allow_html=True)
                        
                if remaining_exports == 1:
                    st.warning(" Dit is je laatste gratis export. Upgrade naar Pro voor onbeperkt gebruik!")
            except Exception as e:
                st.error(f"Fout bij genereren rapport: {str(e)}")
        else:
            st.error(" Je hebt je gratis exports gebruikt. Upgrade naar Pro voor onbeperkt gebruik!")
            col1, col2 = st.columns(2)
            with col1:
                if st.button(" Upgrade naar Pro", type="primary"):
                    st.markdown("### Contact voor Pro Licentie")
                    st.info("Neem contact op via info@beamsolve.nl voor een Pro licentie.")
            with col2:
                st.markdown("""
                <div style='background-color: #f8f9fa; padding: 15px; border-radius: 5px;'>
                <h4> Pro Voordelen</h4>
                <ul>
                <li>Onbeperkt PDF exports</li>
                <li>Geavanceerde belastingcombinaties</li>
                <li>Excel/CAD export</li>
                <li>Email support</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
