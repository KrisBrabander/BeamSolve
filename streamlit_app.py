import streamlit as st
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import os
import sys

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.profiles import get_profile_list, get_profile_dimensions
from src.report import generate_report_html, save_report

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
