import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Initialize session state
if 'loads' not in st.session_state:
    st.session_state.loads = []
if 'load_count' not in st.session_state:
    st.session_state.load_count = 0
if 'supports' not in st.session_state:
    st.session_state.supports = []

# Page config
st.set_page_config(
    page_title="BeamCAD 2025 Enterprise",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS voor licht thema
st.markdown("""
    <style>
        .stApp {
            background-color: #ffffff;
        }
        
        h1, h2, h3 {
            color: #1f77b4;
        }
        
        .css-1d391kg {
            background-color: #f8f9fa;
        }
        
        .stTextInput > div > div > input {
            background-color: #ffffff;
            color: #2c3e50;
        }
        
        .stSelectbox > div > div > select {
            background-color: #ffffff;
            color: #2c3e50;
        }
        
        .stButton > button {
            background-color: #1f77b4;
            color: #ffffff;
        }
        
        .grid-item {
            background: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .plot-container {
            background: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div style='padding: 1rem; background: linear-gradient(90deg, #1f77b4 0%, #4fa1d8 100%); border-radius: 8px; margin-bottom: 2rem;'>
        <h1 style='color: white; margin: 0; font-size: 1.8rem;'>BeamCAD 2025 Enterprise</h1>
        <p style='color: #e6f3ff; margin: 0.5rem 0 0 0; font-size: 1rem;'>Professional Engineering Analysis Suite</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar en main content containers
sidebar = st.sidebar
main = st.container()

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
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("<b>Doorbuiging</b>", "<b>Moment</b>", "<b>Rotatie</b>"),
        vertical_spacing=0.12,
        specs=[[{"type": "scatter"}], [{"type": "scatter"}], [{"type": "scatter"}]]
    )

    fig.add_trace(
        go.Scatter(
            x=x, y=deflection,
            name="Doorbuiging",
            line=dict(color='#1f77b4', width=2),
            hovertemplate="<b>Positie:</b> %{x:.1f} mm<br><b>Doorbuiging:</b> %{y:.2f} mm<extra></extra>"
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=x, y=M,
            name="Moment",
            line=dict(color='#2ca02c', width=2),
            hovertemplate="<b>Positie:</b> %{x:.1f} mm<br><b>Moment:</b> %{y:.0f} Nmm<extra></extra>"
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=x, y=rotation,
            name="Rotatie",
            line=dict(color='#ff7f0e', width=2),
            hovertemplate="<b>Positie:</b> %{x:.1f} mm<br><b>Rotatie:</b> %{y:.6f} rad<extra></extra>"
        ),
        row=3, col=1
    )

    fig.update_layout(
        height=800,
        showlegend=False,
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        font=dict(
            family="Segoe UI, sans-serif",
            color='#2c3e50',
            size=12
        ),
        margin=dict(l=50, r=20, t=60, b=20),
        hovermode='x unified'
    )

    fig.update_xaxes(
        showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)',
        showline=True, linewidth=1, linecolor='#2c3e50',
        zeroline=True, zerolinewidth=1, zerolinecolor='#2c3e50',
        color='#2c3e50',
        title_text="Positie (mm)",
        title_font=dict(size=12, color='#2c3e50'),
        tickfont=dict(size=10)
    )

    fig.update_yaxes(
        showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)',
        showline=True, linewidth=1, linecolor='#2c3e50',
        zeroline=True, zerolinewidth=1, zerolinecolor='#2c3e50',
        color='#2c3e50',
        title_font=dict(size=12, color='#2c3e50'),
        tickfont=dict(size=10)
    )

    fig.update_yaxes(title_text="Doorbuiging (mm)", row=1, col=1)
    fig.update_yaxes(title_text="Moment (Nmm)", row=2, col=1)
    fig.update_yaxes(title_text="Rotatie (rad)", row=3, col=1)

    return fig

def plot_beam_diagram(beam_length, supports, loads):
    """Plot een schematische weergave van de balk met steunpunten en belastingen"""
    fig = go.Figure()
    
    # Teken de balk
    fig.add_trace(go.Scatter(
        x=[0, beam_length],
        y=[0, 0],
        mode='lines',
        name='Balk',
        line=dict(color='#2c3e50', width=4),
        hoverinfo='skip'
    ))
    
    # Teken steunpunten
    for pos, support_type in supports:
        if support_type == "Inklemming":
            # Teken rechthoek voor inklemming
            fig.add_trace(go.Scatter(
                x=[pos-20, pos-20, pos+20, pos+20],
                y=[-40, 40, 40, -40],
                fill="toself",
                mode='lines',
                name='Inklemming',
                line=dict(color='#2ecc71'),
                hovertemplate=f"Inklemming<br>x = {pos} mm"
            ))
        elif support_type == "Scharnier":
            # Teken driehoek voor scharnier
            fig.add_trace(go.Scatter(
                x=[pos-20, pos, pos+20],
                y=[-40, 0, -40],
                fill="toself",
                mode='lines',
                name='Scharnier',
                line=dict(color='#3498db'),
                hovertemplate=f"Scharnier<br>x = {pos} mm"
            ))
        else:  # Rol
            # Teken cirkel met driehoek voor rol
            fig.add_trace(go.Scatter(
                x=[pos-20, pos, pos+20],
                y=[-40, 0, -40],
                fill="toself",
                mode='lines',
                name='Rol',
                line=dict(color='#e74c3c'),
                hovertemplate=f"Rol<br>x = {pos} mm"
            ))
            # Teken cirkels voor rol
            theta = np.linspace(0, 2*np.pi, 50)
            r = 5
            x = r * np.cos(theta) + pos
            y = r * np.sin(theta) - 45
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='lines',
                name='Rol',
                line=dict(color='#e74c3c'),
                hoverinfo='skip',
                showlegend=False
            ))
    
    # Teken belastingen
    for pos, F, load_type, *rest in loads:
        if load_type == "Puntlast":
            # Teken pijl voor puntlast
            arrow_length = 60 if F > 0 else -60
            fig.add_trace(go.Scatter(
                x=[pos, pos],
                y=[0, arrow_length],
                mode='lines+markers',
                name=f'Puntlast {F}N',
                line=dict(color='#e67e22', width=2),
                marker=dict(symbol='arrow-down' if F > 0 else 'arrow-up', size=15),
                hovertemplate=f"Puntlast<br>F = {F} N<br>x = {pos} mm"
            ))
        elif load_type == "Gelijkmatig verdeeld":
            # Teken meerdere pijlen voor verdeelde belasting
            length = float(rest[0])
            q = F / length
            n_arrows = min(int(length/50), 10)  # Maximaal 10 pijlen
            dx = length / n_arrows
            arrow_length = 40 if F > 0 else -40
            for i in range(n_arrows + 1):
                x_pos = pos + i * dx
                if x_pos <= beam_length:
                    fig.add_trace(go.Scatter(
                        x=[x_pos, x_pos],
                        y=[0, arrow_length],
                        mode='lines+markers',
                        name=f'q = {q:.1f} N/mm',
                        line=dict(color='#9b59b6', width=1),
                        marker=dict(symbol='arrow-down' if F > 0 else 'arrow-up', size=10),
                        hovertemplate=f"Verdeelde last<br>q = {q:.1f} N/mm<br>x = {x_pos:.0f} mm",
                        showlegend=i==0
                    ))
            
            # Teken lijn boven de pijlen
            fig.add_trace(go.Scatter(
                x=[pos, pos + length],
                y=[arrow_length, arrow_length],
                mode='lines',
                line=dict(color='#9b59b6', width=2),
                hoverinfo='skip',
                showlegend=False
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
        height=300,
        plot_bgcolor='white',
        paper_bgcolor='white',
        title=dict(
            text="Balkschema",
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top',
            font=dict(size=16, color='#2c3e50')
        ),
        xaxis=dict(
            title="Positie (mm)",
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0,0,0,0.1)',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='#2c3e50',
            showline=True,
            linewidth=1,
            linecolor='#2c3e50',
            mirror=True
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            showticklabels=False,
            range=[-100, 100]
        )
    )
    
    return fig

def analyze_beam(beam_length, supports, loads, profile_type, height, width, wall_thickness, flange_thickness, E):
    """Hoofdfunctie voor balkanalyse"""
    n_points = 200
    x = np.linspace(0, beam_length, n_points)
    M = np.zeros_like(x)
    rotation = np.zeros_like(x)
    deflection = np.zeros_like(x)
    
    # Bereken moment voor elke positie
    for i, xi in enumerate(x):
        # Bereken momenten van belastingen
        for pos, F, load_type, *rest in loads:
            if load_type == "Puntlast":
                if xi >= pos:
                    M[i] += F * (xi - pos)
            elif load_type == "Gelijkmatig verdeeld":
                length = float(rest[0])
                q = F / length
                start = max(pos, xi)
                end = min(beam_length, pos + length)
                if start < end:
                    M[i] += 0.5 * q * (end - start) * (end + start - 2*xi)
        
        # Pas randvoorwaarden toe op basis van ondersteuningen
        for pos, support_type in supports:
            if support_type == "Inklemming" and xi >= pos:
                # Inklemming: Moment = 0 op positie, rotatie = 0
                M[i] = 0
                rotation[i] = 0
            elif support_type in ["Scharnier", "Rol"] and abs(xi - pos) < 1e-6:
                # Scharnier/Rol: Doorbuiging = 0 op positie
                deflection[i] = 0
    
    # Bereken rotatie en doorbuiging
    I = calculate_I(profile_type, height, width, wall_thickness, flange_thickness)
    dx = beam_length / (n_points - 1)
    
    # Voorwaartse integratie voor rotatie en doorbuiging
    for i in range(1, n_points):
        if not any(abs(x[i] - pos) < 1e-6 and type == "Inklemming" for pos, type in supports):
            rotation[i] = rotation[i-1] + M[i-1] * dx / (E * I)
        if not any(abs(x[i] - pos) < 1e-6 and type in ["Scharnier", "Rol"] for pos, type in supports):
            deflection[i] = deflection[i-1] + rotation[i-1] * dx
    
    return x, M, rotation, deflection

# Sidebar content
with sidebar:
    st.markdown("""
        <div style='background: #f8f9fa; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border: 1px solid #e0e0e0;'>
            <h3 style='color: #1f77b4; margin: 0 0 0.5rem 0; font-size: 1.1rem;'>Profielgegevens</h3>
            <div style='height: 2px; background: linear-gradient(90deg, #1f77b4 0%, #4fa1d8 100%); margin-bottom: 1rem;'></div>
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
        <div style='background: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 1rem 0; border: 1px solid #e0e0e0;'>
            <h3 style='color: #1f77b4; margin: 0 0 0.5rem 0; font-size: 1.1rem;'>Overspanning</h3>
            <div style='height: 2px; background: linear-gradient(90deg, #1f77b4 0%, #4fa1d8 100%); margin-bottom: 1rem;'></div>
        </div>
    """, unsafe_allow_html=True)
    
    beam_length = st.number_input(
        "Lengte (mm)",
        min_value=100,
        max_value=10000,
        value=1000,
        help="Totale lengte van de balk in millimeters"
    )

    # Ondersteuningen interface
    support_count = st.selectbox(
        "Aantal steunpunten",
        [1, 2, 3],
        help="Selecteer het aantal steunpunten"
    )

    supports = []
    if support_count == 1:
        support_type = st.selectbox(
            "Type ondersteuning",
            ["Inklemming"],
            help="Type ondersteuning"
        )
        pos = st.slider(
            "Positie inklemming",
            min_value=0,
            max_value=beam_length,
            value=0,
            help="Positie van de inklemming vanaf links"
        )
        supports.append((pos, support_type))
    else:
        for i in range(support_count):
            col1, col2 = st.columns(2)
            with col1:
                support_type = st.selectbox(
                    f"Type steunpunt {i+1}",
                    ["Scharnier", "Rol"],
                    help=f"Type ondersteuning voor steunpunt {i+1}",
                    key=f"support_type_{i}"
                )
            with col2:
                if i == 0:  # Eerste steunpunt
                    pos = st.number_input(
                        f"Positie steunpunt {i+1}",
                        min_value=0,
                        max_value=beam_length//3,
                        value=0,
                        help=f"Positie van steunpunt {i+1} vanaf links",
                        key=f"support_pos_{i}"
                    )
                elif i == support_count - 1:  # Laatste steunpunt
                    pos = st.number_input(
                        f"Positie steunpunt {i+1}",
                        min_value=2*beam_length//3,
                        max_value=beam_length,
                        value=beam_length,
                        help=f"Positie van steunpunt {i+1} vanaf links",
                        key=f"support_pos_{i}"
                    )
                else:  # Middelste steunpunt (alleen bij 3 steunpunten)
                    pos = st.number_input(
                        f"Positie steunpunt {i+1}",
                        min_value=beam_length//3,
                        max_value=2*beam_length//3,
                        value=beam_length//2,
                        help=f"Positie van steunpunt {i+1} vanaf links",
                        key=f"support_pos_{i}"
                    )
            supports.append((pos, support_type))
    
    st.session_state.supports = supports

    st.markdown("""
        <div style='background: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 1rem 0; border: 1px solid #e0e0e0;'>
            <h3 style='color: #1f77b4; margin: 0 0 0.5rem 0; font-size: 1.1rem;'>Materiaal</h3>
            <div style='height: 2px; background: linear-gradient(90deg, #1f77b4 0%, #4fa1d8 100%); margin-bottom: 1rem;'></div>
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
        <div style='background: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 1rem 0; border: 1px solid #e0e0e0;'>
            <h3 style='color: #1f77b4; margin: 0 0 0.5rem 0; font-size: 1.1rem;'>Belastingen</h3>
            <div style='height: 2px; background: linear-gradient(90deg, #1f77b4 0%, #4fa1d8 100%); margin-bottom: 1rem;'></div>
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

# Main content
with main:
    # Toon balkschema bovenaan
    beam_diagram = plot_beam_diagram(beam_length, st.session_state.supports, st.session_state.loads)
    st.plotly_chart(beam_diagram, use_container_width=True, config={
        'displayModeBar': False
    })

    tab1, tab2 = st.tabs(["Analyse", "Belastingen"])
    
    with tab1:
        if len(st.session_state.loads) > 0:
            # Voer analyse uit
            x, M, rotation, deflection = analyze_beam(
                beam_length, st.session_state.supports, st.session_state.loads,
                profile_type, height, width, wall_thickness, flange_thickness, E
            )
            
            # Plot resultaten
            fig = make_subplots(rows=3, cols=1,
                              subplot_titles=("Moment", "Rotatie", "Doorbuiging"),
                              shared_xaxes=True,
                              vertical_spacing=0.1)
            
            # Moment diagram
            fig.add_trace(
                go.Scatter(x=x, y=M, name="Moment",
                          line=dict(color='#2ecc71', width=2)),
                row=1, col=1
            )
            
            # Rotatie diagram
            fig.add_trace(
                go.Scatter(x=x, y=rotation, name="Rotatie",
                          line=dict(color='#3498db', width=2)),
                row=2, col=1
            )
            
            # Doorbuiging diagram
            fig.add_trace(
                go.Scatter(x=x, y=deflection, name="Doorbuiging",
                          line=dict(color='#e74c3c', width=2)),
                row=3, col=1
            )
            
            # Update layout
            fig.update_layout(
                height=600,
                showlegend=False,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            # Update x-assen
            fig.update_xaxes(title_text="Positie (mm)", row=3, col=1,
                           showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)',
                           zeroline=True, zerolinewidth=1, zerolinecolor='rgba(0,0,0,0.2)',
                           showline=True, linewidth=1, linecolor='black', mirror=True)
            
            # Update y-assen
            fig.update_yaxes(title_text="M (Nmm)", row=1, col=1,
                           showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)',
                           zeroline=True, zerolinewidth=1, zerolinecolor='rgba(0,0,0,0.2)',
                           showline=True, linewidth=1, linecolor='black', mirror=True)
            fig.update_yaxes(title_text="φ (rad)", row=2, col=1,
                           showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)',
                           zeroline=True, zerolinewidth=1, zerolinecolor='rgba(0,0,0,0.2)',
                           showline=True, linewidth=1, linecolor='black', mirror=True)
            fig.update_yaxes(title_text="v (mm)", row=3, col=1,
                           showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)',
                           zeroline=True, zerolinewidth=1, zerolinecolor='rgba(0,0,0,0.2)',
                           showline=True, linewidth=1, linecolor='black', mirror=True)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Toon maximale waarden
            max_moment = max(abs(np.min(M)), abs(np.max(M)))
            max_rotation = max(abs(np.min(rotation)), abs(np.max(rotation)))
            max_deflection = max(abs(np.min(deflection)), abs(np.max(deflection)))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Max. moment", f"{max_moment:.2f} Nmm")
            with col2:
                st.metric("Max. rotatie", f"{max_rotation:.6f} rad")
            with col3:
                st.metric("Max. doorbuiging", f"{max_deflection:.2f} mm")
        else:
            st.info("Voeg belastingen toe om de analyse te starten")
    
    with tab2:
        # Belastingen interface
        st.markdown("""
            <div style='background: #f8f9fa; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border: 1px solid #e0e0e0;'>
                <h3 style='color: #1f77b4; margin: 0 0 0.5rem 0; font-size: 1.1rem;'>Belastingen</h3>
                <div style='height: 2px; background: linear-gradient(90deg, #1f77b4 0%, #4fa1d8 100%); margin-bottom: 1rem;'></div>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            load_type = st.selectbox(
                "Type belasting",
                ["Puntlast", "Gelijkmatig verdeeld"],
                key=f"load_type_{st.session_state.load_count}"
            )
        
        with col2:
            if load_type == "Puntlast":
                load_value = st.number_input(
                    "Kracht (N)",
                    value=1000,
                    help="Positieve waarde voor neerwaartse kracht",
                    key=f"load_value_{st.session_state.load_count}"
                )
            else:
                load_value = st.number_input(
                    "Totale kracht (N)",
                    value=1000,
                    help="Totale kracht van de verdeelde belasting",
                    key=f"load_value_{st.session_state.load_count}"
                )
        
        col3, col4 = st.columns(2)
        with col3:
            load_pos = st.number_input(
                "Startpositie (mm)",
                min_value=0,
                max_value=beam_length,
                value=beam_length//2,
                help="Positie vanaf linkerzijde",
                key=f"load_pos_{st.session_state.load_count}"
            )
        
        with col4:
            if load_type == "Gelijkmatig verdeeld":
                load_length = st.number_input(
                    "Lengte (mm)",
                    min_value=10,
                    max_value=beam_length,
                    value=min(500, beam_length),
                    help="Lengte van de verdeelde belasting",
                    key=f"load_length_{st.session_state.load_count}"
                )
        
        col5, col6 = st.columns(2)
        with col5:
            if st.button("Toevoegen", use_container_width=True):
                if load_type == "Puntlast":
                    st.session_state.loads.append((load_pos, load_value, load_type))
                else:
                    st.session_state.loads.append((load_pos, load_value, load_type, load_length))
                st.session_state.load_count += 1
                st.rerun()
        
        with col6:
            if st.button("Reset", use_container_width=True):
                st.session_state.loads = []
                st.session_state.load_count = 0
                st.rerun()
        
        # Toon huidige belastingen
        if len(st.session_state.loads) > 0:
            st.markdown("### Huidige belastingen")
            for i, load in enumerate(st.session_state.loads):
                if load[2] == "Puntlast":
                    st.write(f"{i+1}. Puntlast: {load[1]}N op x = {load[0]}mm")
                else:
                    st.write(f"{i+1}. Verdeelde last: {load[1]}N over {load[3]}mm vanaf x = {load[0]}mm")
