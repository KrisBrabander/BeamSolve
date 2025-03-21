import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.patches as patches

class MaterialTheme:
    def __init__(self):
        # Modern Light theme (default)
        self.light = {
            'bg': '#ffffff',
            'fg': '#2c3e50',
            'primary': '#3498db',
            'secondary': '#2ecc71',
            'accent': '#e74c3c',
            'surface': '#f8f9fa',
            'border': '#dee2e6'
        }
        
        # Modern Dark theme
        self.dark = {
            'bg': '#212529',
            'fg': '#f8f9fa',
            'primary': '#00b0ff',
            'secondary': '#00e676',
            'accent': '#ff1744',
            'surface': '#343a40',
            'border': '#495057'
        }
        
        self.current = self.light

    def toggle(self):
        self.current = self.dark if self.current == self.light else self.light
        return self.current

class ModernBuigingsCalculator:
    def __init__(self, root):
        self.root = root
        self.theme = MaterialTheme()
        root.title("Buigingsberekeningen")
        
        # Materiaal eigenschappen
        self.materials = {
            "Staal": 210000,  # N/mm²
            "Aluminium": 70000,  # N/mm²
            "RVS": 200000,  # N/mm²
            "Messing": 100000,
            "Koper": 120000,
            "Titanium": 114000
        }
        
        # HEM profielen bibliotheek
        self.profiles = {
            "HEM": {
                "100": {"h": 120, "b": 106, "tw": 12.0, "tf": 20.0},
                "120": {"h": 140, "b": 126, "tw": 12.5, "tf": 21.0},
                "140": {"h": 160, "b": 146, "tw": 13.0, "tf": 22.0},
                "160": {"h": 180, "b": 166, "tw": 14.0, "tf": 23.0},
                "180": {"h": 200, "b": 186, "tw": 14.5, "tf": 24.0},
                "200": {"h": 220, "b": 206, "tw": 15.0, "tf": 25.0},
                "220": {"h": 240, "b": 226, "tw": 15.5, "tf": 26.0},
                "240": {"h": 270, "b": 248, "tw": 18.0, "tf": 32.0},
                "260": {"h": 290, "b": 268, "tw": 18.0, "tf": 32.5},
                "280": {"h": 310, "b": 288, "tw": 18.5, "tf": 33.0},
                "300": {"h": 340, "b": 310, "tw": 21.0, "tf": 39.0},
                "320": {"h": 359, "b": 309, "tw": 21.0, "tf": 40.0},
                "340": {"h": 377, "b": 309, "tw": 21.0, "tf": 40.0},
                "360": {"h": 395, "b": 308, "tw": 21.0, "tf": 40.0},
                "400": {"h": 432, "b": 307, "tw": 21.0, "tf": 40.0},
                "450": {"h": 478, "b": 307, "tw": 21.0, "tf": 40.0},
                "500": {"h": 524, "b": 306, "tw": 21.0, "tf": 40.0},
                "550": {"h": 572, "b": 306, "tw": 21.0, "tf": 40.0},
                "600": {"h": 620, "b": 305, "tw": 21.0, "tf": 40.0},
                "650": {"h": 668, "b": 305, "tw": 21.0, "tf": 40.0},
                "700": {"h": 716, "b": 304, "tw": 21.0, "tf": 40.0},
                "800": {"h": 814, "b": 303, "tw": 21.0, "tf": 40.0},
                "900": {"h": 910, "b": 302, "tw": 21.0, "tf": 40.0},
                "1000": {"h": 1008, "b": 302, "tw": 21.0, "tf": 40.0}
            }
        }
        
        # Initialiseer belangrijke variabelen
        self.inputs = {}  # Dictionary voor invoervelden
        self.loads = []   # Lijst voor belastingen
        self.support_inputs = []  # Voor oplegging posities
        self.support_types = []   # Voor oplegging types
        
        # Basis styling
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Hoofdframe
        self.main_frame = ttk.Frame(root, padding=20)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Toolbar
        toolbar = ttk.Frame(self.main_frame)
        toolbar.pack(fill=tk.X, pady=(0, 15))
        
        # Titel
        ttk.Label(toolbar, text="Buigingsberekeningen").pack(side=tk.LEFT)
        
        # Theme toggle
        self.theme_btn = ttk.Button(toolbar, 
                                  text=" Dark Mode",
                                  command=self.toggle_theme)
        self.theme_btn.pack(side=tk.RIGHT)
        
        # Balkanimatie bovenaan
        self.beam_frame = ttk.LabelFrame(self.main_frame, text="Balkvervorming")
        self.beam_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Maak figuur voor balkanimatie
        self.fig_beam = Figure(figsize=(12, 4))
        self.ax_beam = self.fig_beam.add_subplot(111)
        self.canvas_beam = FigureCanvasTkAgg(self.fig_beam, master=self.beam_frame)
        self.canvas_beam.draw()
        self.canvas_beam.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Maak figuur voor grafieken
        self.fig_plots = Figure(figsize=(12, 8))
        self.axes = []
        for i in range(4):
            self.axes.append(self.fig_plots.add_subplot(4, 1, i+1))
        
        # Input container met drie kolommen
        self.input_container = ttk.Frame(self.main_frame)
        self.input_container.pack(fill=tk.X, pady=10)
        
        # Setup UI
        self.create_inputs()
        self.create_supports()
        self.setup_loads()
        
        # Grafieken onderaan
        self.plots_frame = ttk.LabelFrame(self.main_frame, text="Grafieken")
        self.plots_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.canvas_plots = FigureCanvasTkAgg(self.fig_plots, master=self.plots_frame)
        self.canvas_plots.draw()
        self.canvas_plots.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Statusbalk
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(root, textvariable=self.status_var)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        
        # Update grafieken
        self.update_beam_plot()
        
    def create_inputs(self):
        # Profielgegevens frame (links)
        profile_frame = ttk.LabelFrame(self.input_container, text="Profielgegevens")
        profile_frame.pack(side=tk.LEFT, padx=10, pady=5, fill=tk.Y)
        
        # Materiaal selector
        ttk.Label(profile_frame, text="Materiaal:").pack(anchor='w')
        self.material_type = ttk.Combobox(profile_frame, values=list(self.materials.keys()), state="readonly")
        self.material_type.set("Staal")
        self.material_type.pack(fill=tk.X, pady=(0, 10))
        
        # Profieltype selector
        ttk.Label(profile_frame, text="Profieltype:").pack(anchor='w')
        self.profile_type = ttk.Combobox(profile_frame, values=["HEM", "Koker", "I-profiel"], state="readonly")
        self.profile_type.set("HEM")
        self.profile_type.pack(fill=tk.X, pady=(0, 10))
        self.profile_type.bind('<<ComboboxSelected>>', self.on_profile_type_change)
        
        # HEM profiel selector
        ttk.Label(profile_frame, text="HEM Profiel:").pack(anchor='w')
        self.hem_type = ttk.Combobox(profile_frame, values=list(self.profiles["HEM"].keys()), state="readonly")
        self.hem_type.set("200")
        self.hem_type.pack(fill=tk.X, pady=(0, 10))
        self.hem_type.bind('<<ComboboxSelected>>', self.on_hem_type_change)
        
        # Standaard invoervelden
        standard_inputs = [
            ("Hoogte (mm)", "220"),
            ("Breedte (mm)", "206"),
            ("Wanddikte (mm)", "15.0"),
            ("Flensdikte (mm)", "25.0"),
        ]
        
        for label, default in standard_inputs:
            ttk.Label(profile_frame, text=label).pack(anchor='w')
            entry = ttk.Entry(profile_frame)
            entry.insert(0, default)
            entry.pack(fill=tk.X, pady=(0, 10))
            self.inputs[label] = entry
            entry.bind('<KeyRelease>', self.update_beam_plot)
        
        # Overspanning
        span_frame = ttk.LabelFrame(self.input_container, text="Overspanning")
        span_frame.pack(side=tk.LEFT, padx=10, pady=5, fill=tk.Y)
        
        ttk.Label(span_frame, text="Lengte (mm)").pack(anchor='w')
        length_entry = ttk.Entry(span_frame)
        length_entry.insert(0, "1000")
        length_entry.pack(fill=tk.X, pady=(0, 10))
        self.inputs["Lengte (mm)"] = length_entry

    def create_supports(self):
        """Maak invoervelden voor opleggingen"""
        support_frame = ttk.LabelFrame(self.input_container, text="Opleggingen")
        support_frame.pack(side=tk.LEFT, padx=10, pady=5, fill=tk.Y)
        
        # Aantal opleggingen selector
        ttk.Label(support_frame, text="Aantal opleggingen:").pack(anchor='w')
        self.support_count = ttk.Combobox(support_frame, values=["1 inklemming", "2 opleggingen", "3 opleggingen"], state="readonly")
        self.support_count.set("2 opleggingen")
        self.support_count.pack(fill=tk.X, pady=(0, 10))
        self.support_count.bind('<<ComboboxSelected>>', self.on_support_count_change)
        
        # Frame voor opleggingen
        self.supports_container = ttk.Frame(support_frame)
        self.supports_container.pack(fill=tk.BOTH, expand=True)
        
        # Maak standaard 2 opleggingen
        self.on_support_count_change()

    def setup_loads(self):
        # Belastingen frame (rechts)
        loads_frame = ttk.LabelFrame(self.input_container, text="Belastingen")
        loads_frame.pack(side=tk.LEFT, padx=10, pady=5, fill=tk.Y)
        
        # Type belasting
        ttk.Label(loads_frame, text="Type:").pack(anchor='w')
        self.load_type = tk.StringVar(value="Puntlast")
        load_type_cb = ttk.Combobox(loads_frame, textvariable=self.load_type,
                                   values=["Puntlast", "Gelijkmatig verdeeld", "Moment"],
                                   state="readonly", width=20)
        load_type_cb.pack(fill=tk.X, pady=(0, 10))
        
        # Waarde
        ttk.Label(loads_frame, text="Waarde:").pack(anchor='w')
        self.load_value = ttk.Entry(loads_frame, width=10)
        self.load_value.pack(fill=tk.X, pady=(0, 10))
        
        # Positie
        ttk.Label(loads_frame, text="Positie:").pack(anchor='w')
        self.load_position = ttk.Entry(loads_frame, width=10)
        self.load_position.pack(fill=tk.X, pady=(0, 10))
        
        # Lengte (voor verdeelde last)
        self.length_label = ttk.Label(loads_frame, text="Lengte:")
        self.length_label.pack(anchor='w')
        self.load_length = ttk.Entry(loads_frame, width=10)
        self.load_length.pack(fill=tk.X, pady=(0, 10))
        
        # Knoppen voor toevoegen/verwijderen
        btn_frame = ttk.Frame(loads_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(btn_frame, text="Toevoegen", command=self.add_load).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Verwijderen", command=self.remove_load).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Verwijder alle krachten", command=self.clear_loads).pack(side=tk.LEFT, padx=5)
        
        # Lijst met belastingen
        self.load_list = tk.Listbox(loads_frame, height=6)
        self.load_list.pack(fill=tk.BOTH, expand=True, pady=5)

    def toggle_theme(self):
        theme = self.theme.toggle()
        self.theme_btn.configure(text=" Light Mode" if theme == self.theme.dark else " Dark Mode")
        self.apply_theme()

    def apply_theme(self):
        """Pas het huidige thema toe op alle widgets"""
        theme = self.theme.current
        
        # Update style configuratie
        self.style.configure('TFrame', background=theme['bg'])
        self.style.configure('TLabelframe', background=theme['bg'])
        self.style.configure('TLabelframe.Label', background=theme['bg'], foreground=theme['fg'])
        self.style.configure('TLabel', background=theme['bg'], foreground=theme['fg'])
        self.style.configure('TButton', background=theme['primary'], foreground=theme['fg'])
        
        # Update listbox
        self.load_list.configure(bg=theme['bg'], fg=theme['fg'])
        
        # Update figuren
        self.fig_beam.patch.set_facecolor(theme['bg'])
        self.ax_beam.set_facecolor(theme['bg'])
        self.fig_plots.patch.set_facecolor(theme['bg'])
        for ax in self.axes:
            ax.set_facecolor(theme['bg'])
            ax.tick_params(colors=theme['fg'])
        
        # Update canvassen
        self.canvas_beam.draw()
        self.canvas_plots.draw()

    def update_beam_plot(self):
        """Update de balkvisualisatie"""
        try:
            self.ax_beam.clear()
            
            # Haal balklengte op
            L = float(self.inputs["Lengte (mm)"].get())
            
            # Bereken doorbuiging
            x = np.linspace(0, L, 100)
            y = np.zeros_like(x)
            for i, xi in enumerate(x):
                y[i] = -self.calculate_beam_response(xi)['y']  # Vermenigvuldig met -1
            
            # Schaal doorbuiging voor visualisatie
            scale = 1.0  # Basis schaalfactor
            if np.any(y != 0):
                # Automatische schaling om doorbuiging zichtbaar te maken
                max_defl = np.max(np.abs(y))
                if max_defl > 0:
                    desired_height = L / 10  # We willen dat de max doorbuiging ongeveer 1/10 van de lengte is
                    scale = desired_height / max_defl
            
            # Teken onvervormde balk (gestippeld)
            self.ax_beam.plot([0, L], [0, 0], 'k--', alpha=0.3)
            
            # Teken vervormde balk (doorgetrokken)
            self.ax_beam.plot(x, y * scale, 'b-', linewidth=2)
            
            # Teken opleggingen met duidelijke symbolen
            support_height = L/40
            for i, var in enumerate(self.support_inputs):
                try:
                    pos = float(var.get())
                    type = self.support_types[i].get()
                    
                    # Bereken hoogte op positie van oplegging
                    y_pos = -self.calculate_beam_response(pos)['y'] * scale
                    
                    if type == "Scharnier":
                        # Driehoek voor scharnier
                        self.ax_beam.plot([pos, pos-support_height/2, pos+support_height/2, pos],
                                       [y_pos, y_pos-support_height, y_pos-support_height, y_pos],
                                       'k-', linewidth=2)
                    elif type == "Rol":
                        # Cirkel voor rol
                        circle = patches.Circle((pos, y_pos-support_height/2),
                                             support_height/2, color='k', fill=False)
                        self.ax_beam.add_patch(circle)
                    elif type == "Inklemming":
                        # Verticale lijnen voor inklemming
                        inkl_width = support_height/2
                        y_range = np.array([y_pos-support_height, y_pos+support_height])
                        for x_pos in [pos-inkl_width/2, pos, pos+inkl_width/2]:
                            self.ax_beam.plot([x_pos, x_pos], y_range, 'k-', linewidth=2)
                except ValueError:
                    continue
            
            # Teken belastingen
            arrow_height = L/10
            for load in self.loads:
                pos, F, load_type, *rest = load
                try:
                    pos = float(pos)
                    F = float(F)
                    
                    # Bereken hoogte op positie van last
                    y_pos = -self.calculate_beam_response(pos)['y'] * scale
                    
                    if load_type == "Puntlast":
                        # Pijl voor puntlast (naar beneden voor positieve kracht)
                        if F > 0:  # Positieve kracht is naar beneden
                            self.ax_beam.arrow(pos, y_pos + arrow_height, 0, -arrow_height/2,
                                            head_width=L/50, head_length=arrow_height/4,
                                            fc='r', ec='r', linewidth=2)
                            # Waarde van de kracht
                            self.ax_beam.text(pos, y_pos + arrow_height*1.1,
                                           f'{F:.0f}N', ha='center', va='bottom')
                        else:  # Negatieve kracht is naar boven
                            self.ax_beam.arrow(pos, y_pos - arrow_height, 0, arrow_height/2,
                                            head_width=L/50, head_length=arrow_height/4,
                                            fc='r', ec='r', linewidth=2)
                            # Waarde van de kracht
                            self.ax_beam.text(pos, y_pos - arrow_height*1.1,
                                           f'{F:.0f}N', ha='center', va='top')
                    
                    elif load_type == "Gelijkmatig verdeeld":
                        length = float(rest[0])
                        q = F / length  # N/mm
                        x_range = np.linspace(pos, pos + length, 20)
                        arrow_spacing = length / 10
                        
                        # Teken pijlen voor verdeelde last
                        for x in np.arange(pos, pos + length + arrow_spacing/2, arrow_spacing):
                            y_val = -self.calculate_beam_response(x)['y'] * scale
                            if F > 0:  # Positieve last is naar beneden
                                self.ax_beam.arrow(x, y_val + arrow_height/2, 0, -arrow_height/4,
                                                head_width=L/100, head_length=arrow_height/8,
                                                fc='r', ec='r', linewidth=1)
                            else:  # Negatieve last is naar boven
                                self.ax_beam.arrow(x, y_val - arrow_height/2, 0, arrow_height/4,
                                                head_width=L/100, head_length=arrow_height/8,
                                                fc='r', ec='r', linewidth=1)
                        
                        # Waarde van de verdeelde last
                        mid_pos = pos + length/2
                        if F > 0:
                            self.ax_beam.text(mid_pos, y_pos + arrow_height*0.6,
                                           f'{q:.1f}N/mm', ha='center', va='bottom')
                        else:
                            self.ax_beam.text(mid_pos, y_pos - arrow_height*0.6,
                                           f'{q:.1f}N/mm', ha='center', va='top')
                except ValueError:
                    continue
            
            # Stel plotgrenzen in
            self.ax_beam.set_xlim(-L/20, L*1.05)
            total_height = L/5
            self.ax_beam.set_ylim(-total_height/2, total_height/2)
            
            # Labels
            self.ax_beam.set_xlabel('Positie (mm)')
            self.ax_beam.set_ylabel('Doorbuiging (geschaald)')
            self.ax_beam.grid(True, alpha=0.3)
            
            # Update canvas
            self.canvas_beam.draw()
            
        except Exception as e:
            self.status_var.set(f"Fout bij plotten: {str(e)}")

    def calculate_beam_response(self, x):
        """Bereken de mechanische respons op positie x"""
        try:
            # Valideer alle benodigde invoer
            required_inputs = ["Lengte (mm)", "Hoogte (mm)", "Breedte (mm)", "Wanddikte (mm)"]
            for input_name in required_inputs:
                if not self.inputs[input_name].get().strip():
                    print(f"Waarschuwing: {input_name} is niet ingevuld")
                    return {'M': 0, 'V': 0, 'y': 0, 'sigma': 0}
            
            L = float(self.inputs["Lengte (mm)"].get())
            E = self.get_E()  # N/mm²
            
            # Haal profielgegevens op
            h = float(self.inputs["Hoogte (mm)"].get())
            b = float(self.inputs["Breedte (mm)"].get())
            t = float(self.inputs["Wanddikte (mm)"].get())
            
            # Valideer numerieke waarden
            if L <= 0 or h <= 0 or b <= 0 or t <= 0:
                print("Waarschuwing: Alle afmetingen moeten positief zijn")
                return {'M': 0, 'V': 0, 'y': 0, 'sigma': 0}
            
            # Debug info
            if x == L/2:  # Print info voor middenpunt
                print(f"\nProfiel: {b}x{h}x{t}mm")
                print(f"E-modulus: {E} N/mm²")
            
            # Bereken traagheidsmoment (mm⁴)
            if self.profile_type.get() == "Koker":
                I = (b * h**3 - (b-2*t) * (h-2*t)**3) / 12
            else:  # I-profiel of HEM
                tf = float(self.inputs["Flensdikte (mm)"].get())
                hw = h - 2*tf  # Hoogte van het lijf
                I = (b * h**3 - (b-t) * hw**3) / 12
            
            if x == L/2:  # Print info voor middenpunt
                print(f"Traagheidsmoment I: {I:.0f} mm⁴")
            
            # Verzamel opleggingen
            supports = []
            for i, var in enumerate(self.support_inputs):
                try:
                    pos = float(var.get())
                    type = self.support_types[i].get()
                    supports.append((pos, type))
                except ValueError:
                    continue
            
            if x == L/2:  # Print info voor middenpunt
                print("\nOpleggingen:")
                for pos, type in supports:
                    print(f"  {type} @ {pos}mm")
            
            if not supports:
                return {'M': 0, 'V': 0, 'y': 0, 'sigma': 0}
            
            # Sorteer opleggingen op positie
            supports.sort(key=lambda x: x[0])
            
            # Voor één inklemming
            if len(supports) == 1 and supports[0][1] == "Inklemming":
                x0 = supports[0][0]  # Positie van inklemming
                
                # Initialiseer resultaten
                M = 0  # Moment
                V = 0  # Dwarskracht
                y = 0  # Doorbuiging
                
                # Voor elke belasting
                for load in self.loads:
                    pos, F, load_type, *rest = load
                    try:
                        F = float(F)
                        
                        if load_type == "Puntlast":
                            # Alleen als de last voorbij de inklemming valt
                            if pos > x0:
                                a = pos - x0  # Afstand vanaf inklemming
                                
                                if x <= x0:  # Voor de inklemming
                                    y = 0
                                    V = -F  # Negatief voor juiste richting
                                    M = -F * a  # Negatief voor juiste richting
                                elif x <= pos:  # Tussen inklemming en last
                                    y = F * (x - x0)**2 * (3*a - (x - x0)) / (6 * E * I)
                                    V = -F  # Negatief voor juiste richting
                                    M = -F * (pos - x)  # Negatief voor juiste richting
                                else:  # Voorbij de last
                                    y = F * a**2 * (3*(x - x0) - a) / (6 * E * I)
                                    V = 0
                                    M = 0
                        
                        elif load_type == "Gelijkmatig verdeeld":
                            length = float(rest[0])
                            start = max(x0, pos)
                            end = pos + length
                            
                            if start < end:
                                q = F / length  # N/mm
                                
                                if x <= x0:  # Voor de inklemming
                                    y = 0
                                    V = -F  # Negatief voor juiste richting
                                    M = -q * (end - start) * ((end + start)/2 - x0)  # Negatief voor juiste richting
                                else:  # Na de inklemming
                                    # Complexe formule voor doorbuiging bij verdeelde last
                                    if x <= start:  # Voor de last
                                        y = q * (
                                            (x - x0)**2 * (4*end - start - 3*x) / 24
                                        ) / (E * I)
                                    elif x <= end:  # Onder de last
                                        y = q * (
                                            (x - x0)**2 * (4*end - start - 3*x) / 24 +
                                            (x - start)**4 / 24
                                        ) / (E * I)
                                    else:  # Na de last
                                        y = q * length * (x - (end + start)/2) * (x - x0)**2 / (6 * E * I)
                                    
                                    # Dwarskracht en moment
                                    if x <= end:
                                        V = -q * (end - max(x, start))  # Negatief voor juiste richting
                                        M = -V * (end - x)/2  # Negatief voor juiste richting
                                    else:
                                        V = 0
                                        M = 0
                    except ValueError:
                        continue
                
                # Bereken spanning
                sigma = abs(M) * h/(2*I) if I > 0 else 0
                
                if x == L/2:
                    print(f"\nDoorbuiging op x={x:.0f}mm:")
                    print(f"  y = {y:.3f}mm")
                    print(f"  M = {M:.0f}Nmm")
                    print(f"  V = {V:.0f}N")
                    print(f"  σ = {sigma:.1f}N/mm²")
                
                return {
                    'M': M,
                    'V': V,
                    'y': y,
                    'sigma': sigma
                }
            
            # Voor twee scharnieren
            elif len(supports) == 2 and all(type == "Scharnier" for _, type in supports):
                x1, x2 = supports[0][0], supports[1][0]
                L_eff = x2 - x1  # Effectieve lengte tussen opleggingen
                
                # Initialiseer resultaten
                M = 0  # Moment
                V = 0  # Dwarskracht
                y = 0  # Doorbuiging
                
                # Voor elke belasting
                for load in self.loads:
                    pos, F, load_type, *rest = load
                    try:
                        F = float(F)
                        
                        if load_type == "Puntlast":
                            # Alleen als de last tussen de opleggingen valt
                            if x1 <= pos <= x2:
                                a = pos - x1  # Afstand vanaf linker oplegging
                                b = x2 - pos   # Afstand tot rechter oplegging
                                
                                # Reactiekrachten
                                R2 = F * a / L_eff
                                R1 = F - R2
                                
                                # Doorbuiging voor puntlast
                                if x <= pos:
                                    # Links van de last
                                    y += F * b * x * (L_eff**2 - b**2 - x**2) / (6 * E * I)
                                else:
                                    # Rechts van de last
                                    y += F * a * (L_eff - x) * (2*L_eff*x - x**2 - a**2) / (6 * E * I)
                                
                                # Moment
                                if x1 <= x <= x2:
                                    if x <= pos:
                                        M += R1 * (x - x1)
                                    else:
                                        M += R1 * (x - x1) - F * (x - pos)
                                
                                # Dwarskracht
                                if x < pos:
                                    V += R1
                                else:
                                    V += R1 - F
                        
                        elif load_type == "Gelijkmatig verdeeld":
                            length = float(rest[0])
                            start = max(x1, pos)
                            end = min(x2, pos + length)
                            
                            if start < end:
                                q = F / length  # N/mm
                                L_load = end - start
                                x_c = (start + end) / 2  # Zwaartepunt van de last
                                F_total = q * L_load
                                
                                # Reactiekrachten
                                R2 = F_total * (x_c - x1) / L_eff
                                R1 = F_total - R2
                                
                                # Doorbuiging voor gelijkmatig verdeelde last
                                if x1 <= x <= x2:
                                    y += q * (
                                        (x - x1)**4 / 24 -  # Effect van linker reactiekracht
                                        L_eff * (x - x1)**3 / 12 +  # Effect van moment
                                        L_eff**2 * (x - x1) / 24  # Effect van doorbuiging
                                    ) / (E * I)
                                
                                # Moment
                                if x1 <= x <= x2:
                                    M += R1 * (x - x1) - q * (max(0, x - start))**2 / 2
                                
                                # Dwarskracht
                                if x < start:
                                    V += R1
                                elif x < end:
                                    V += R1 - q * (x - start)
                                else:
                                    V += R1 - q * L_load
                    except ValueError:
                        continue
                
                # Bereken spanning
                sigma = abs(M) * h/(2*I) if I > 0 else 0
                
                if x == L/2:
                    print(f"\nDoorbuiging op x={x:.0f}mm:")
                    print(f"  y = {y:.3f}mm")
                    print(f"  M = {M:.0f}Nmm")
                    print(f"  V = {V:.0f}N")
                    print(f"  σ = {sigma:.1f}N/mm²")
                
                return {
                    'M': M,
                    'V': V,
                    'y': y,
                    'sigma': sigma
                }
            
            # Voor andere gevallen
            return {'M': 0, 'V': 0, 'y': 0, 'sigma': 0}
            
        except Exception as e:
            print(f"Fout in berekening: {str(e)}")
            return {'M': 0, 'V': 0, 'y': 0, 'sigma': 0}

    def get_E(self):
        """Haal E-modulus op van geselecteerd materiaal"""
        try:
            return self.materials[self.material_type.get()]
        except:
            return 210000  # Standaard voor staal
    
    def on_load_type_change(self, event=None):
        """Update interface op basis van belastingtype"""
        load_type = self.load_type.get()
        
        # Toon/verberg lengte veld
        if load_type == "Gelijkmatig verdeeld":
            self.length_label.pack()
            self.load_length.pack()
        else:
            self.length_label.pack_forget()
            self.load_length.pack_forget()
    
    def add_load(self):
        """Voeg een nieuwe belasting toe"""
        try:
            # Haal waarden op
            load_type = self.load_type.get()
            value = float(self.load_value.get())
            position = float(self.load_position.get())
            
            # Valideer positie
            L = float(self.inputs["Lengte (mm)"].get())
            if not (0 <= position <= L):
                raise ValueError(f"Positie moet tussen 0 en {L}mm liggen")
            
            # Haal lengte op voor verdeelde last
            length = 0
            if load_type == "Gelijkmatig verdeeld":
                length = float(self.load_length.get())
                if not (0 < length <= L - position):
                    raise ValueError(f"Lengte moet tussen 0 en {L-position}mm liggen")
            
            # Voeg toe aan lijst
            load = [position, value, load_type]
            if length > 0:
                load.append(length)
            
            self.loads.append(load)
            
            # Update listbox
            if load_type == "Gelijkmatig verdeeld":
                self.load_list.insert(tk.END, 
                    f"{load_type}: {value/length:.1f}N/mm over {length}mm @ {position:.0f}mm")
            else:
                unit = "N" if load_type == "Puntlast" else "Nmm"
                self.load_list.insert(tk.END, 
                    f"{load_type}: {value}{unit} @ {position:.0f}mm")
            
            # Wis invoervelden
            self.load_value.delete(0, tk.END)
            self.load_position.delete(0, tk.END)
            if load_type == "Gelijkmatig verdeeld":
                self.load_length.delete(0, tk.END)
            
            # Update visualisatie
            self.update_beam_plot()
            
        except ValueError as e:
            self.status_var.set(str(e))
    
    def remove_load(self):
        """Verwijder geselecteerde belasting"""
        try:
            selection = self.load_list.curselection()
            if selection:
                index = selection[0]
                self.load_list.delete(index)
                self.loads.pop(index)
                
                # Update visualisatie
                self.update_beam_plot()
        except Exception as e:
            self.status_var.set(f"Fout bij verwijderen: {str(e)}")

    def clear_loads(self):
        """Verwijder alle belastingen"""
        self.loads.clear()  # Maak de lijst met belastingen leeg
        self.load_list.delete(0, tk.END)  # Leeg de listbox
        
        # Update de plot
        self.update_beam_plot()

    def on_profile_type_change(self, event=None):
        """Update interface wanneer profieltype verandert"""
        profile_type = self.profile_type.get()
        
        if profile_type == "HEM":
            # Toon HEM selector en update waarden
            self.hem_type.pack(fill=tk.X, pady=(0, 10))
            self.on_hem_type_change()
        else:
            # Verberg HEM selector
            self.hem_type.pack_forget()
            
            if profile_type == "Koker":
                # Standaardwaarden voor koker
                self.inputs["Hoogte (mm)"].delete(0, tk.END)
                self.inputs["Hoogte (mm)"].insert(0, "100")
                self.inputs["Breedte (mm)"].delete(0, tk.END)
                self.inputs["Breedte (mm)"].insert(0, "50")
                self.inputs["Wanddikte (mm)"].delete(0, tk.END)
                self.inputs["Wanddikte (mm)"].insert(0, "4")
                self.inputs["Flensdikte (mm)"].delete(0, tk.END)
                self.inputs["Flensdikte (mm)"].insert(0, "4")
            else:  # I-profiel
                # Standaardwaarden voor I-profiel
                self.inputs["Hoogte (mm)"].delete(0, tk.END)
                self.inputs["Hoogte (mm)"].insert(0, "200")
                self.inputs["Breedte (mm)"].delete(0, tk.END)
                self.inputs["Breedte (mm)"].insert(0, "100")
                self.inputs["Wanddikte (mm)"].delete(0, tk.END)
                self.inputs["Wanddikte (mm)"].insert(0, "6")
                self.inputs["Flensdikte (mm)"].delete(0, tk.END)
                self.inputs["Flensdikte (mm)"].insert(0, "8")
        
        self.update_beam_plot()

    def on_hem_type_change(self, event=None):
        """Update profielafmetingen wanneer HEM type verandert"""
        if self.profile_type.get() == "HEM":
            hem_size = self.hem_type.get()
            profile = self.profiles["HEM"][hem_size]
            
            # Update invoervelden
            self.inputs["Hoogte (mm)"].delete(0, tk.END)
            self.inputs["Hoogte (mm)"].insert(0, str(profile["h"]))
            self.inputs["Breedte (mm)"].delete(0, tk.END)
            self.inputs["Breedte (mm)"].insert(0, str(profile["b"]))
            self.inputs["Wanddikte (mm)"].delete(0, tk.END)
            self.inputs["Wanddikte (mm)"].insert(0, str(profile["tw"]))
            self.inputs["Flensdikte (mm)"].delete(0, tk.END)
            self.inputs["Flensdikte (mm)"].insert(0, str(profile["tf"]))
            
            self.update_beam_plot()

    def on_support_count_change(self, event=None):
        """Update oplegging inputs wanneer aantal verandert"""
        # Verwijder bestaande inputs
        for widget in self.supports_container.winfo_children():
            widget.destroy()
        self.support_inputs.clear()
        self.support_types.clear()
        
        # Bepaal aantal opleggingen
        config = self.support_count.get()
        if config == "1 inklemming":
            count = 1
            default_type = "Inklemming"
        else:
            count = 3 if "3" in config else 2
            default_type = "Scharnier"
        
        # Maak nieuwe inputs voor elke oplegging
        for i in range(count):
            frame = ttk.Frame(self.supports_container)
            frame.pack(fill=tk.X, pady=2)
            
            # Label
            ttk.Label(frame, text=f"Oplegging {i+1}:").pack(side=tk.LEFT)
            
            # Positie invoer
            pos_var = ttk.Entry(frame, width=10)
            pos_var.pack(side=tk.LEFT, padx=5)
            if count == 1:  # Inklemming
                pos_var.insert(0, "0")  # Standaard op begin van de balk
            else:  # Scharnieren/rollen
                if i == 0:
                    pos_var.insert(0, "0")  # Eerste op begin
                elif i == count-1:
                    try:
                        L = float(self.inputs["Lengte (mm)"].get())
                        pos_var.insert(0, str(L))  # Laatste op eind
                    except ValueError:
                        pos_var.insert(0, "1000")
                else:
                    try:
                        L = float(self.inputs["Lengte (mm)"].get())
                        pos_var.insert(0, str(L/2))  # Middelste op helft
                    except ValueError:
                        pos_var.insert(0, "500")
            
            self.support_inputs.append(pos_var)
            
            # Type selector
            type_var = ttk.Combobox(frame, values=["Scharnier", "Rol", "Inklemming"] if count > 1 else ["Inklemming"],
                                  width=10, state="readonly")
            type_var.set(default_type)
            type_var.pack(side=tk.LEFT, padx=5)
            self.support_types.append(type_var)
            
            # Bind updates
            pos_var.bind('<KeyRelease>', lambda e: self.update_beam_plot())
            type_var.bind('<<ComboboxSelected>>', lambda e: self.update_beam_plot())
        
        # Update berekeningen
        self.update_beam_plot()

if __name__ == '__main__':
    root = tk.Tk()
    root.geometry("1200x800")
    app = ModernBuigingsCalculator(root)
    root.mainloop()
