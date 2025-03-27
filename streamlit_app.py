import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
import io
import os
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
# from scipy.integrate import cumtrapz
# from beam_solver import BeamSolver

# Alternatief voor cumtrapz als scipy niet beschikbaar is
def custom_cumtrapz(y, x, initial=0):
    """Eigen implementatie van cumtrapz voor het geval scipy niet beschikbaar is"""
    result = np.zeros_like(y)
    result[0] = initial
    for i in range(1, len(y)):
        result[i] = result[i-1] + 0.5 * (y[i] + y[i-1]) * (x[i] - x[i-1])
    return result

# Geïntegreerde BeamSolver klasse
class BeamSolver:
    def __init__(self, beam_length, supports, loads, EI):
        self.L = beam_length
        self.supports = sorted(supports, key=lambda x: x[0])
        self.loads = loads
        self.EI = EI
        self.x = np.linspace(0, beam_length, 500)
        self._validate_input()

    def _validate_input(self):
        """Controleer invoerconsistentie"""
        if any(pos < 0 or pos > self.L for pos, _ in self.supports):
            raise ValueError("Ongeldige steunpuntpositie")
        if self.EI <= 0:
            raise ValueError("Buigstijfheid moet positief zijn")

    def solve(self):
        """Hoofdberekeningsroutine"""
        self._calculate_reactions()
        self._calculate_internal_forces()
        self._calculate_deflection()
        return self.get_results()

    def get_results(self):
        return {
            'x': self.x,
            'V': self.V,
            'M': self.M,
            'y': self.y,
            'theta': self.theta,
            'reactions': self.reactions
        }

    def _calculate_reactions(self):
        """Bepaal reactiekrachten met drie-momentenvergelijking"""
        n = len(self.supports)
        support_pos = [s[0] for s in self.supports]

        if n == 1:  # Inklemming
            self._fixed_support_reactions()
        elif n == 2:  # Statisch bepaald
            self._simple_beam_reactions()
        else:  # Statisch onbepaald
            self._continuous_beam_reactions()

    def _fixed_support_reactions(self):
        """Reacties voor ingeklemde balk"""
        pos = self.supports[0][0]
        V_total = 0
        M_total = 0

        for load in self.loads:
            p, val, ltype, *rest = load
            if ltype.lower() == "puntlast":
                V_total += val
                M_total += val * (p - pos)
            elif ltype.lower() == "verdeelde last":
                length = rest[0]
                q = val
                V_total += q * length
                M_total += q * length * (p + length/2 - pos)
            elif ltype.lower() == "moment":
                M_total += val

        self.reactions = {pos: -V_total, f"M_{pos}": -M_total}

    def _simple_beam_reactions(self):
        """Statisch bepaalde ligger met 2 steunpunten"""
        a, b = [s[0] for s in self.supports]
        L = b - a
        R_a, R_b = 0, 0
        M_a, M_b = 0, 0

        for load in self.loads:
            p, val, ltype, *rest = load
            if ltype.lower() == "puntlast":
                R_a += val * (b - p)/L
                R_b += val * (p - a)/L
            elif ltype.lower() == "verdeelde last":
                q, length = val, rest[0]
                x1 = max(a, p)
                x2 = min(b, p + length)
                if x1 < x2:
                    R_a += q * ((x2**2 - x1**2)/(2*L) - a*(x2 - x1)/L)
                    R_b += q * (b*(x2 - x1)/L - (x2**2 - x1**2)/(2*L))
            elif ltype.lower() == "moment":
                M_a += val * (b - p)/L
                M_b += val * (p - a)/L

        self.reactions = {a: R_a, b: R_b}
        
        # Voeg inklemming momenten toe indien nodig
        if self.supports[0][1].lower() == "inklemming":
            self.reactions[f"M_{a}"] = M_a
        if self.supports[1][1].lower() == "inklemming":
            self.reactions[f"M_{b}"] = M_b

    def _continuous_beam_reactions(self):
        """Statisch onbepaalde ligger met meer dan 2 steunpunten"""
        n = len(self.supports)
        support_pos = [s[0] for s in self.supports]
        reactions = {}

        for i in range(n-1):
            a, b = support_pos[i], support_pos[i+1]
            L = b - a
            R_a, R_b = 0, 0
            M_a, M_b = 0, 0

            for load in self.loads:
                p, val, ltype, *rest = load
                if ltype.lower() == "puntlast":
                    if a <= p <= b:
                        R_a += val * (b - p)/L
                        R_b += val * (p - a)/L
                elif ltype.lower() == "verdeelde last":
                    q, length = val, rest[0]
                    x1 = max(a, p)
                    x2 = min(b, p + length)
                    if x1 < x2:
                        R_a += q * ((x2**2 - x1**2)/(2*L) - a*(x2 - x1)/L)
                        R_b += q * (b*(x2 - x1)/L - (x2**2 - x1**2)/(2*L))
                elif ltype.lower() == "moment":
                    if a <= p <= b:
                        M_a += val * (b - p)/L
                        M_b += val * (p - a)/L

            reactions[a] = R_a
            reactions[b] = R_b

            # Voeg inklemming momenten toe indien nodig
            if self.supports[i][1].lower() == "inklemming":
                reactions[f"M_{a}"] = M_a
            if self.supports[i+1][1].lower() == "inklemming":
                reactions[f"M_{b}"] = M_b

        self.reactions = reactions

    def _calculate_internal_forces(self):
        """Gecorrigeerde integratie met superpositie"""
        V = np.zeros_like(self.x)
        M = np.zeros_like(self.x)
        
        # Verzamel steunpunt posities voor controle
        support_positions = [pos for pos, _ in self.supports]
        
        # Reactiekrachten
        for pos, R in self.reactions.items():
            if not pos.startswith("M_"):  # Alleen krachten, geen momenten
                idx = np.searchsorted(self.x, float(pos))
                if idx < len(self.x):
                    V[idx:] += R
                    M[idx:] += R * (self.x[idx:] - float(pos))
        
        # Reactiemomenten
        for key, val in self.reactions.items():
            if key.startswith("M_"):
                pos = float(key.split("_")[1])
                idx = np.searchsorted(self.x, pos)
                if idx < len(self.x):
                    # Moment heeft geen effect op dwarskracht, alleen op moment
                    M[idx:] += val
        
        # Uitwendige belastingen
        for load in self.loads:
            pos, val, ltype, *rest = load
            
            # Controleer of de last op een steunpunt valt
            on_support = any(abs(pos - sp) < 1e-6 for sp in support_positions)
            
            if ltype.lower() == "puntlast":
                # Als de puntlast op een steunpunt valt, wordt deze direct opgenomen
                # door het steunpunt en heeft geen effect op de interne krachten
                if not on_support:
                    idx = np.searchsorted(self.x, pos)
                    if idx < len(self.x):
                        V[idx:] -= val
                        M[idx:] -= val * (self.x[idx:] - pos)
            elif ltype.lower() == "verdeelde last":
                length = rest[0]
                q = val
                start_idx = np.searchsorted(self.x, pos)
                end_idx = np.searchsorted(self.x, pos + length)
                
                # Punten binnen de verdeelde last
                for i in range(start_idx, min(end_idx, len(self.x))):
                    dx = self.x[i] - pos
                    V[i] -= q * dx
                    M[i] -= q * dx**2 / 2
                
                # Punten voorbij de verdeelde last
                if end_idx < len(self.x):
                    V[end_idx:] -= q * length
                    M[end_idx:] -= q * length * (self.x[end_idx:] - pos - length/2)
            elif ltype.lower() == "moment":
                idx = np.searchsorted(self.x, pos)
                if idx < len(self.x):
                    # Extern moment heeft geen effect op dwarskracht, alleen op moment
                    M[idx:] -= val
        
        self.V = V
        self.M = M

    def _calculate_deflection(self):
        """Dubbele integratie met correcte randvoorwaarden"""
        # Bereken dwarskracht en moment
        V, M = self.V, self.M
        
        # Eerste integratie: hoekverdraaiing
        theta = custom_cumtrapz(M/self.EI, self.x, initial=0)
        
        # Tweede integratie: doorbuiging
        y = custom_cumtrapz(theta, self.x, initial=0)
        
        # Pas randvoorwaarden aan
        support_positions = [s[0] for s in self.supports]
        support_indices = [np.abs(self.x - pos).argmin() for pos in support_positions]
        
        # Stel lineair systeem op voor correctie
        A = []
        b = []
        
        for idx in support_indices:
            A.append([1, self.x[idx]])
            b.append(-y[idx])
        
        # Los op met least squares
        A = np.array(A)
        b = np.array(b)
        
        if len(A) >= 2:  # Minstens 2 steunpunten nodig voor unieke oplossing
            try:
                C, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                # Corrigeer doorbuiging
                y_corrected = y + C[0] + C[1] * self.x
            except:
                # Fallback als lstsq faalt
                st.warning("⚠️ Randvoorwaarden konden niet exact worden toegepast")
                # Eenvoudige correctie: verschuif zodat eerste steunpunt op 0 ligt
                y_corrected = y - y[support_indices[0]]
        else:
            # Eén steunpunt: verschuif zodat dat punt op 0 ligt
            y_corrected = y - y[support_indices[0]]
        
        self.theta = theta
        self.y = y_corrected

def _free_moment(x1, x2, loads):
    """Bereken vrije veldmoment tussen x1 en x2"""
    M = 0
    for load in loads:
        pos, val, ltype, *rest = load
        if ltype.lower() == "puntlast" and x1 <= pos <= x2:
            a = pos - x1
            L = x2 - x1
            M += val * a * (L**2 - a**2) / (6*L)
        elif ltype.lower() == "verdeelde last":
            length = rest[0]
            start = max(x1, pos)
            end = min(x2, pos + length)
            if start < end:
                a = start - x1
                b = end - x1
                L = x2 - x1
                M += val * (b**3 - a**3)/(6*L) - val * (b**4 - a**4)/(24*L**2)
    return M

def _point_load_contribution(x, loads):
    """Bereken de bijdrage van puntlasten op positie x"""
    R = 0
    for load in loads:
        pos, val, ltype, *rest = load
        if ltype.lower() == "puntlast" and abs(pos - x) < 1e-6:
            R += val
        elif ltype.lower() == "verdeelde last":
            length = rest[0]
            if x >= pos and x <= pos + length:
                # Punt ligt binnen de verdeelde last
                R += val * (min(x + 1e-6, pos + length) - max(x - 1e-6, pos))
    return R

def calculate_reactions_matrix(beam_length, supports, loads):
    """Bereken reactiekrachten voor statisch onbepaalde systemen met matrixmethode.
    Specifiek geoptimaliseerd voor 3 steunpunten."""
    
    # Sorteer steunpunten
    supports = sorted(supports, key=lambda x: x[0])
    n = len(supports)
    
    if n != 3:
        return None  # Deze functie is alleen voor 3 steunpunten
    
    try:
        # Haal posities en types op
        pos1, type1 = supports[0]
        pos2, type2 = supports[1]
        pos3, type3 = supports[2]
        
        # Bereken lengtes van overspanningen
        L1 = pos2 - pos1
        L2 = pos3 - pos2
        
        # Stel stijfheidsmatrix op voor 3 steunpunten
        # Voor eenvoud: we nemen aan dat EI constant is over de balk
        
        # Compatibiliteitsmatrix voor 3 steunpunten
        # [ a11 a12 a13 ] [ R1 ]   [ d1 ]
        # [ a21 a22 a23 ] [ R2 ] = [ d2 ]
        # [ a31 a32 a33 ] [ R3 ]   [ d3 ]
        
        # Flexibiliteitscoëfficiënten (invloedsgetallen)
        # a_ij = doorbuiging op punt i door eenheidslast op punt j
        
        # Voor een balk met 3 steunpunten:
        a11 = L1**3 / 3  # Doorbuiging op punt 1 door eenheidslast op punt 1
        a12 = L1**2 * L2 / 2  # Doorbuiging op punt 1 door eenheidslast op punt 2
        a13 = L1 * L2**2 / 2  # Doorbuiging op punt 1 door eenheidslast op punt 3
        
        a21 = L1**2 * L2 / 2  # Doorbuiging op punt 2 door eenheidslast op punt 1
        a22 = (L1 + L2)**3 / 3  # Doorbuiging op punt 2 door eenheidslast op punt 2
        a23 = L2**2 * L1 / 2  # Doorbuiging op punt 2 door eenheidslast op punt 3
        
        a31 = L1 * L2**2 / 2  # Doorbuiging op punt 3 door eenheidslast op punt 1
        a32 = L2**2 * L1 / 2  # Doorbuiging op punt 3 door eenheidslast op punt 2
        a33 = L2**3 / 3  # Doorbuiging op punt 3 door eenheidslast op punt 3
        
        # Stel matrix op
        A = np.array([
            [a11, a12, a13],
            [a21, a22, a23],
            [a31, a32, a33]
        ])
        
        # Bereken belastingsvector (doorbuigingen door externe belasting)
        d = np.zeros(3)
        
        # Verwerk belastingen
        for load in loads:
            pos, value, load_type, *rest = load
            
            if load_type.lower() == "puntlast":
                # Puntlast: bereken doorbuiging op elk steunpunt
                for i, (sp, _) in enumerate(supports):
                    if pos < sp:
                        # Last links van steunpunt
                        d[i] += value * (pos - pos1) * (sp - pos)**2 / 6
                    else:
                        # Last rechts van steunpunt
                        d[i] += value * (pos3 - pos) * (pos - sp)**2 / 6
                        
            elif load_type.lower() == "verdeelde last":
                length = rest[0]
                q = value
                
                # Verdeelde last: bereken doorbuiging op elk steunpunt
                start = pos
                end = pos + length
                
                for i, (sp, _) in enumerate(supports):
                    # Complexe integratie voor doorbuiging door verdeelde last
                    # Vereenvoudigde benadering: vervang door equivalente puntlast
                    equiv_pos = (start + end) / 2
                    equiv_val = q * length
                    
                    if equiv_pos < sp:
                        # Last links van steunpunt
                        d[i] += equiv_val * (equiv_pos - pos1) * (sp - equiv_pos)**2 / 6
                    else:
                        # Last rechts van steunpunt
                        d[i] += equiv_val * (pos3 - equiv_pos) * (equiv_pos - sp)**2 / 6
        
        # Los matrix op: A * R = d
        # Omdat we doorbuiging = 0 willen op steunpunten, is d = 0
        # Maar we hebben evenwichtsvergelijking nodig: som krachten = som belastingen
        
        # Bereken totale belasting
        total_load = 0
        for load in loads:
            pos, value, load_type, *rest = load
            if load_type.lower() == "puntlast":
                total_load += value
            elif load_type.lower() == "verdeelde last":
                length = rest[0]
                total_load += value * length
        
        # Evenwichtsvergelijking: R1 + R2 + R3 = total_load
        # Vervang laatste rij door evenwichtsvergelijking
        A[2] = np.array([1, 1, 1])
        d[2] = total_load
        
        # Los matrix op
        R = np.linalg.solve(A, d)
        
        # Maak reactions dictionary
        reactions = {}
        for i, (pos, _) in enumerate(supports):
            reactions[pos] = -R[i]  # Negatief omdat reacties omhoog positief zijn
        
        # Voeg inklemming momenten toe indien nodig
        if type1.lower() == "inklemming":
            # Bereken moment in inklemming
            M1 = 0
            for load in loads:
                pos_load, value, load_type, *rest = load
                if load_type.lower() == "puntlast":
                    M1 += value * (pos_load - pos1)
                elif load_type.lower() == "verdeelde last":
                    length = rest[0]
                    q = value
                    x_c = pos_load + length/2
                    M1 += q * length * (x_c - pos1)
            reactions[f"M_{pos1}"] = -M1
            
        if type3.lower() == "inklemming":
            # Bereken moment in inklemming
            M3 = 0
            for load in loads:
                pos_load, value, load_type, *rest = load
                if load_type.lower() == "puntlast":
                    M3 += value * (pos3 - pos_load)
                elif load_type.lower() == "verdeelde last":
                    length = rest[0]
                    q = value
                    x_c = pos_load + length/2
                    M3 += q * length * (pos3 - x_c)
            reactions[f"M_{pos3}"] = M3
        
        return reactions
        
    except Exception as e:
        st.error(f"❌ Fout bij matrixberekening: {str(e)}")
        return None

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
                    V_total += value
                    M_total += value * (pos_load - pos)
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
                
        elif n == 3:
            # Drie steunpunten - statisch onbepaald systeem
            # Gebruik matrixmethode voor exacte oplossing
            matrix_reactions = calculate_reactions_matrix(beam_length, supports, loads)
            if matrix_reactions:
                return matrix_reactions
            else:
                # Fallback naar benaderde methode als matrixmethode faalt
                st.warning("⚠️ Matrixmethode gefaald, gebruik benaderde methode")
                # Benaderde methode: verdeel belasting over steunpunten
                # Bereken totale belasting
                V_total = 0
                for load in loads:
                    pos, value, load_type, *rest = load
                    if load_type.lower() == "puntlast":
                        V_total += value
                    elif load_type.lower() == "verdeelde last":
                        length = rest[0]
                        V_total += value * length
                
                # Verdeel belasting over steunpunten (vereenvoudigde benadering)
                # Middelste steunpunt krijgt meer belasting
                weights = [0.25, 0.5, 0.25]  # Gewichten voor links, midden, rechts
                
                for i, (pos, type) in enumerate(supports):
                    reactions[pos] = -V_total * weights[i]
                
                # Voeg inklemming momenten toe indien nodig
                for i, (pos, type) in enumerate(supports):
                    if type.lower() == "inklemming":
                        # Vereenvoudigde momentberekening voor inklemming
                        if i == 0:  # Eerste steunpunt
                            x_ref = pos
                            M_total = 0
                            for load in loads:
                                pos_load, value, load_type, *rest = load
                                if load_type.lower() == "puntlast":
                                    M_total += value * (pos_load - x_ref) * 0.5
                                elif load_type.lower() == "verdeelde last":
                                    length = rest[0]
                                    q = value
                                    x_c = pos_load + length/2
                                    M_total += q * length * (x_c - x_ref) * 0.5
                            reactions[f"M_{pos}"] = -M_total
                        elif i == n-1:  # Laatste steunpunt
                            x_ref = pos
                            M_total = 0
                            for load in loads:
                                pos_load, value, load_type, *rest = load
                                if load_type.lower() == "puntlast":
                                    M_total += value * (x_ref - pos_load) * 0.5
                                elif load_type.lower() == "verdeelde last":
                                    length = rest[0]
                                    q = value
                                    x_c = pos_load + length/2
                                    M_total += q * length * (x_ref - x_c) * 0.5
                            reactions[f"M_{pos}"] = M_total
        
        else:
            # Meer dan 3 steunpunten - complexe statisch onbepaalde systemen
            st.warning("⚠️ Systemen met meer dan 3 steunpunten worden benaderd (niet exact)")
            
            # Bereken totale belasting
            V_total = 0
            for load in loads:
                pos, value, load_type, *rest = load
                if load_type.lower() == "puntlast":
                    V_total += value
                elif load_type.lower() == "verdeelde last":
                    length = rest[0]
                    V_total += value * length
            
            # Verdeel belasting over steunpunten (vereenvoudigde benadering)
            # Betere benadering: gewogen verdeling op basis van positie
            weights = []
            for i, (pos, _) in enumerate(supports):
                if i == 0 or i == n-1:
                    # Randsteunpunten krijgen minder belasting
                    weights.append(0.5)
                else:
                    # Tussensteunpunten krijgen meer belasting
                    weights.append(1.0)
            
            total_weight = sum(weights)
            for i, (pos, type) in enumerate(supports):
                reactions[pos] = -V_total * weights[i] / total_weight
            
            # Voeg inklemming momenten toe indien nodig
            for i, (pos, type) in enumerate(supports):
                if type.lower() == "inklemming":
                    # Vereenvoudigde momentberekening voor inklemming
                    if i == 0:  # Eerste steunpunt
                        x_ref = pos
                        M_total = 0
                        for load in loads:
                            pos_load, value, load_type, *rest = load
                            if load_type.lower() == "puntlast":
                                M_total += value * (pos_load - x_ref) * 0.5
                            elif load_type.lower() == "verdeelde last":
                                length = rest[0]
                                q = value
                                x_c = pos_load + length/2
                                M_total += q * length * (x_c - x_ref) * 0.5
                        reactions[f"M_{pos}"] = -M_total
                    elif i == n-1:  # Laatste steunpunt
                        x_ref = pos
                        M_total = 0
                        for load in loads:
                            pos_load, value, load_type, *rest = load
                            if load_type.lower() == "puntlast":
                                M_total += value * (x_ref - pos_load) * 0.5
                            elif load_type.lower() == "verdeelde last":
                                length = rest[0]
                                q = value
                                x_c = pos_load + length/2
                                M_total += q * length * (x_ref - x_c) * 0.5
                        reactions[f"M_{pos}"] = M_total
    
    except Exception as e:
        st.error(f"❌ Fout bij berekenen reactiekrachten: {str(e)}")
        return None
        
    return reactions

def calculate_internal_forces(x, beam_length, supports, loads, reactions):
    """Gecorrigeerde integratie met superpositie"""
    V = np.zeros_like(x)
    M = np.zeros_like(x)
    
    # Verzamel steunpunt posities voor controle
    support_positions = [pos for pos, _ in supports]
    
    # Reactiekrachten
    for pos, R in reactions.items():
        if not pos.startswith("M_"):  # Alleen krachten, geen momenten
            idx = np.searchsorted(x, float(pos))
            if idx < len(x):
                V[idx:] += R
                M[idx:] += R * (x[idx:] - float(pos))
    
    # Reactiemomenten
    for key, val in reactions.items():
        if key.startswith("M_"):
            pos = float(key.split("_")[1])
            idx = np.searchsorted(x, pos)
            if idx < len(x):
                # Moment heeft geen effect op dwarskracht, alleen op moment
                M[idx:] += val
    
    # Uitwendige belastingen
    for load in loads:
        pos, val, ltype, *rest = load
        
        # Controleer of de last op een steunpunt valt
        on_support = any(abs(pos - sp) < 1e-6 for sp in support_positions)
        
        if ltype.lower() == "puntlast":
            # Als de puntlast op een steunpunt valt, wordt deze direct opgenomen
            # door het steunpunt en heeft geen effect op de interne krachten
            if not on_support:
                idx = np.searchsorted(x, pos)
                if idx < len(x):
                    V[idx:] -= val
                    M[idx:] -= val * (x[idx:] - pos)
        elif ltype.lower() == "verdeelde last":
            length = rest[0]
            q = val
            start_idx = np.searchsorted(x, pos)
            end_idx = np.searchsorted(x, pos + length)
            
            # Punten binnen de verdeelde last
            for i in range(start_idx, min(end_idx, len(x))):
                dx = x[i] - pos
                V[i] -= q * dx
                M[i] -= q * dx**2 / 2
            
            # Punten voorbij de verdeelde last
            if end_idx < len(x):
                V[end_idx:] -= q * length
                M[end_idx:] -= q * length * (x[end_idx:] - pos - length/2)
        elif ltype.lower() == "moment":
            idx = np.searchsorted(x, pos)
            if idx < len(x):
                # Extern moment heeft geen effect op dwarskracht, alleen op moment
                M[idx:] -= val
    
    return V, M

def calculate_deflection(x, beam_length, supports, loads, reactions, EI):
    """Dubbele integratie met correcte randvoorwaarden"""
    # Bereken dwarskracht en moment
    V, M = calculate_internal_forces(x, beam_length, supports, loads, reactions)
    
    # Eerste integratie: hoekverdraaiing
    theta = custom_cumtrapz(M/EI, x, initial=0)
    
    # Tweede integratie: doorbuiging
    y = custom_cumtrapz(theta, x, initial=0)
    
    # Pas randvoorwaarden aan
    support_positions = [s[0] for s in supports]
    support_indices = [np.abs(x - pos).argmin() for pos in support_positions]
    
    # Stel lineair systeem op voor correctie
    A = []
    b = []
    
    for idx in support_indices:
        A.append([1, x[idx]])
        b.append(-y[idx])
    
    # Los op met least squares
    A = np.array(A)
    b = np.array(b)
    
    if len(A) >= 2:  # Minstens 2 steunpunten nodig voor unieke oplossing
        try:
            C, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            # Corrigeer doorbuiging
            y_corrected = y + C[0] + C[1] * x
        except:
            # Fallback als lstsq faalt
            st.warning("⚠️ Randvoorwaarden konden niet exact worden toegepast")
            # Eenvoudige correctie: verschuif zodat eerste steunpunt op 0 ligt
            y_corrected = y - y[support_indices[0]]
    else:
        # Eén steunpunt: verschuif zodat dat punt op 0 ligt
        y_corrected = y - y[support_indices[0]]
    
    return theta, y_corrected

def analyze_beam(beam_length, supports, loads, profile_type, height, width, wall_thickness, flange_thickness, E):
    """Analyseer de balk met de BeamSolver klasse"""
    try:
        # Bereken traagheidsmoment
        I = calculate_moment_of_inertia(profile_type, height, width, wall_thickness, flange_thickness)
        if I is None or I <= 0:
            st.error("❌ Ongeldige profielafmetingen")
            return None, None, None, None, None, None
        
        # Bereken buigstijfheid EI
        EI = E * I
        
        # Gebruik de BeamSolver klasse voor de berekeningen
        solver = BeamSolver(beam_length, supports, loads, EI)
        results = solver.solve()
        
        # Haal resultaten op
        x = results['x']
        V = results['V']
        M = results['M']
        theta = results['theta']
        y = results['y']
        reactions = results['reactions']
        
        return x, V, M, theta, y, reactions
    
    except Exception as e:
        st.error(f"❌ Fout bij analyse: {str(e)}")
        return None, None, None, None, None, None

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
            num_arrows = min(int(length*8), 15)  # Aantal pijlen afhankelijk van lengte
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
    """Plot resultaten in één figuur met moderne styling"""
    
    # Kleurenpalet
    colors = {
        'shear': '#2ecc71',     # Groen
        'moment': '#e74c3c',    # Rood
        'rotation': '#f39c12',  # Oranje
        'deflection': '#3498db', # Blauw
        'grid': '#ecf0f1',      # Lichtgrijs
        'zero': '#bdc3c7',      # Middengrijs
        'support': '#34495e',   # Donkerblauw
        'load': '#e74c3c'       # Rood
    }
    
    # Maak figuur met subplots - balk bovenaan en groter
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            "<b>Doorbuiging [mm]</b>",
            "<b>Dwarskracht [kN]</b>",
            "<b>Moment [kNm]</b>",
            "<b>Rotatie [rad]</b>"
        ),
        vertical_spacing=0.08,
        row_heights=[0.4, 0.2, 0.2, 0.2]  # Balk krijgt meer ruimte
    )
    
    # Plot doorbuiging (nu bovenaan en groter)
    fig.add_trace(
        go.Scatter(
            x=x/1000, y=y,
            mode='lines',
            name='Doorbuiging',
            line=dict(color=colors['deflection'], width=4),
            fill='tozeroy',
            fillcolor=f'rgba(52, 152, 219, 0.2)'
        ),
        row=1, col=1
    )
    
    # Teken de balk zelf als een lijn
    fig.add_trace(
        go.Scatter(
            x=x/1000, 
            y=[0] * len(x),
            mode='lines',
            name='Balk',
            line=dict(color='black', width=2),
            showlegend=False
        ),
        row=1, col=1
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
                    size=16,
                    color=colors['support'],
                    line=dict(width=2, color='white')
                ),
                showlegend=False
            ),
            row=1, col=1
        )
        # Voeg label toe voor steunpunttype
        fig.add_annotation(
            x=pos/1000,
            y=-max(abs(min(y)), abs(max(y)))*0.2,
            text=type,
            showarrow=False,
            font=dict(size=10, color=colors['support']),
            row=1, col=1
        )
    
    # Voeg belastingen toe aan doorbuigingsgrafiek
    for load in loads:
        pos, value, load_type, *rest = load
        
        if load_type.lower() == "puntlast":
            # Puntlast pijl omlaag
            fig.add_trace(
                go.Scatter(
                    x=[pos/1000],
                    y=[max(abs(min(y)), abs(max(y)))*0.5 if any(y) else 10.0],  # Veilige offset boven de balk
                    mode='markers',
                    name=f'{value/1000:.1f} kN',
                    marker=dict(
                        symbol='arrow-down',
                        size=16,
                        color=colors['load']
                    ),
                    showlegend=False
                ),
                row=1, col=1
            )
            # Voeg waarde label toe
            fig.add_annotation(
                x=pos/1000,
                y=max(abs(min(y)), abs(max(y)))*0.7 if any(y) else 14.0,
                text=f"{value/1000:.1f} kN",
                showarrow=False,
                font=dict(size=12, color=colors['load']),
                row=1, col=1
            )
        elif load_type.lower() == "verdeelde last":
            # Verdeelde last als lijn met pijlen
            length = rest[0]
            start = pos/1000
            end = (pos + length)/1000
            
            # Teken verdeelde last lijn
            fig.add_trace(
                go.Scatter(
                    x=[start, end],
                    y=[max(abs(min(y)), abs(max(y)))*0.5 if any(y) else 10.0, max(abs(min(y)), abs(max(y)))*0.5 if any(y) else 10.0],  # Veilige offset boven de balk
                    mode='lines',
                    name=f'{value/1000:.1f} kN/m',
                    line=dict(color=colors['load'], width=3),
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # Voeg pijlen toe langs de verdeelde last
            num_arrows = min(int(length/500) + 2, 10)  # Aantal pijlen afhankelijk van lengte
            for i in range(num_arrows):
                arrow_pos = start + (end - start) * i / (num_arrows - 1)
                fig.add_trace(
                    go.Scatter(
                        x=[arrow_pos],
                        y=[max(abs(min(y)), abs(max(y)))*0.5 if any(y) else 10.0],
                        mode='markers',
                        marker=dict(
                            symbol='arrow-down',
                            size=12,
                            color=colors['load']
                        ),
                        showlegend=False
                    ),
                    row=1, col=1
                )
            
            # Voeg waarde label toe in het midden
            fig.add_annotation(
                x=(start + end) / 2,
                y=max(abs(min(y)), abs(max(y)))*0.7 if any(y) else 14.0,
                text=f"{value/1000:.1f} kN/m",
                showarrow=False,
                font=dict(size=12, color=colors['load']),
                row=1, col=1
            )
    
    # Plot dwarskracht (nu tweede)
    fig.add_trace(
        go.Scatter(
            x=x/1000, y=V/1000,
            mode='lines',
            name='Dwarskracht',
            line=dict(color=colors['shear'], width=3),
            fill='tozeroy',
            fillcolor=f'rgba(46, 204, 113, 0.2)'
        ),
        row=2, col=1
    )
    
    # Plot moment (nu derde)
    fig.add_trace(
        go.Scatter(
            x=x/1000, y=M/1e6,
            mode='lines',
            name='Moment',
            line=dict(color=colors['moment'], width=3),
            fill='tozeroy',
            fillcolor=f'rgba(231, 76, 60, 0.2)'
        ),
        row=3, col=1
    )
    
    # Plot rotatie (nu onderaan)
    fig.add_trace(
        go.Scatter(
            x=x/1000, y=theta,
            mode='lines',
            name='Rotatie',
            line=dict(color=colors['rotation'], width=3),
            fill='tozeroy',
            fillcolor=f'rgba(243, 156, 18, 0.2)'
        ),
        row=4, col=1
    )
    
    # Verbeter layout
    fig.update_layout(
        height=800,  # Groter figuur
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor='white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        font=dict(
            family="Arial, sans-serif",
            size=14
        )
    )
    
    # Update x-assen
    for i in range(1, 5):
        fig.update_xaxes(
            title_text="Positie [m]" if i == 4 else None,
            gridcolor=colors['grid'],
            zerolinecolor=colors['zero'],
            zerolinewidth=2,
            row=i, col=1
        )
    
    # Update y-assen
    fig.update_yaxes(
        gridcolor=colors['grid'],
        zerolinecolor=colors['zero'],
        zerolinewidth=2
    )
    
    # Voeg nulpunten toe op steunpunten in doorbuigingsgrafiek
    for pos, _ in supports:
        fig.add_shape(
            type="line",
            x0=pos/1000, y0=-max(abs(min(y)), abs(max(y)))*0.05 if any(y) else -5.0,
            x1=pos/1000, y1=max(abs(min(y)), abs(max(y)))*0.05 if any(y) else 5.0,
            line=dict(color=colors['support'], width=1, dash="dot"),
            row=1, col=1
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

def calculate_moment_of_inertia(profile_type, h, b, t_w, t_f=None):
    """Bereken traagheidsmoment voor verschillende profieltypes"""
    try:
        # Controleer invoer
        if h <= 0 or b <= 0 or t_w <= 0:
            st.error("❌ Profielafmetingen moeten positief zijn")
            return None
            
        if t_w >= min(h, b)/2:
            st.error("❌ Wanddikte te groot voor profiel")
            return None
            
        # Controleer of profile_type een string is
        if not isinstance(profile_type, str):
            st.error(f"❌ Ongeldig profieltype: {profile_type} (type: {type(profile_type)})")
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
            tf = t_f   # Flensdikte
            
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
        return None

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
        page_title="BeamSolved",
        page_icon="🔧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.write("✅ App is gestart")


    # App configuratie

    # Custom CSS voor moderne styling
    st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        font-weight: 500;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #2980b9;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f8f9fa;
        border-radius: 4px 4px 0 0;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3498db !important;
        color: white !important;
    }
    .stMarkdown a {
        color: #3498db;
        text-decoration: none;
    }
    .stMarkdown a:hover {
        text-decoration: underline;
    }
    .css-1y4p8pa {
        max-width: 1200px;
    }
    .stExpander {
        border: 1px solid #e6e9ef;
        border-radius: 6px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .stExpander [data-testid="stExpander"] {
        background-color: #f8f9fa;
        border-radius: 4px;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 6px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .stMetric [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2c3e50;
    }
    .stMetric [data-testid="stMetricLabel"] {
        font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("BeamSolved")
        st.markdown("#### Geavanceerde balkberekeningen voor constructeurs")
    
    # Sidebar voor invoer
    with st.sidebar:
        st.header("Invoergegevens")
        
        # Profiel tab
        with st.expander("Profiel", expanded=True):
            profile_type = st.selectbox(
                "Profieltype",
                ["Koker", "I-profiel", "Rechthoek", "Cirkel", "Standaard profiel"]
            )
            
            if profile_type == "Standaard profiel":
                profile_category = st.selectbox(
                    "Categorie",
                    ["HEA", "HEB", "IPE", "UNP", "Koker"]
                )
                profile_list = get_profile_list(profile_category)
                profile_name = st.selectbox("Profiel", profile_list)
                
                # Haal dimensies op
                height, width, wall_thickness, flange_thickness = get_profile_dimensions(profile_category, profile_name)
                
                # Toon dimensies
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Hoogte", f"{height} mm")
                    st.metric("Wanddikte", f"{wall_thickness} mm")
                with col2:
                    st.metric("Breedte", f"{width} mm")
                    if flange_thickness:
                        st.metric("Flensdikte", f"{flange_thickness} mm")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    height = st.number_input(
                        "Hoogte", 
                        value=100.0,
                        min_value=10.0,
                        step=10.0,
                        format="%.1f",
                        help="mm"
                    )
                with col2:
                    width = st.number_input(
                        "Breedte", 
                        value=50.0,
                        min_value=10.0,
                        step=10.0,
                        format="%.1f",
                        help="mm"
                    )
                
                col1, col2 = st.columns(2)
                with col1:
                    wall_thickness = st.number_input(
                        "Wanddikte", 
                        value=5.0,
                        min_value=1.0,
                        step=0.5,
                        format="%.1f",
                        help="mm"
                    )
                with col2:
                    if profile_type == "I-profiel":
                        flange_thickness = st.number_input(
                            "Flensdikte", 
                            value=8.0,
                            min_value=1.0,
                            step=0.5,
                            format="%.1f",
                            help="mm"
                        )
                    else:
                        flange_thickness = None
            
            # Materiaal eigenschappen
            E = st.number_input(
                "E-modulus", 
                value=210000.0,
                min_value=1000.0,
                step=1000.0,
                format="%.1f",
                help="N/mm²"
            )
            
            # Bereken en toon eigenschappen
            if profile_type != "Standaard profiel":
                I = calculate_moment_of_inertia(profile_type, height, width, wall_thickness, flange_thickness)
                A = calculate_A(profile_type, height, width, wall_thickness, flange_thickness)
                
                if I is not None and A is not None:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("I", f"{I:.2e} mm⁴")
                    with col2:
                        st.metric("A", f"{A:.1f} mm²")
        
        # Balk tab
        with st.expander("Balk", expanded=True):
            beam_length = st.number_input(
                "Overspanning", 
                value=3000.0,
                min_value=100.0,
                step=100.0,
                format="%.0f",
                help="mm"
            )
            
            # Steunpunten
            st.subheader("Steunpunten")
            num_supports = st.slider("Aantal steunpunten", 2, 5, 2)
            
            supports = []
            for i in range(num_supports):
                col1, col2 = st.columns(2)
                with col1:
                    if i == 0:
                        pos = 0.0
                        st.number_input(
                            f"Positie {i+1}", 
                            value=pos,
                            min_value=0.0,
                            max_value=0.0,
                            step=100.0,
                            format="%.0f",
                            help="mm",
                            key=f"support_pos_{i}",
                            disabled=True
                        )
                    elif i == num_supports - 1:
                        pos = beam_length
                        st.number_input(
                            f"Positie {i+1}", 
                            value=pos,
                            min_value=beam_length,
                            max_value=beam_length,
                            step=100.0,
                            format="%.0f",
                            help="mm",
                            key=f"support_pos_{i}",
                            disabled=True
                        )
                    else:
                        pos = st.number_input(
                            f"Positie {i+1}", 
                            value=i * beam_length / (num_supports - 1),
                            min_value=0.0,
                            max_value=beam_length,
                            step=100.0,
                            format="%.0f",
                            help="mm",
                            key=f"support_pos_{i}"
                        )
                with col2:
                    type = st.selectbox(
                        f"Type {i+1}",
                        ["Scharnier", "Rol", "Inklemming"],
                        key=f"support_type_{i}"
                    )
                supports.append((pos, type))
            
            # Belastingen
            st.subheader("Belastingen")
            num_loads = st.slider("Aantal belastingen", 1, 5, 1)
            
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
        x, V, M, theta, y, reactions = analyze_beam(beam_length, supports, loads, profile_type, height, width, wall_thickness, flange_thickness, E)
        
        # Teken balkschema
        beam_fig = plot_beam_diagram(beam_length, supports, loads)
        
        # Toon resultaten als analyse succesvol was
        if x is not None and V is not None and M is not None and theta is not None and y is not None and reactions is not None:
            # Resultaten tabs
            tab1, tab2, tab3 = st.tabs(["Grafieken", "Balkschema", "Resultaten"])
            
            with tab1:
                # Plot resultaten
                results_fig = plot_results(x, V, M, theta, y, beam_length, supports, loads)
                st.plotly_chart(results_fig, use_container_width=True)
                
                # Toon maximale waarden
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    max_V = np.max(np.abs(V))
                    st.metric("Max. dwarskracht", f"{max_V/1000:.2f} kN")
                with col2:
                    max_M = np.max(np.abs(M))
                    st.metric("Max. moment", f"{max_M/1e6:.2f} kNm")
                with col3:
                    max_theta = np.max(np.abs(theta))
                    st.metric("Max. rotatie", f"{max_theta:.6f} rad")
                with col4:
                    max_y = np.max(np.abs(y))
                    st.metric("Max. doorbuiging", f"{max_y:.2f} mm")
            
            with tab2:
                # Toon balkschema
                st.plotly_chart(beam_fig, use_container_width=True)
            
            with tab3:
                # Toon reactiekrachten
                st.subheader("Reactiekrachten")
                reaction_data = []
                for pos, support_type in supports:
                    reaction = 0
                    if pos in reactions:
                        reaction = -reactions[pos]  # Negatief omdat reacties omhoog positief zijn
                    reaction_data.append([f"{pos:.0f} mm", support_type, f"{reaction/1000:.2f} kN"])
                
                st.table({
                    "Positie": [row[0] for row in reaction_data],
                    "Type": [row[1] for row in reaction_data],
                    "Reactiekracht": [row[2] for row in reaction_data]
                })
                
                # Toon PDF export optie
                if st.button("Exporteer rapport (PDF)", use_container_width=True):
                    # Verzamel data voor rapport
                    beam_data = {
                        "profile_type": profile_type,
                        "height": height,
                        "width": width,
                        "wall_thickness": wall_thickness,
                        "flange_thickness": flange_thickness,
                        "beam_length": beam_length,
                        "supports": supports,
                        "loads": loads,
                        "reactions": reactions,
                        "max_values": {
                            "V": max_V,
                            "M": max_M,
                            "theta": max_theta,
                            "y": max_y
                        }
                    }
                    
                    # Genereer rapport
                    report_content = generate_pdf_report(beam_data, results_fig)
                    
                    # Download knop
                    now = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"beamsolve_rapport_{now}.pdf"
                    # Converteer naar base64 voor download
                    b64 = base64.b64encode(report_content).decode()
                    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">Download PDF rapport</a>'
                    st.markdown(href, unsafe_allow_html=True)
    else:
        # Toon welkomstscherm
        st.markdown("""
        ## Welkom bij BeamSolved
        
        Dit is een geavanceerde tool voor het analyseren van balken en liggers.
        
        ### Hoe te gebruiken:
        1. Configureer het profiel in de sidebar
        2. Definieer de balk en steunpunten
        3. Voeg belastingen toe
        4. Klik op 'Bereken'
        
        ### Mogelijkheden:
        - Verschillende profieltypes (Koker, I-profiel, Rechthoek, Cirkel, HEA, HEB, IPE, UNP)
        - Meerdere steunpunten (2-5)
        - Verschillende belastingtypes
        - Visualisatie van dwarskracht, moment, rotatie en doorbuiging
        - Export naar PDF rapport
        """)
        
       
if __name__ == "__main__":
    main()
