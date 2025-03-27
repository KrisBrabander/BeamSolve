import numpy as np

# Alternatief voor scipy functies
def custom_cumtrapz(y, x, initial=0):
    """Eigen implementatie van cumtrapz"""
    result = np.zeros_like(y)
    result[0] = initial
    for i in range(1, len(y)):
        result[i] = result[i-1] + 0.5 * (y[i] + y[i-1]) * (x[i] - x[i-1])
    return result

def custom_solve_banded(l_and_u, ab, b):
    """Vereenvoudigde implementatie voor bandmatrix"""
    # Converteer naar volledige matrix en gebruik numpy.linalg.solve
    n = len(b)
    A = np.zeros((n, n))
    
    for i in range(n):
        for j in range(max(0, i-l_and_u[0]), min(n, i+l_and_u[1]+1)):
            A[i, j] = ab[l_and_u[0] + j - i, i]
    
    return np.linalg.solve(A, b)

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
        """Drie-momentenvergelijking voor doorlopende liggers"""
        n = len(self.supports)
        positions = [s[0] for s in self.supports]
        L = [positions[i+1] - positions[i] for i in range(n-1)]
        
        # Bouw matrix voor driemomentenvergelijking
        A = np.zeros((n, n))
        rhs = np.zeros(n)

        # Vul matrix voor interne steunpunten
        for i in range(1, n-1):
            A[i, i-1] = L[i-1]
            A[i, i] = 2*(L[i-1] + L[i])
            A[i, i+1] = L[i]
            rhs[i] = -6*(self._calc_free_moment(i-1) + self._calc_free_moment(i))

        # Randvoorwaarden
        A[0, 0] = 1  # M0 = 0
        A[-1, -1] = 1  # Mn = 0
        
        try:
            # Los matrix op
            moments = np.linalg.solve(A, rhs)
        except:
            # Fallback als solve faalt
            # Vereenvoudigde methode: verdeel belasting over steunpunten
            self.reactions = {}
            total_load = 0
            for load in self.loads:
                pos, value, load_type, *rest = load
                if load_type.lower() == "puntlast":
                    total_load += value
                elif load_type.lower() == "verdeelde last":
                    length = rest[0]
                    total_load += value * length
            
            # Verdeel belasting over steunpunten
            weights = []
            for i in range(n):
                if i == 0 or i == n-1:
                    weights.append(0.5)  # Randsteunpunten krijgen minder
                else:
                    weights.append(1.0)  # Tussensteunpunten krijgen meer
            
            total_weight = sum(weights)
            for i, (pos, _) in enumerate(self.supports):
                self.reactions[pos] = -total_load * weights[i] / total_weight
            
            return

        # Bereken reacties uit momenten
        self.reactions = {}
        for i in range(n):
            R = 0
            if i > 0:
                R += (moments[i-1] + moments[i])/(2*L[i-1])
            if i < n-1:
                R += (moments[i] + moments[i+1])/(2*L[i])
            R += self._point_load_contribution(positions[i])
            self.reactions[positions[i]] = R
            
            # Voeg inklemming momenten toe indien nodig
            if self.supports[i][1].lower() == "inklemming":
                self.reactions[f"M_{positions[i]}"] = moments[i]

    def _calc_free_moment(self, span_idx):
        """Bereken vrije veldmoment voor overspanning"""
        x1 = self.supports[span_idx][0]
        x2 = self.supports[span_idx+1][0]
        M = 0

        for load in self.loads:
            p, val, ltype, *rest = load
            if ltype.lower() == "puntlast" and x1 <= p <= x2:
                a = p - x1
                L = x2 - x1
                M += val * a * (L**2 - a**2) / (6*L)
            elif ltype.lower() == "verdeelde last":
                start = max(x1, p)
                end = min(x2, p + rest[0])
                if start < end:
                    a = start - x1
                    b = end - x1
                    L = x2 - x1
                    M += val * (b**3 - a**3)/(6*L) - val * (b**4 - a**4)/(24*L**2)
        return M

    def _point_load_contribution(self, pos):
        """Bijdrage puntlasten op steunpunt"""
        return sum(load[1] for load in self.loads 
                 if load[2].lower() == "puntlast" and abs(load[0] - pos) < 1e-6)

    def _calculate_internal_forces(self):
        """Bereken dwarskrachten en momentenlijn"""
        self.V = np.zeros_like(self.x)
        self.M = np.zeros_like(self.x)

        # Verzamel steunpunt posities voor controle
        support_positions = [s[0] for s in self.supports]

        # Reactiekrachten
        for pos, R in self.reactions.items():
            if not pos.startswith("M_"):  # Alleen krachten, geen momenten
                idx = np.searchsorted(self.x, float(pos))
                if idx < len(self.x):
                    self.V[idx:] += R
                    self.M[idx:] += R * (self.x[idx:] - float(pos))
        
        # Reactiemomenten
        for key, val in self.reactions.items():
            if key.startswith("M_"):
                pos = float(key.split("_")[1])
                idx = np.searchsorted(self.x, pos)
                if idx < len(self.x):
                    # Moment heeft geen effect op dwarskracht, alleen op moment
                    self.M[idx:] += val

        # Uitwendige belastingen
        for load in self.loads:
            pos, val, ltype, *rest = load
            
            # Controleer of de last op een steunpunt valt
            on_support = any(abs(pos - sp) < 1e-6 for sp in support_positions)
            
            if ltype.lower() == "puntlast":
                # Als de puntlast op een steunpunt valt, wordt deze direct opgenomen
                if not on_support:
                    self._apply_point_load(pos, val)
            elif ltype.lower() == "verdeelde last":
                self._apply_distributed_load(pos, val, rest[0])
            elif ltype.lower() == "moment":
                self._apply_moment(pos, val)

    def _apply_point_load(self, pos, P):
        idx = np.searchsorted(self.x, pos)
        self.V[idx:] -= P
        self.M[idx:] -= P * (self.x[idx:] - pos)

    def _apply_distributed_load(self, start, q, length):
        end = start + length
        start_idx = np.searchsorted(self.x, start)
        end_idx = np.searchsorted(self.x, end)

        # Onder belasting
        for i in range(start_idx, min(end_idx, len(self.x))):
            dx = self.x[i] - start
            self.V[i] -= q * dx
            self.M[i] -= 0.5 * q * dx**2

        # Na belasting
        if end_idx < len(self.x):
            self.V[end_idx:] -= q * length
            self.M[end_idx:] -= q * length * (self.x[end_idx:] - start - 0.5*length)

    def _apply_moment(self, pos, M):
        idx = np.searchsorted(self.x, pos)
        if idx < len(self.x):
            self.M[idx:] -= M

    def _calculate_deflection(self):
        """Bereken doorbuiging via dubbele integratie"""
        # Eerste integratie: hoekverdraaiing
        self.theta = custom_cumtrapz(self.M / self.EI, self.x, initial=0)

        # Tweede integratie: doorbuiging
        y = custom_cumtrapz(self.theta, self.x, initial=0)

        # Pas randvoorwaarden aan
        A = []
        b = []
        for pos, type in self.supports:
            idx = np.abs(self.x - pos).argmin()
            if type.lower() == "inklemming":
                A.append([self.x[idx], 1])  # y = 0
                A.append([1, 0])           # dy/dx = 0
                b.extend([-y[idx], -self.theta[idx]])
            else:
                A.append([self.x[idx], 1])
                b.append(-y[idx])

        # Los kleinste kwadraten op
        A = np.array(A)
        b = np.array(b)
        
        try:
            C, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            self.y = y + C[0]*self.x + C[1]
        except:
            # Fallback als lstsq faalt
            self.y = y
            # Verschuif zodat eerste steunpunt op 0 ligt
            idx = np.abs(self.x - self.supports[0][0]).argmin()
            self.y -= self.y[idx]
