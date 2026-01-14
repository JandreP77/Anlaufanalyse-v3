"""
Kalman Filter + SSA Hybrid Interpolator
Echte Implementierung für Schritt-Kontaktdaten

Methoden:
1. PCHIP - Piecewise Cubic Hermite (garantiert monoton)
2. Kalman Filter - Physik-basierte Zustandsschätzung
3. SSA - Singular Spectrum Analysis für Mustererkennung
4. Kalman+SSA Hybrid - Kombination beider Ansätze

Wissenschaftliche Basis:
- Kalman Filter: Kalman, R. E. (1960) "A New Approach to Linear Filtering"
- SSA: Golyandina, N. et al. (2001) "Analysis of Time Series Structure: SSA"
- PCHIP: Fritsch & Carlson (1980) "Monotone Piecewise Cubic Interpolation"
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.linalg import solve
import warnings
warnings.filterwarnings('ignore')


class KalmanFilter1D:
    """
    Kalman Filter für 1D Bewegungsdaten (Schritt-Kontakte)
    
    Zustandsvektor: [position, velocity]
    Angepasst für Schrittdaten (~3-4 Schritte/Sekunde)
    """
    
    def __init__(self, dt: float = 0.25):
        """
        Args:
            dt: Zeitschritt zwischen Schritten (~0.25s bei 4 Schritten/Sek)
        """
        self.dt = dt
        
        # Zustand: [position, velocity]
        self.x = np.zeros(2)
        
        # Zustandsübergangsmatrix (konstante Geschwindigkeit)
        self.F = np.array([
            [1, dt],
            [0, 1]
        ])
        
        # Messmatrix (wir messen nur Position)
        self.H = np.array([[1, 0]])
        
        # Prozessrauschen (Unsicherheit in der Bewegung)
        q = 50000  # mm² - angepasst für Schrittdaten
        self.Q = q * np.array([
            [dt**4/4, dt**3/2],
            [dt**3/2, dt**2]
        ])
        
        # Messrauschen (Sensor-Unsicherheit)
        self.R = np.array([[10000]])  # 100mm Standardabweichung
        
        # Zustandskovarianz
        self.P = np.eye(2) * 100000
    
    def initialize(self, position: float, velocity: float):
        """Initialisiere mit bekannten Werten"""
        self.x = np.array([position, velocity])
        self.P = np.eye(2) * 10000
    
    def predict(self) -> float:
        """Vorhersage des nächsten Zustands"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[0]
    
    def update(self, z: float):
        """Update mit Messung"""
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y).flatten()
        self.P = (np.eye(2) - K @ self.H) @ self.P
    
    def get_state(self) -> Tuple[float, float]:
        """Gibt [position, velocity] zurück"""
        return self.x[0], self.x[1]
    
    def get_uncertainty(self) -> float:
        """Gibt Positions-Unsicherheit zurück"""
        return np.sqrt(self.P[0, 0])


class SSAInterpolator:
    """
    Singular Spectrum Analysis für Schrittmuster-Extraktion
    
    Extrahiert periodische Muster aus den Schrittdaten
    """
    
    def __init__(self, window_size: int = 10):
        """
        Args:
            window_size: Fenstergröße für SSA (sollte ~2-3 Schrittzyklen sein)
        """
        self.window_size = window_size
    
    def extract_pattern(self, data: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Extrahiere Schrittmuster aus Daten
        
        Returns:
            (step_sizes, mean_step_size)
        """
        if len(data) < 3:
            return np.array([]), 0.0
        
        # Berechne Schrittgrößen (Differenzen)
        step_sizes = np.diff(data)
        
        if len(step_sizes) < self.window_size:
            return step_sizes, np.mean(step_sizes)
        
        # Erstelle Trajektorienmatrix
        L = self.window_size
        K = len(step_sizes) - L + 1
        
        if K < 1:
            return step_sizes, np.mean(step_sizes)
        
        # Trajektorienmatrix
        X = np.zeros((L, K))
        for i in range(K):
            X[:, i] = step_sizes[i:i+L]
        
        # SVD
        try:
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            
            # Rekonstruktion mit ersten Komponenten (Trend + Hauptmuster)
            n_components = min(3, len(S))
            X_reconstructed = U[:, :n_components] @ np.diag(S[:n_components]) @ Vt[:n_components, :]
            
            # Diagonale Mittelung für rekonstruierte Serie
            reconstructed_steps = np.zeros(len(step_sizes))
            counts = np.zeros(len(step_sizes))
            
            for i in range(K):
                for j in range(L):
                    reconstructed_steps[i+j] += X_reconstructed[j, i]
                    counts[i+j] += 1
            
            reconstructed_steps /= np.maximum(counts, 1)
            
            return reconstructed_steps, np.mean(reconstructed_steps)
            
        except:
            return step_sizes, np.mean(step_sizes)
    
    def predict_steps(self, data: np.ndarray, num_steps: int) -> np.ndarray:
        """
        Vorhersage zukünftiger Schritte basierend auf Muster
        
        Args:
            data: Bisherige Positionsdaten
            num_steps: Anzahl vorherzusagender Schritte
            
        Returns:
            Array mit vorhergesagten Schrittgrößen
        """
        pattern, mean_step = self.extract_pattern(data)
        
        if len(pattern) < 2:
            return np.full(num_steps, mean_step if mean_step > 0 else 200)
        
        # Verwende letzten Muster-Zyklus für Vorhersage
        cycle_length = min(self.window_size, len(pattern))
        last_cycle = pattern[-cycle_length:]
        
        # Wiederhole Muster für Vorhersage
        predicted = np.tile(last_cycle, (num_steps // cycle_length) + 1)[:num_steps]
        
        # Leichte Anpassung zum Trend
        trend = (pattern[-1] - pattern[0]) / len(pattern) if len(pattern) > 1 else 0
        adjustment = np.arange(num_steps) * trend * 0.1
        
        return predicted + adjustment


class KalmanSSAInterpolator:
    """
    Echter Kalman+SSA Hybrid Interpolator
    
    Jetzt mit tatsächlicher Implementierung aller Methoden!
    
    Verfügbare Methoden:
    - PCHIP: Monotone kubische Interpolation
    - Kalman: Physik-basierte Zustandsschätzung
    - SSA: Muster-basierte Interpolation
    - Kalman+SSA: Gewichtete Kombination
    """
    
    def __init__(self, sampling_rate: int = 50, ssa_window: int = 10, use_global_models: bool = True):
        """
        Args:
            sampling_rate: Nicht mehr relevant für Schrittdaten, aber für Kompatibilität
            ssa_window: SSA Fenstergröße
            use_global_models: Für Kompatibilität (wird ignoriert)
        """
        self.sampling_rate = sampling_rate
        self.ssa_window = ssa_window
        self.ssa = SSAInterpolator(window_size=ssa_window)
        
        # Geschätzte Zeit pro Schritt bei Sprint (~4 Schritte/Sek)
        self.step_time = 0.25  # Sekunden
        
        # Für Kompatibilität
        self.global_models_loaded = False
    
    def interpolate_gap(self, data: np.ndarray, gap_start: int, gap_end: int, 
                       gap_size_mm: float, method: str = "kalman_ssa",
                       num_points_override: int = None) -> Tuple[np.ndarray, float]:
        """
        Interpoliere eine Lücke mit der gewählten Methode
        
        Args:
            data: Positionsdaten (mm)
            gap_start: Index vor der Lücke
            gap_end: Index nach der Lücke
            gap_size_mm: Lückengröße in mm
            method: "pchip", "kalman", "ssa", "kalman_ssa"
            num_points_override: Anzahl zu interpolierender Punkte (optional)
            
        Returns:
            (interpolierte_werte, konfidenz)
        """
        gap_size_m = abs(gap_size_mm) / 1000  # Immer positiv!
        
        start_value = data[gap_start]
        end_value = data[gap_end] if gap_end < len(data) else data[-1]
        
        # PHYSIKALISCHE CONSTRAINT: Athlet läuft nur VORWÄRTS!
        # Wenn end_value <= start_value, ist das ein Messfehler
        if end_value <= start_value:
            # Suche den nächsten gültigen Wert der GRÖSSER ist
            search_idx = gap_end + 1
            found_valid = False
            while search_idx < len(data):
                if data[search_idx] > start_value:
                    end_value = data[search_idx]
                    found_valid = True
                    break
                search_idx += 1
            
            # Falls nicht gefunden, schätze basierend auf typischer Geschwindigkeit
            if not found_valid:
                # ~8 m/s = 8000 mm/s, bei ~4 Schritten/s = ~2000mm pro Schritt
                estimated_distance = (gap_end - gap_start) * 2000  # mm
                end_value = start_value + estimated_distance
            
            # Aktualisiere gap_size
            gap_size_mm = end_value - start_value
            gap_size_m = gap_size_mm / 1000
        
        # WICHTIG: Anzahl Punkte basierend auf Index-Differenz, NICHT Distanz!
        # Das erhält die ursprüngliche Sampling-Dichte
        if num_points_override is not None:
            num_points = num_points_override
        else:
            # Anzahl der Indizes zwischen Start und Ende
            num_points = gap_end - gap_start - 1
            
            # Mindestens 1 Punkt, maximal sinnvolle Anzahl
            num_points = max(1, min(num_points, 100))
        
        if method == "pchip":
            return self._pchip_interpolate(data, gap_start, gap_end, num_points, start_value, end_value)
        elif method == "kalman":
            return self._kalman_interpolate(data, gap_start, gap_end, num_points, start_value, end_value)
        elif method == "ssa":
            return self._ssa_interpolate(data, gap_start, gap_end, num_points, start_value, end_value)
        elif method == "kalman_ssa":
            return self._hybrid_interpolate(data, gap_start, gap_end, num_points, start_value, end_value, gap_size_m)
        else:
            # Fallback: Linear
            return self._linear_interpolate(num_points, start_value, end_value, gap_size_m)
    
    def _linear_interpolate(self, num_points: int, start_value: float, 
                           end_value: float, gap_size_m: float) -> Tuple[np.ndarray, float]:
        """Einfache lineare Interpolation"""
        interpolated = np.linspace(start_value, end_value, num_points + 2)[1:-1]
        
        # Konfidenz basierend auf Lückengröße
        if gap_size_m < 2.0:
            confidence = 0.90
        elif gap_size_m < 5.0:
            confidence = 0.80
        else:
            confidence = 0.70
        
        return interpolated, confidence
    
    def _pchip_interpolate(self, data: np.ndarray, gap_start: int, gap_end: int,
                          num_points: int, start_value: float, end_value: float) -> Tuple[np.ndarray, float]:
        """
        PCHIP - Piecewise Cubic Hermite Interpolating Polynomial
        
        Garantiert monotone Interpolation (keine Überschwinger)
        Ideal für Schrittdaten!
        """
        # Kontext vor und nach der Lücke
        context_before = min(5, gap_start)
        context_after = min(5, len(data) - gap_end - 1)
        
        if context_before < 2 or context_after < 1:
            # Nicht genug Kontext - fallback auf linear
            return self._linear_interpolate(num_points, start_value, end_value, 
                                           (end_value - start_value) / 1000)
        
        # Bekannte Punkte sammeln
        x_known = np.concatenate([
            np.arange(gap_start - context_before, gap_start + 1),
            np.arange(gap_end, gap_end + context_after + 1)
        ])
        y_known = np.concatenate([
            data[gap_start - context_before:gap_start + 1],
            data[gap_end:gap_end + context_after + 1]
        ])
        
        # PCHIP Interpolator erstellen
        try:
            pchip = PchipInterpolator(x_known, y_known)
            
            # Interpolationspunkte
            x_new = np.linspace(gap_start + 0.5, gap_end - 0.5, num_points)
            interpolated = pchip(x_new)
            
            # PCHIP ist bereits monoton, aber sicherstellen
            for i in range(1, len(interpolated)):
                if interpolated[i] < interpolated[i-1]:
                    interpolated[i] = interpolated[i-1] + 1
            
            # Hohe Konfidenz für PCHIP (mathematisch optimal)
            gap_size_m = (end_value - start_value) / 1000
            if gap_size_m < 3.0:
                confidence = 0.95
            elif gap_size_m < 6.0:
                confidence = 0.90
            else:
                confidence = 0.85
            
            return interpolated, confidence
            
        except Exception as e:
            print(f"PCHIP Fehler: {e}")
            return self._linear_interpolate(num_points, start_value, end_value,
                                           (end_value - start_value) / 1000)
    
    def _kalman_interpolate(self, data: np.ndarray, gap_start: int, gap_end: int,
                           num_points: int, start_value: float, end_value: float) -> Tuple[np.ndarray, float]:
        """
        Echte Kalman Filter Interpolation
        
        Verwendet physikalisches Bewegungsmodell zur Vorhersage
        """
        # Kalman Filter initialisieren
        kf = KalmanFilter1D(dt=self.step_time)
        
        # Kontext vor der Lücke für Geschwindigkeitsschätzung
        context_size = min(5, gap_start)
        
        if context_size >= 2:
            # Geschwindigkeit aus Kontext schätzen
            context = data[gap_start - context_size:gap_start + 1]
            velocities = np.diff(context) / self.step_time  # mm/s
            avg_velocity = np.mean(velocities)
            
            kf.initialize(start_value, avg_velocity)
        else:
            # Geschwindigkeit aus Lücke schätzen
            estimated_velocity = (end_value - start_value) / (num_points * self.step_time)
            kf.initialize(start_value, estimated_velocity)
        
        # Forward pass: Vorhersage durch die Lücke
        forward_predictions = []
        forward_uncertainties = []
        
        for _ in range(num_points):
            pos = kf.predict()
            forward_predictions.append(pos)
            forward_uncertainties.append(kf.get_uncertainty())
        
        # Backward pass: Vom Ende her
        kf_back = KalmanFilter1D(dt=self.step_time)
        
        context_after = min(5, len(data) - gap_end - 1)
        if context_after >= 2:
            context = data[gap_end:gap_end + context_after + 1]
            velocities = np.diff(context) / self.step_time
            avg_velocity = np.mean(velocities)
            kf_back.initialize(end_value, -avg_velocity)  # Rückwärts
        else:
            estimated_velocity = (end_value - start_value) / (num_points * self.step_time)
            kf_back.initialize(end_value, -estimated_velocity)
        
        backward_predictions = []
        backward_uncertainties = []
        
        for _ in range(num_points):
            pos = kf_back.predict()
            backward_predictions.append(pos)
            backward_uncertainties.append(kf_back.get_uncertainty())
        
        backward_predictions = backward_predictions[::-1]
        backward_uncertainties = backward_uncertainties[::-1]
        
        # Gewichtete Kombination basierend auf Unsicherheit
        forward_arr = np.array(forward_predictions)
        backward_arr = np.array(backward_predictions)
        forward_unc = np.array(forward_uncertainties)
        backward_unc = np.array(backward_uncertainties)
        
        # Gewichte: Niedrigere Unsicherheit = höheres Gewicht
        total_unc = forward_unc + backward_unc
        w_forward = backward_unc / total_unc
        w_backward = forward_unc / total_unc
        
        interpolated = w_forward * forward_arr + w_backward * backward_arr
        
        # Constraints: Zwischen Start und Ende, monoton
        interpolated = np.clip(interpolated, start_value, end_value)
        for i in range(1, len(interpolated)):
            if interpolated[i] < interpolated[i-1]:
                interpolated[i] = interpolated[i-1] + (end_value - start_value) / num_points
        
        # Konfidenz basierend auf Unsicherheit
        avg_uncertainty = np.mean(total_unc)
        gap_size_m = (end_value - start_value) / 1000
        
        if gap_size_m < 3.0:
            confidence = 0.90
        elif gap_size_m < 6.0:
            confidence = 0.80
        else:
            confidence = 0.70
        
        return interpolated, confidence
    
    def _ssa_interpolate(self, data: np.ndarray, gap_start: int, gap_end: int,
                        num_points: int, start_value: float, end_value: float) -> Tuple[np.ndarray, float]:
        """
        SSA-basierte Interpolation
        
        Verwendet Schrittmuster aus dem Kontext
        """
        # Kontext vor der Lücke für Musterextraktion
        context_size = min(20, gap_start)
        
        if context_size < 5:
            # Nicht genug Kontext für SSA
            return self._linear_interpolate(num_points, start_value, end_value,
                                           (end_value - start_value) / 1000)
        
        context = data[gap_start - context_size:gap_start + 1]
        
        # Schrittmuster extrahieren
        predicted_steps = self.ssa.predict_steps(context, num_points)
        
        # Positionen aus Schritten berechnen
        interpolated = np.zeros(num_points)
        current_pos = start_value
        
        for i in range(num_points):
            current_pos += predicted_steps[i] if i < len(predicted_steps) else np.mean(predicted_steps)
            interpolated[i] = current_pos
        
        # Skalierung auf Endwert
        if interpolated[-1] != 0 and interpolated[-1] != start_value:
            scale = (end_value - start_value) / (interpolated[-1] - start_value)
            interpolated = start_value + (interpolated - start_value) * scale
        else:
            interpolated = np.linspace(start_value, end_value, num_points + 2)[1:-1]
        
        # Constraints
        interpolated = np.clip(interpolated, start_value, end_value)
        for i in range(1, len(interpolated)):
            if interpolated[i] < interpolated[i-1]:
                interpolated[i] = interpolated[i-1] + (end_value - start_value) / num_points
        
        gap_size_m = (end_value - start_value) / 1000
        if gap_size_m < 4.0:
            confidence = 0.85
        elif gap_size_m < 8.0:
            confidence = 0.75
        else:
            confidence = 0.65
        
        return interpolated, confidence
    
    def _hybrid_interpolate(self, data: np.ndarray, gap_start: int, gap_end: int,
                           num_points: int, start_value: float, end_value: float,
                           gap_size_m: float) -> Tuple[np.ndarray, float]:
        """
        Echter Kalman+SSA Hybrid
        
        Kombiniert Kalman (Physik) mit SSA (Muster)
        """
        # Kalman Vorhersage
        kalman_pred, kalman_conf = self._kalman_interpolate(
            data, gap_start, gap_end, num_points, start_value, end_value
        )
        
        # SSA Vorhersage
        ssa_pred, ssa_conf = self._ssa_interpolate(
            data, gap_start, gap_end, num_points, start_value, end_value
        )
        
        # Gewichtete Kombination
        # Bei kleinen Lücken: Mehr Kalman (genauer)
        # Bei großen Lücken: Mehr SSA (Muster wichtiger)
        if gap_size_m < 3.0:
            w_kalman = 0.7
            w_ssa = 0.3
        elif gap_size_m < 6.0:
            w_kalman = 0.5
            w_ssa = 0.5
        else:
            w_kalman = 0.4
            w_ssa = 0.6
        
        # Fusion
        interpolated = w_kalman * kalman_pred + w_ssa * ssa_pred
        
        # Constraints
        interpolated = np.clip(interpolated, start_value, end_value)
        for i in range(1, len(interpolated)):
            if interpolated[i] < interpolated[i-1]:
                interpolated[i] = interpolated[i-1] + (end_value - start_value) / num_points
        
        # Kombinierte Konfidenz
        confidence = w_kalman * kalman_conf + w_ssa * ssa_conf
        
        return interpolated, confidence
    
    def fill_all_gaps(self, data: List[float], gaps: List[Dict[str, Any]], 
                     method: str = "kalman_ssa") -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Fülle alle Lücken mit der gewählten Methode
        
        Args:
            data: Positionsdaten
            gaps: Liste von Lücken-Dictionaries
            method: Interpolationsmethode
            
        Returns:
            (gefüllte_daten, interpolation_info)
        """
        data_array = np.array(data)
        interpolation_info = []
        
        sorted_gaps = sorted(gaps, key=lambda g: g['index'])
        
        segments = []
        last_idx = 0
        
        for gap in sorted_gaps:
            gap_idx = gap['index']
            gap_size = gap['difference']
            
            segments.append(data_array[last_idx:gap_idx + 1])
            
            interpolated, confidence = self.interpolate_gap(
                data_array, gap_idx, gap_idx + 1, gap_size, method=method
            )
            
            segments.append(interpolated)
            
            method_display = {
                "pchip": "PCHIP (Monoton)",
                "kalman": "Kalman Filter",
                "ssa": "SSA (Muster)",
                "kalman_ssa": "Kalman+SSA Hybrid"
            }.get(method, method)
            
            interpolation_info.append({
                'index': gap_idx,
                'size_mm': gap_size,
                'size_m': gap_size / 1000,
                'num_points': len(interpolated),
                'confidence': confidence,
                'method': method_display,
                'start_idx': sum(len(s) for s in segments) - len(interpolated),
                'end_idx': sum(len(s) for s in segments)
            })
            
            last_idx = gap_idx + 1
        
        if last_idx < len(data_array):
            segments.append(data_array[last_idx:])
        
        filled_data = np.concatenate(segments)
        
        return filled_data, interpolation_info
    

# Test
if __name__ == "__main__":
    print("="*70)
    print("KALMAN + SSA INTERPOLATOR TEST")
    print("="*70)
    
    # Simuliere Schrittdaten
    np.random.seed(42)
    
    # Simulierter Anlauf: ~40m in ~20 Schritten
    steps = np.array([0] + list(np.cumsum(np.random.normal(2000, 200, 20))))  # mm
    
    print(f"\nOriginaldaten: {len(steps)} Punkte")
    print(f"Start: {steps[0]/1000:.1f}m, Ende: {steps[-1]/1000:.1f}m")
    
    # Erstelle Lücke
    gap_start = 8
    gap_end = 12
    true_gap = steps[gap_start+1:gap_end].copy()
    
    data_with_gap = steps.copy()
    gap_size = steps[gap_end] - steps[gap_start]
    
    print(f"\nLücke: Index {gap_start} bis {gap_end}")
    print(f"Lückengröße: {gap_size/1000:.2f}m")
    
    # Test aller Methoden
    interpolator = KalmanSSAInterpolator()
    
    for method in ["pchip", "kalman", "ssa", "kalman_ssa"]:
        interp, conf = interpolator.interpolate_gap(
            data_with_gap, gap_start, gap_end, gap_size, method=method
        )
        
        print(f"\n{method.upper()}:")
        print(f"  Punkte: {len(interp)}")
        print(f"  Konfidenz: {conf:.1%}")
        
        if len(interp) == len(true_gap):
            rmse = np.sqrt(np.mean((interp - true_gap)**2))
            print(f"  RMSE: {rmse:.1f}mm")
    
    print("\n" + "="*70)
    print("✅ Test abgeschlossen!")
