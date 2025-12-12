"""
Data Cleaner für Bewegungsdaten
Erkennt und entfernt MESSFEHLER (nicht nur Lücken!)

Problem:
- Manchmal gibt es FALSCHE Messwerte (z.B. Sprung zurück)
- Diese sind keine Lücken, sondern FEHLER
- Sie müssen ENTFERNT werden, dann wird interpoliert

Beispiel Brinkmann-1:
- Normale Daten: 12m → 45m (steigend)
- FEHLER: 45m → 31m → 24m (Athlet läuft nicht zurück!)
- Normale Daten: 49m → 60m (steigend)
- FEHLER sollte ENTFERNT werden, dann 45m → 49m interpoliert
"""

import numpy as np
from typing import List, Tuple, Dict, Any


class DataCleaner:
    """
    Bereinigt Bewegungsdaten von Messfehlern
    
    Regel: Distanz muss (fast) immer steigend sein
    - Kleine Rückschritte (<500mm) sind ok (Messrauschen)
    - Große Rückschritte (>1000mm) sind FEHLER
    """
    
    def __init__(self, max_backward_mm: float = 500, min_forward_velocity: float = 2.0):
        """
        Args:
            max_backward_mm: Maximale erlaubte Rückwärtsbewegung (Rauschen)
            min_forward_velocity: Minimale erwartete Vorwärtsgeschwindigkeit (m/s)
        """
        self.max_backward_mm = max_backward_mm
        self.min_forward_velocity = min_forward_velocity
    
    def clean_and_interpolate(self, data: List[float], sampling_rate: int = 50) -> Tuple[np.ndarray, List[Dict], List[int]]:
        """
        Bereinige Daten und interpoliere Lücken
        
        Args:
            data: Originale Messdaten (in mm)
            sampling_rate: Abtastrate in Hz
            
        Returns:
            Tuple von:
            - cleaned_data: Bereinigte und interpolierte Daten
            - gaps_info: Info über gefundene Lücken/Fehler
            - removed_indices: Indizes der entfernten Fehlerwerte
        """
        data_array = np.array(data, dtype=np.float64)
        n = len(data_array)
        
        # Schritt 1: Finde fehlerhafte Bereiche (Werte DEUTLICH unter dem bisherigen Maximum)
        valid_mask = np.ones(n, dtype=bool)
        removed_indices = []
        
        # Berechne das Maximum bis zu jedem Punkt
        max_so_far = np.maximum.accumulate(data_array)
        
        # Markiere ALLE Punkte, die mehr als max_backward_mm unter dem Maximum liegen
        # Das erfasst sowohl die Sprünge ALS AUCH die folgenden falschen Werte
        for i in range(1, n):
            if max_so_far[i-1] - data_array[i] > self.max_backward_mm:
                valid_mask[i] = False
                removed_indices.append(i)
        
        # WICHTIG: Auch Punkte nach einem Fehler prüfen, die immer noch zu niedrig sind
        # Das erfasst z.B. die 24m Werte nach dem Sprung von 45m
        for i in range(1, n):
            if valid_mask[i]:  # Nur noch nicht markierte Punkte
                # Prüfe ob dieser Wert deutlich unter dem letzten GÜLTIGEN Maximum liegt
                last_valid_max = 0
                for j in range(i-1, -1, -1):
                    if valid_mask[j]:
                        last_valid_max = max(last_valid_max, data_array[j])
                        break
                
                if last_valid_max - data_array[i] > self.max_backward_mm:
                    valid_mask[i] = False
                    removed_indices.append(i)
        
        # Erwartete Geschwindigkeit: ~6-10 m/s = 120-200 mm pro Frame
        expected_step = self.min_forward_velocity * 1000 / sampling_rate
        
        # Schritt 2: Finde zusammenhängende Fehlerbereiche
        error_regions = []
        in_error = False
        error_start = 0
        
        for i in range(n):
            if not valid_mask[i] and not in_error:
                in_error = True
                error_start = i
            elif valid_mask[i] and in_error:
                in_error = False
                error_regions.append((error_start, i))
        
        if in_error:
            error_regions.append((error_start, n))
        
        # Schritt 3: Erstelle bereinigte Daten mit Interpolation
        cleaned_data = data_array.copy()
        gaps_info = []
        
        for region_start, region_end in error_regions:
            # Finde gültige Punkte vor und nach dem Fehlerbereich
            valid_before = region_start - 1
            while valid_before >= 0 and not valid_mask[valid_before]:
                valid_before -= 1
            
            valid_after = region_end
            while valid_after < n and not valid_mask[valid_after]:
                valid_after += 1
            
            if valid_before >= 0 and valid_after < n:
                # Interpoliere zwischen gültigen Punkten
                start_value = data_array[valid_before]
                end_value = data_array[valid_after]
                
                # Berechne wie viele Punkte zwischen den gültigen Werten liegen sollten
                gap_size_mm = end_value - start_value
                expected_frames = int(gap_size_mm / expected_step)
                
                # Lineare Interpolation für den gesamten Bereich
                num_points = valid_after - valid_before - 1
                interpolated = np.linspace(start_value, end_value, num_points + 2)[1:-1]
                
                # Füge leichte Variation hinzu (Schrittmuster)
                step_freq = 4.0  # Hz
                frames_per_step = sampling_rate / step_freq
                variation = 0.01 * gap_size_mm / num_points * np.sin(
                    2 * np.pi * np.arange(num_points) / frames_per_step
                )
                interpolated += variation
                
                # Stelle sicher, dass Werte monoton steigend sind
                for j in range(1, len(interpolated)):
                    if interpolated[j] < interpolated[j-1]:
                        interpolated[j] = interpolated[j-1] + 1
                
                # Ersetze Fehlerwerte
                cleaned_data[valid_before + 1:valid_after] = interpolated
        
        # Schritt 4: Finale Glättung - entferne kleine Rückwärtsbewegungen (Messrauschen)
        # Das ist optional, macht die Kurve aber sauberer
        for i in range(1, len(cleaned_data)):
            if cleaned_data[i] < cleaned_data[i-1]:
                # Kleine Rückwärtsbewegung (Rauschen) → setze auf vorherigen Wert
                cleaned_data[i] = cleaned_data[i-1]
                
                gaps_info.append({
                    'type': 'error_region',
                    'start_idx': valid_before + 1,
                    'end_idx': valid_after,
                    'start_value': start_value,
                    'end_value': end_value,
                    'gap_size_mm': gap_size_mm,
                    'original_points': region_end - region_start,
                    'removed_points': sum(1 for i in range(region_start, region_end) if not valid_mask[i])
                })
        
        return cleaned_data, gaps_info, removed_indices
    
    def analyze_data_quality(self, data: List[float]) -> Dict[str, Any]:
        """
        Analysiere Datenqualität
        
        Returns:
            Dictionary mit Qualitätsmetriken
        """
        data_array = np.array(data, dtype=np.float64)
        
        # Berechne Differenzen
        diffs = np.diff(data_array)
        
        # Finde Vorwärts- und Rückwärtsbewegungen
        forward = diffs[diffs > 0]
        backward = diffs[diffs < 0]
        
        # Finde große Sprünge
        large_jumps = np.abs(diffs) > 5000  # >5m Sprünge
        
        return {
            'total_points': len(data),
            'forward_movements': len(forward),
            'backward_movements': len(backward),
            'avg_forward_step': np.mean(forward) if len(forward) > 0 else 0,
            'avg_backward_step': np.mean(backward) if len(backward) > 0 else 0,
            'large_jumps': np.sum(large_jumps),
            'large_jump_indices': np.where(large_jumps)[0].tolist(),
            'max_backward': np.min(diffs) if len(diffs) > 0 else 0,
            'is_mostly_increasing': len(forward) > len(backward) * 10
        }


def demonstrate_cleaning():
    """Demonstriere Data Cleaning mit Beispieldaten"""
    print("="*70)
    print("DATA CLEANER DEMONSTRATION")
    print("="*70)
    
    # Beispiel: Brinkmann-1 ähnliche Daten
    # Normal: 12000 → 45000 (steigend)
    # FEHLER: 31569, 24131 (Sprung zurück!)
    # Normal: 49000 → 80000 (steigend)
    
    data = list(range(12000, 45000, 200))  # Normal bis 45m
    data.append(31569)  # FEHLER: Sprung zurück
    data.extend([24131, 24154, 24160, 24151])  # FEHLER: falsche Werte
    data.extend(list(range(49000, 80000, 200)))  # Normal ab 49m
    
    print(f"\nOriginaldaten: {len(data)} Punkte")
    print(f"  Start: {data[0]/1000:.1f}m")
    print(f"  Ende: {data[-1]/1000:.1f}m")
    
    # Analysiere Qualität
    cleaner = DataCleaner()
    quality = cleaner.analyze_data_quality(data)
    
    print(f"\nQualitätsanalyse:")
    print(f"  Vorwärtsbewegungen: {quality['forward_movements']}")
    print(f"  Rückwärtsbewegungen: {quality['backward_movements']}")
    print(f"  Große Sprünge: {quality['large_jumps']}")
    print(f"  Max. Rückwärts: {quality['max_backward']/1000:.1f}m")
    
    # Bereinige Daten
    cleaned, gaps_info, removed = cleaner.clean_and_interpolate(data)
    
    print(f"\nBereinigung:")
    print(f"  Entfernte Punkte: {len(removed)}")
    print(f"  Bereinigte Daten: {len(cleaned)} Punkte")
    
    for i, gap in enumerate(gaps_info):
        print(f"\n  Fehlerbereich {i+1}:")
        print(f"    Start: {gap['start_value']/1000:.1f}m → Ende: {gap['end_value']/1000:.1f}m")
        print(f"    Entfernte Punkte: {gap['removed_points']}")
        print(f"    Lückengröße: {gap['gap_size_mm']/1000:.1f}m")
    
    # Prüfe Monotonie
    is_monotonic = all(cleaned[i] <= cleaned[i+1] for i in range(len(cleaned)-1))
    print(f"\nErgebnis:")
    print(f"  Monoton steigend: {'✅ Ja' if is_monotonic else '❌ Nein'}")
    print(f"  Min: {min(cleaned)/1000:.1f}m")
    print(f"  Max: {max(cleaned)/1000:.1f}m")
    
    return cleaned, gaps_info


if __name__ == "__main__":
    demonstrate_cleaning()

