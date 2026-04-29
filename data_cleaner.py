"""
Data Cleaner für Laveg-Laser-Bewegungsdaten
Erkennt und entfernt MESSFEHLER in Anlauf-Positionsdaten.

Erkennt zwei Fehlertypen:
1. Rückwärtssprünge: Position fällt >500mm unter bisheriges Maximum
2. Vorwärtssprünge:  Position springt >1000mm in einem Frame (Laveg-Threshold)

Nach Erkennung werden fehlerhafte Bereiche markiert und die Lücken-Info
an den Aufrufer zurückgegeben, der die eigentliche Interpolation durchführt.
"""

import numpy as np
from typing import List, Tuple, Dict, Any


GAP_THRESHOLD_MM = 1000.0
MAX_BACKWARD_MM = 500.0
MIN_FORWARD_VELOCITY_MS = 2.0


class DataCleaner:
    """
    Bereinigt Laveg-Laser-Bewegungsdaten von Messfehlern.

    Physikalische Annahme: Athlet bewegt sich monoton vorwärts.
    Typische Schrittgröße bei 50Hz: 20-40mm pro Frame (~6-10 m/s).
    """

    def __init__(self, max_backward_mm: float = MAX_BACKWARD_MM,
                 min_forward_velocity: float = MIN_FORWARD_VELOCITY_MS,
                 gap_threshold_mm: float = GAP_THRESHOLD_MM):
        self.max_backward_mm = max_backward_mm
        self.min_forward_velocity = min_forward_velocity
        self.gap_threshold_mm = gap_threshold_mm

    def clean_and_interpolate(self, data: List[float], sampling_rate: int = 50) -> Tuple[np.ndarray, List[Dict], List[int]]:
        """
        Bereinige Daten: erkenne Rückwärts- UND Vorwärtssprünge, markiere
        fehlerhafte Bereiche, und liefere lineare Basis-Interpolation.

        Die eigentliche wissenschaftliche Interpolation (PCHIP, Kalman, SSA)
        wird vom KalmanSSAInterpolator auf Basis der gaps_info durchgeführt.

        Returns:
            (cleaned_data, gaps_info, removed_indices)
        """
        data_array = np.array(data, dtype=np.float64)
        n = len(data_array)

        expected_step = self.min_forward_velocity * 1000 / sampling_rate

        valid_mask = np.ones(n, dtype=bool)
        removed_indices = []

        # --- Schritt 1a: Rückwärtssprünge erkennen ---
        # Punkte die >max_backward_mm unter dem bisherigen Maximum liegen
        max_so_far = np.maximum.accumulate(data_array)
        for i in range(1, n):
            if max_so_far[i - 1] - data_array[i] > self.max_backward_mm:
                valid_mask[i] = False
                removed_indices.append(i)

        # Zweiter Pass: Punkte die nach einem Fehler immer noch zu niedrig sind
        last_valid_max = data_array[0] if valid_mask[0] else 0
        for i in range(1, n):
            if valid_mask[i]:
                if last_valid_max - data_array[i] > self.max_backward_mm:
                    valid_mask[i] = False
                    removed_indices.append(i)
                else:
                    last_valid_max = max(last_valid_max, data_array[i])

        # --- Schritt 1b: Vorwärtssprünge erkennen ---
        # Sprünge >gap_threshold_mm in einem Frame sind Messausfälle
        # (Laveg hat Kontakte übersprungen)
        for i in range(1, n):
            if not valid_mask[i] or not valid_mask[i - 1]:
                continue
            forward_jump = data_array[i] - data_array[i - 1]
            if forward_jump > self.gap_threshold_mm:
                # Markiere den Sprungpunkt und alle folgenden Punkte
                # bis die Position wieder "erreichbar" wäre
                # (basierend auf erwarteter Geschwindigkeit seit letztem gültigen Punkt)
                last_valid_idx = i - 1
                last_valid_val = data_array[last_valid_idx]
                for j in range(i, n):
                    if not valid_mask[j]:
                        continue
                    frames_elapsed = j - last_valid_idx
                    max_plausible_pos = last_valid_val + frames_elapsed * expected_step * 5
                    if data_array[j] > max_plausible_pos:
                        valid_mask[j] = False
                        removed_indices.append(j)
                    else:
                        break

        removed_indices = sorted(set(removed_indices))

        # --- Schritt 2: Zusammenhängende Fehlerbereiche finden ---
        # Benachbarte Regionen mit <3 gültigen Frames dazwischen zusammenführen
        raw_regions = []
        in_error = False
        error_start = 0

        for i in range(n):
            if not valid_mask[i] and not in_error:
                in_error = True
                error_start = i
            elif valid_mask[i] and in_error:
                in_error = False
                raw_regions.append((error_start, i))

        if in_error:
            raw_regions.append((error_start, n))

        error_regions = []
        for region in raw_regions:
            if error_regions and region[0] - error_regions[-1][1] < 3:
                error_regions[-1] = (error_regions[-1][0], region[1])
            else:
                error_regions.append(region)

        # --- Schritt 3: Basis-Interpolation und Gap-Info ---
        cleaned_data = data_array.copy()
        gaps_info = []

        for region_start, region_end in error_regions:
            valid_before = region_start - 1
            while valid_before >= 0 and not valid_mask[valid_before]:
                valid_before -= 1

            valid_after = region_end
            while valid_after < n and not valid_mask[valid_after]:
                valid_after += 1

            if valid_before < 0:
                continue

            start_value = data_array[valid_before]

            if valid_after >= n:
                # Fehlerbereich reicht bis zum Datenende.
                # Schätze Endwert basierend auf erwarteter Geschwindigkeit.
                num_frames = n - valid_before
                end_value = start_value + num_frames * expected_step
                valid_after = n
            else:
                end_value = data_array[valid_after]

            if end_value <= start_value:
                search_idx = (valid_after + 1) if valid_after < n else valid_after
                found_valid_end = False
                while search_idx < n:
                    if valid_mask[search_idx] and data_array[search_idx] > start_value:
                        valid_after = search_idx
                        end_value = data_array[valid_after]
                        found_valid_end = True
                        break
                    search_idx += 1

                if not found_valid_end:
                    num_frames = min(region_end, n) - valid_before
                    end_value = start_value + num_frames * expected_step

            gap_size_mm = end_value - start_value
            if gap_size_mm <= 0:
                gap_size_mm = (min(valid_after, n) - valid_before) * expected_step

            actual_end = min(valid_after, n)
            num_points = actual_end - valid_before - 1
            if num_points <= 0:
                continue

            interpolated = np.linspace(start_value, end_value, num_points + 2)[1:-1]

            for j in range(1, len(interpolated)):
                if interpolated[j] <= interpolated[j - 1]:
                    interpolated[j] = interpolated[j - 1] + 1

            cleaned_data[valid_before + 1:actual_end] = interpolated

            gaps_info.append({
                'type': 'error_region',
                'start_idx': valid_before + 1,
                'end_idx': actual_end,
                'start_value': start_value,
                'end_value': end_value,
                'gap_size_mm': gap_size_mm,
                'original_points': region_end - region_start,
                'removed_points': sum(1 for idx in range(region_start, min(region_end, n))
                                      if not valid_mask[idx])
            })

        # --- Schritt 4: Messrauschen glätten ---
        for i in range(1, len(cleaned_data)):
            if cleaned_data[i] < cleaned_data[i - 1]:
                cleaned_data[i] = cleaned_data[i - 1]

        # gaps_info-Werte an geglättete Daten anpassen
        for gap in gaps_info:
            si, ei = gap['start_idx'], gap['end_idx']
            if si > 0:
                gap['start_value'] = float(cleaned_data[si - 1])
            if ei < len(cleaned_data):
                gap['end_value'] = float(cleaned_data[ei])
            gap['gap_size_mm'] = max(gap['end_value'] - gap['start_value'], 1.0)

        return cleaned_data, gaps_info, removed_indices
    
    def analyze_data_quality(self, data: List[float]) -> Dict[str, Any]:
        """Analysiere Datenqualität vor der Bereinigung."""
        data_array = np.array(data, dtype=np.float64)
        diffs = np.diff(data_array)

        forward = diffs[diffs > 0]
        backward = diffs[diffs < 0]

        backward_jumps = np.abs(diffs[diffs < -self.max_backward_mm])
        forward_jumps = diffs[diffs > self.gap_threshold_mm]
        all_large = (np.abs(diffs) > self.gap_threshold_mm)

        return {
            'total_points': len(data),
            'forward_movements': len(forward),
            'backward_movements': len(backward),
            'avg_forward_step': float(np.mean(forward)) if len(forward) > 0 else 0,
            'avg_backward_step': float(np.mean(backward)) if len(backward) > 0 else 0,
            'large_jumps': int(np.sum(all_large)),
            'large_jump_indices': np.where(all_large)[0].tolist(),
            'forward_jump_count': len(forward_jumps),
            'backward_jump_count': len(backward_jumps),
            'max_backward': float(np.min(diffs)) if len(diffs) > 0 else 0,
            'max_forward_jump': float(np.max(diffs)) if len(diffs) > 0 else 0,
            'is_mostly_increasing': len(forward) > len(backward) * 10
        }


if __name__ == "__main__":
    cleaner = DataCleaner()

    # Test: Rückwärtssprung
    data = list(range(12000, 45000, 200))
    data.append(31569)
    data.extend([24131, 24154, 24160, 24151])
    data.extend(list(range(49000, 80000, 200)))

    cleaned, gaps, removed = cleaner.clean_and_interpolate(data)
    print(f"Backward-Jump Test: {len(removed)} entfernt, {len(gaps)} Gaps")

    # Test: Vorwärtssprung (wie in echten Laveg-Daten)
    data2 = list(range(12000, 52000, 200))
    data2.append(68800)  # Forward-Jump: +16800mm
    data2.extend(list(range(68850, 70000, 50)))

    cleaned2, gaps2, removed2 = cleaner.clean_and_interpolate(data2)
    print(f"Forward-Jump Test: {len(removed2)} entfernt, {len(gaps2)} Gaps")
    for g in gaps2:
        print(f"  Gap: {g['start_value']:.0f} -> {g['end_value']:.0f}mm, "
              f"{g['removed_points']} Punkte entfernt")

