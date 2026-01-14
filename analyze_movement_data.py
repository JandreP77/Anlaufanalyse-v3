import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
from pyts.decomposition import SingularSpectrumAnalysis  # SSA für Interpolation
import csv

class MovementDataAnalyzer:
    def __init__(self, folders: List[str]):
        """
        Initialize the analyzer with folders containing .dat files
        
        Args:
            folders (List[str]): List of folder paths containing the data files
        """
        self.folders = folders
        self.sampling_rate = None  # Will be determined dynamically
        self.TECHNICAL_ERROR_THRESHOLD = 1000  # mm per frame
        self.MAX_VELOCITY = 12.0  # m/s (based on sprint world record)
        self.MIN_VELOCITY = -2.0  # m/s
        self.QUALITY_THRESHOLDS = {
            'excellent': {'gaps': 0, 'noise': 100},  # no gaps, max noise 100mm
            'good': {'gaps': 1, 'noise': 200},       # max 1 gap, max noise 200mm
            'fair': {'gaps': 2, 'noise': 300},       # max 2 gaps, max noise 300mm
            'poor': {'gaps': float('inf'), 'noise': float('inf')}  # anything worse
        }

    def determine_sampling_rate(self, distances: List[float], takeoff_point: float) -> int:
        """
        Determine sampling rate based on velocity in 1m-6m interval before takeoff
        
        Args:
            distances (List[float]): List of distance measurements (in mm)
            takeoff_point (float): Takeoff point distance (in mm)
            
        Returns:
            int: Sampling rate (50 or 100 Hz)
        """
        # Convert distances to meters
        distances_m = [d/1000 for d in distances]
        takeoff_point_m = takeoff_point/1000
        
        # Find indices for 1m and 6m before takeoff
        start_idx = None
        end_idx = None
        
        # Iterate through distances to find the 1m-6m region before takeoff
        for i, d in enumerate(distances_m):
            if d <= takeoff_point_m - 1 and start_idx is None:  # 1m before takeoff
                start_idx = i
            if d <= takeoff_point_m - 6:  # 6m before takeoff
                end_idx = i
                break
        
        if start_idx is None or end_idx is None or start_idx <= end_idx:
            return 50  # default if region not found or invalid indices
        
        # Calculate distance covered
        distance_covered = abs(distances_m[start_idx] - distances_m[end_idx])  # in meters
        time_steps = start_idx - end_idx
        
        if time_steps == 0:
            return 50  # default if no time difference
        
        # Calculate velocity with 50Hz
        velocity_50hz = distance_covered / (time_steps / 50)
        
        # If velocity with 50Hz is > 6 m/s, use 50Hz, else 100Hz
        return 50 if velocity_50hz > 6 else 100

    def validate_velocity(self, velocity: float) -> float:
        """
        Validate and clean velocity values with improved validation
        
        Args:
            velocity (float): Calculated velocity
            
        Returns:
            float: Validated velocity or NaN if invalid
        """
        # Check for NaN or infinite values
        if np.isnan(velocity) or np.isinf(velocity):
            return float('nan')
            
        # More realistic velocity range for sprint/jump
        # World record sprint speed is ~12.4 m/s (Usain Bolt)
        if velocity < -2 or velocity > 12:
            return float('nan')
            
        return velocity

    def calculate_velocity(self, distances: List[float]) -> List[float]:
        """
        Calculate velocity from distance measurements with validation
        
        Args:
            distances (List[float]): List of distance measurements
            
        Returns:
            List[float]: List of validated velocities in m/s
        """
        velocities = []
        prev_valid_velocity = None
        window_size = 5  # Use a small window for smoothing
        
        for i in range(len(distances)-1):
            # If we have enough points, use a moving average
            if i >= window_size-1:
                window = distances[i-window_size+1:i+2]
                # Calculate velocity using linear regression
                x = np.arange(len(window)) / self.sampling_rate
                slope, _ = np.polyfit(x, window, 1)
                velocity = slope / 1000  # Convert to m/s
            else:
                delta_d = (distances[i+1] - distances[i]) / 1000  # convert to meters
                delta_t = 1/self.sampling_rate
                
                # Skip calculation if points are identical
                if distances[i+1] == distances[i]:
                    # Use previous valid velocity if available
                    velocity = prev_valid_velocity if prev_valid_velocity is not None else 0
                else:
                    velocity = delta_d / delta_t
            
            validated_velocity = self.validate_velocity(velocity)
            if not np.isnan(validated_velocity):
                prev_valid_velocity = validated_velocity
            
            velocities.append(validated_velocity)
            
        return velocities

    def analyze_step_pattern(self, distances: List[float], window_size: int = 5) -> Dict[str, float]:
        """
        Analyze step patterns in the movement data with improved validation
        
        Args:
            distances (List[float]): List of distance measurements
            window_size (int): Size of the window for pattern analysis
            
        Returns:
            Dictionary containing step pattern statistics
        """
        steps = []
        for i in range(len(distances)-window_size):
            window = distances[i:i+window_size]
            step_sizes = np.diff(window)
            
            # Filter out unrealistic step sizes (> 300mm or < -300mm per frame at 50Hz)
            # This is based on maximum sprint velocity of ~12 m/s = 240mm per frame
            valid_steps = [s for s in step_sizes if abs(s) <= 300]
            
            if valid_steps:
                steps.extend(valid_steps)
        
        if not steps:
            return {
                'mean_step_size': 0,
                'std_step_size': 0,
                'min_step_size': 0,
                'max_step_size': 0
            }
        
        # Remove outliers using IQR method
        steps = np.array(steps)
        q1 = np.percentile(steps, 25)
        q3 = np.percentile(steps, 75)
        iqr = q3 - q1
        steps = steps[(steps >= q1 - 1.5*iqr) & (steps <= q3 + 1.5*iqr)]
        
        if len(steps) == 0:
            return {
                'mean_step_size': 0,
                'std_step_size': 0,
                'min_step_size': 0,
                'max_step_size': 0
            }
        
        return {
            'mean_step_size': np.mean(steps),
            'std_step_size': np.std(steps),
            'min_step_size': np.min(steps),
            'max_step_size': np.max(steps)
        }

    def read_data_file(self, filepath: str) -> Tuple[float, List[float], str, int]:
        """
        Read and parse a .dat file
        
        Args:
            filepath (str): Path to the data file
            
        Returns:
            Tuple containing:
            - takeoff point (float)
            - movement data (List[float])
            - athlete name (str)
            - attempt number (int)
        """
        # Try different encodings
        for encoding in ['latin1', 'utf-8']:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    lines = f.readlines()
                break
            except UnicodeDecodeError:
                continue

        # Extract metadata
        athlete_name = lines[1].strip()
        attempt_num = int(lines[2].strip())
        # Convert takeoff point correctly: 45.680 -> 45680 mm
        takeoff_str = lines[3].strip().replace(',', '.')
        if '.' in takeoff_str:
            # Remove the decimal point and any trailing zeros
            takeoff_str = takeoff_str.replace('.', '')
        takeoff_point = float(takeoff_str)

        # Extract movement data
        movement_data = []
        for line in lines[8:]:  # Data starts from line 9
            try:
                val = float(line.strip().replace(',', '.'))
                movement_data.append(val)
            except:
                continue

        # Determine sampling rate
        self.sampling_rate = self.determine_sampling_rate(movement_data, takeoff_point)

        return takeoff_point, movement_data, athlete_name, attempt_num

    def check_for_gaps(self, data: List[float], takeoff_point: float) -> List[Tuple[int, float, float, float]]:
        """
        Check for gaps in the movement data before the takeoff point
        
        Args:
            data (List[float]): Movement data points
            takeoff_point (float): Takeoff point distance from line 4
            
        Returns:
            List of tuples containing gap information (index, value1, value2, difference)
        """
        gaps = []
        for i in range(len(data)-1):
            # Only check points before takeoff
            if data[i] >= takeoff_point:
                break
            diff = abs(data[i+1] - data[i])
            if diff > 1000:  # Gap threshold of 1 meter
                gaps.append((i, data[i], data[i+1], diff))
        return gaps

    def ssa_interpolate_gap(self, data: List[float], gap_start: int, gap_end: int, window_size: int = 20) -> List[float]:
        """
        Interpoliert eine Lücke im Zeitverlauf mit SSA und gibt die rekonstruierten Werte für die Lücke zurück.
        Args:
            data (List[float]): Zeitreihe
            gap_start (int): Index vor der Lücke
            gap_end (int): Index nach der Lücke
            window_size (int): SSA-Fenstergröße
        Returns:
            List[float]: Interpolierte Werte für die Lücke
        """
        # Nur den Bereich um die Lücke für SSA nehmen
        left = max(0, gap_start - window_size)
        right = min(len(data), gap_end + window_size)
        segment = np.array(data[left:right])
        # Lücke maskieren
        mask = np.ones_like(segment, dtype=bool)
        mask[gap_start-left+1:gap_end-left] = False
        segment_masked = segment.copy()
        segment_masked[~mask] = np.nan
        # SSA auf den maskierten Bereich anwenden
        ssa = SingularSpectrumAnalysis(window_size=window_size)
        # SSA kann keine NaNs, daher einfache lineare Interpolation als Fallback für NaNs
        segment_filled = segment_masked.copy()
        nans = np.isnan(segment_filled)
        if np.any(nans):
            segment_filled[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), segment_filled[~nans])
        # SSA-Rekonstruktion
        segment_ssa = ssa.fit_transform(segment_filled.reshape(1, -1))[0]
        # Nur die Lücke extrahieren
        interpolated = segment_ssa[gap_start-left+1:gap_end-left]
        return interpolated.tolist()

    def fill_gaps_with_ssa(self, data: List[float], gaps: List[Dict[str, Any]], window_size: int = 20) -> Tuple[List[float], List[Tuple[int, int]]]:
        """
        Füllt alle Lücken im Datensatz mit SSA-Interpolation und gibt die neue Zeitreihe sowie die Lückenbereiche zurück.
        Args:
            data (List[float]): Zeitreihe
            gaps (List[dict]): Liste der Lücken (mit 'index')
            window_size (int): SSA-Fenstergröße
        Returns:
            Tuple[List[float], List[Tuple[int, int]]]: Neue Zeitreihe, Liste der Lückenbereiche (start, end)
        """
        data_filled = data.copy()
        gap_ranges = []
        for gap in gaps:
            idx = gap['index']
            # Lücke ist zwischen idx und idx+1
            gap_start = idx
            gap_end = idx+1
            # Suche, ob mehrere aufeinanderfolgende Lücken existieren
            while gap_end < len(data)-1 and any(g['index'] == gap_end for g in gaps):
                gap_end += 1
            # Interpolieren
            interpolated = self.ssa_interpolate_gap(data_filled, gap_start, gap_end, window_size)
            data_filled[gap_start+1:gap_end] = interpolated
            gap_ranges.append((gap_start+1, gap_end))
        return data_filled, gap_ranges

    def plot_movement_profile(self, data: List[float], takeoff_point: float, 
                            athlete_name: str, attempt_num: int, 
                            save_path: str = None, gaps: list = None, ssa_filled: List[float] = None, ssa_ranges: List[Tuple[int, int]] = None, status: str = None):
        """
        Plot movement profile mit SSA-Interpolation und Statusanzeige.
        """
        velocities = self.calculate_velocity(data)
        valid_velocities = [v for v in velocities if not np.isnan(v)]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        # Originaldaten
        ax1.plot(data, label='Distance', color='blue', linewidth=2)
        # SSA-interpolierte Daten
        if ssa_filled is not None:
            ax1.plot(ssa_filled, label='SSA Interpolated', color='purple', linestyle='--', linewidth=2)
            # Lückenbereiche hervorheben
            if ssa_ranges:
                for start, end in ssa_ranges:
                    ax1.axvspan(start, end, color='purple', alpha=0.2)
        # Takeoff
        ax1.axhline(y=takeoff_point, color='r', linestyle='--', label=f'Takeoff Point ({takeoff_point:.2f} mm)')
        ax1.set_title(f'Movement Profile - {athlete_name} (Attempt {attempt_num})', fontsize=14)
        ax1.set_xlabel('Measurement Points', fontsize=12)
        ax1.set_ylabel('Distance (mm)', fontsize=12)
        # Gaps farbig markieren
        if gaps:
            for gap in gaps:
                idx = gap['index']
                if gap['zone_6_1']:
                    ax1.axvline(x=idx, color='red', linestyle='-', linewidth=2, alpha=0.7, label='Gap 6-1m (ROT)')
                elif gap['zone_11_6']:
                    ax1.axvline(x=idx, color='gold', linestyle='-', linewidth=2, alpha=0.7, label='Gap 11-6m (GELB)')
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax1.legend(by_label.values(), by_label.keys(), fontsize=10)
        # Status-Text
        if status:
            ax1.text(0.98, 0.02, f'Status: {status.upper()}', transform=ax1.transAxes, fontsize=14, color={'grün':'green','gelb':'gold','rot':'red'}.get(status,'black'), ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        # Velocity
        valid_indices = [i for i, v in enumerate(velocities) if not np.isnan(v)]
        valid_velocities = [v for v in velocities if not np.isnan(v)]
        if valid_velocities:
            ax2.plot(valid_indices, valid_velocities, label='Velocity', color='green', linewidth=2)
            mean_vel = np.mean(valid_velocities)
            std_vel = np.std(valid_velocities)
            ax2.axhline(y=mean_vel, color='r', linestyle='--', label=f'Mean: {mean_vel:.2f} m/s')
            ax2.fill_between(valid_indices, [mean_vel - 2*std_vel]*len(valid_indices), [mean_vel + 2*std_vel]*len(valid_indices), color='green', alpha=0.2, label='95% Confidence')
        ax2.set_title('Velocity Profile', fontsize=14)
        ax2.set_xlabel('Measurement Points', fontsize=12)
        ax2.set_ylabel('Velocity (m/s)', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def analyze_technical_quality(self, distances: List[float]) -> Dict[str, Any]:
        """
        Analyze technical quality of the measurement
        
        Args:
            distances (List[float]): List of distance measurements
            
        Returns:
            Dictionary containing quality metrics
        """
        # Calculate frame-to-frame differences
        diffs = np.diff(distances)
        
        # Detect technical problems
        problems = {
            'large_jumps': [(i, diff) for i, diff in enumerate(diffs) if abs(diff) > self.TECHNICAL_ERROR_THRESHOLD],
            'static_regions': [],
            'noise_level': np.std(diffs),
            'missing_data': [i for i, d in enumerate(distances) if d == 0]
        }
        
        # Detect static regions (no movement)
        static_start = None
        for i, diff in enumerate(diffs):
            if abs(diff) < 1:  # 1mm threshold for static
                if static_start is None:
                    static_start = i
            elif static_start is not None:
                if i - static_start > 5:  # static for more than 5 frames
                    problems['static_regions'].append((static_start, i))
                static_start = None
        
        # Determine quality rating
        quality = 'poor'
        for rating, thresholds in self.QUALITY_THRESHOLDS.items():
            if (len(problems['large_jumps']) <= thresholds['gaps'] and 
                problems['noise_level'] <= thresholds['noise']):
                quality = rating
                break
        
        problems['quality_rating'] = quality
        return problems

    def calculate_runup_velocity(self, distances: List[float], takeoff_point: float) -> Dict[str, Any]:
        """
        Calculate velocity statistics for the run-up phase
        
        Args:
            distances (List[float]): List of distance measurements
            takeoff_point (float): Takeoff point distance
            
        Returns:
            Dictionary containing velocity statistics
        """
        # Find takeoff index
        takeoff_idx = None
        for i, d in enumerate(distances):
            if d >= takeoff_point:
                takeoff_idx = i
                break
        
        if takeoff_idx is None:
            takeoff_idx = len(distances)
        
        # Calculate velocities for run-up phase only
        runup_distances = distances[:takeoff_idx]
        runup_velocities = self.calculate_velocity(runup_distances)
        
        # Calculate overall velocities
        all_velocities = self.calculate_velocity(distances)
        
        # Filter out invalid velocities
        valid_runup = [v for v in runup_velocities if not np.isnan(v)]
        valid_all = [v for v in all_velocities if not np.isnan(v)]
        
        return {
            'runup_velocity': {
                'mean': np.mean(valid_runup) if valid_runup else 0,
                'max': np.max(valid_runup) if valid_runup else 0,
                'std': np.std(valid_runup) if valid_runup else 0
            },
            'overall_velocity': {
                'mean': np.mean(valid_all) if valid_all else 0,
                'max': np.max(valid_all) if valid_all else 0,
                'std': np.std(valid_all) if valid_all else 0
            }
        }

    def analyze_movement_data(self, filepath: str) -> Dict[str, Any]:
        """
        Analyze a single movement data file
        
        Args:
            filepath (str): Path to the data file
            
        Returns:
            Dictionary containing analysis results
        """
        # Read and parse file
        takeoff_point, distances, athlete_name, attempt_num = self.read_data_file(filepath)
        
        # Analyze technical quality
        quality_analysis = self.analyze_technical_quality(distances)
        
        # Calculate velocities
        velocity_analysis = self.calculate_runup_velocity(distances, takeoff_point)
        
        # Analyze step patterns
        step_analysis = self.analyze_step_pattern(distances)
        
        # Find gaps
        gaps = self.check_for_gaps(distances, takeoff_point)
        
        return {
            'metadata': {
                'athlete': athlete_name,
                'attempt': attempt_num,
                'takeoff_point': takeoff_point,
                'sampling_rate': self.sampling_rate
            },
            'technical_quality': quality_analysis,
            'velocity_analysis': velocity_analysis,
            'step_analysis': step_analysis,
            'gaps': gaps
        }

    def print_analysis_results(self, results: Dict[str, Any]):
        """
        Print analysis results in a formatted way
        
        Args:
            results (Dict[str, Any]): Analysis results dictionary
        """
        print(f"\nFile Analysis Results:")
        print(f"Athlete: {results['metadata']['athlete']}")
        print(f"Attempt: {results['metadata']['attempt']}")
        print(f"Takeoff point: {results['metadata']['takeoff_point']:.2f} mm")
        print(f"Sampling rate: {results['metadata']['sampling_rate']} Hz")
        
        print("\nTechnical Quality:")
        print(f"Quality Rating: {results['technical_quality']['quality_rating']}")
        print(f"Noise Level: {results['technical_quality']['noise_level']:.2f} mm")
        if results['technical_quality']['large_jumps']:
            print("Large Data Jumps Detected:")
            for idx, jump in results['technical_quality']['large_jumps']:
                print(f"  → Frame {idx}: {jump:.1f} mm")
        
        print("\nVelocity Analysis:")
        print("Run-up Phase:")
        print(f"  Average: {results['velocity_analysis']['runup_velocity']['mean']:.2f} m/s")
        print(f"  Maximum: {results['velocity_analysis']['runup_velocity']['max']:.2f} m/s")
        print("Overall:")
        print(f"  Average: {results['velocity_analysis']['overall_velocity']['mean']:.2f} m/s")
        print(f"  Maximum: {results['velocity_analysis']['overall_velocity']['max']:.2f} m/s")
        
        print("\nStep Analysis:")
        print(f"Mean step size: {results['step_analysis']['mean_step_size']:.2f} mm")
        print(f"Step size std: {results['step_analysis']['std_step_size']:.2f} mm")
        
        if results['gaps']:
            print("\nGaps found before takeoff point:")
            for g in results['gaps']:
                print(f"  → Point {g[0]}: {g[1]:.1f} → {g[2]:.1f} (Gap: {g[3]:.1f} mm)")
        else:
            print("\nNo gaps found before takeoff point.")

    def get_status_from_gaps(self, gaps: List[Dict[str, Any]]) -> str:
        """
        Gibt den Status (grün/gelb/rot) je nach Lückenlage zurück.
        """
        for gap in gaps:
            if gap['zone_6_1']:
                return 'rot'
        for gap in gaps:
            if gap['zone_11_6']:
                return 'gelb'
        return 'grün'

    def analyze_all_files(self):
        """
        Analyze all .dat files in the specified folders and generate plots with highlighted gaps in critical zones and SSA-interpolierten Bereichen. Exportiere eine CSV-Übersicht.
        """
        print("\n=== Movement Data Analysis Report ===\n")
        csv_rows = []
        for folder in self.folders:
            print(f"\nAnalyzing folder: {folder}\n")
            for fname in os.listdir(folder):
                if fname.lower().endswith('.dat'):
                    fpath = os.path.join(folder, fname)
                    try:
                        takeoff_point, distances, athlete_name, attempt_num = self.read_data_file(fpath)
                        gap_analysis = self.analyze_gaps_until_takeoff(fpath)
                        results = self.analyze_movement_data(fpath)
                        # SSA-Interpolation
                        ssa_filled, ssa_ranges = self.fill_gaps_with_ssa(distances, gap_analysis['gaps'])
                        # Status
                        status = self.get_status_from_gaps(gap_analysis['gaps'])
                        # Print results
                        self.print_analysis_results(results)
                        # Plot
                        plot_save_path = os.path.join(folder, f"{fname[:-4]}_analysis.png")
                        self.plot_movement_profile(distances, takeoff_point, athlete_name, attempt_num, plot_save_path, gaps=gap_analysis['gaps'], ssa_filled=ssa_filled, ssa_ranges=ssa_ranges, status=status)
                        # CSV-Export
                        csv_rows.append({
                            'file': fname,
                            'athlete': athlete_name,
                            'attempt': attempt_num,
                            'status': status,
                            'num_gaps': gap_analysis['number_of_gaps'],
                            'num_gaps_11_6': len(gap_analysis['gaps_11_6']),
                            'num_gaps_6_1': len(gap_analysis['gaps_6_1']),
                            'max_gap': max([g['difference'] for g in gap_analysis['gaps']] or [0]),
                            'mean_gap': np.mean([g['difference'] for g in gap_analysis['gaps']] or [0]),
                            'takeoff_point': takeoff_point
                        })
                    except Exception as e:
                        print(f"Error processing {fname}: {str(e)}")
        # Schreibe CSV
        with open('gap_status_report.csv', 'w', newline='') as csvfile:
            fieldnames = ['file', 'athlete', 'attempt', 'status', 'num_gaps', 'num_gaps_11_6', 'num_gaps_6_1', 'max_gap', 'mean_gap', 'takeoff_point']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in csv_rows:
                writer.writerow(row)
        print("\nCSV-Report 'gap_status_report.csv' wurde erzeugt.")

    def analyze_gaps_until_takeoff(self, filepath: str) -> Dict[str, Any]:
        """
        Analyze gaps in the data specifically until takeoff point and check if they are in the 11-6m or 6-1m zones before takeoff.
        
        Args:
            filepath (str): Path to the data file
            
        Returns:
            Dictionary containing gap analysis results
        """
        # Read file
        takeoff_point, distances, athlete_name, attempt_num = self.read_data_file(filepath)
        
        # Find takeoff index
        takeoff_idx = None
        for i, d in enumerate(distances):
            if d >= takeoff_point:
                takeoff_idx = i
                break
        
        if takeoff_idx is None:
            takeoff_idx = len(distances)
        
        # Analyze only data until takeoff
        runup_data = distances[:takeoff_idx]
        
        # Define critical zones (in mm)
        zone_11_6_lower = takeoff_point - 11000
        zone_11_6_upper = takeoff_point - 6000
        zone_6_1_lower = takeoff_point - 6000
        zone_6_1_upper = takeoff_point - 1000
        
        # Find gaps and check zones
        gaps = []
        gaps_11_6 = []
        gaps_6_1 = []
        for i in range(len(runup_data)-1):
            diff = abs(runup_data[i+1] - runup_data[i])
            if diff > 1000:  # Gap threshold of 1 meter
                gap_info = {
                    'index': i,
                    'start_value': runup_data[i],
                    'end_value': runup_data[i+1],
                    'difference': diff,
                    'zone_11_6': False,
                    'zone_6_1': False
                }
                # Check if gap is in 11-6m zone
                if (zone_11_6_lower <= runup_data[i] <= zone_11_6_upper) or (zone_11_6_lower <= runup_data[i+1] <= zone_11_6_upper):
                    gap_info['zone_11_6'] = True
                    gaps_11_6.append(gap_info)
                # Check if gap is in 6-1m zone
                if (zone_6_1_lower <= runup_data[i] <= zone_6_1_upper) or (zone_6_1_lower <= runup_data[i+1] <= zone_6_1_upper):
                    gap_info['zone_6_1'] = True
                    gaps_6_1.append(gap_info)
                gaps.append(gap_info)
        
        return {
            'file': os.path.basename(filepath),
            'athlete': athlete_name,
            'attempt': attempt_num,
            'takeoff_point': takeoff_point,
            'takeoff_index': takeoff_idx,
            'total_points_to_takeoff': takeoff_idx,
            'number_of_gaps': len(gaps),
            'gaps': gaps,
            'gaps_11_6': gaps_11_6,
            'gaps_6_1': gaps_6_1
        }

    def analyze_all_files_gaps(self):
        """
        Analyze gaps in all files and provide a summary, including critical zones 11-6m and 6-1m.
        """
        print("\n=== Gap Analysis Report ===\n")
        
        total_files = 0
        files_with_gaps = 0
        all_gaps = []
        all_gaps_11_6 = []
        all_gaps_6_1 = []
        
        for folder in self.folders:
            print(f"\nAnalyzing folder: {folder}")
            
            for fname in os.listdir(folder):
                if fname.lower().endswith('.dat'):
                    total_files += 1
                    fpath = os.path.join(folder, fname)
                    
                    try:
                        analysis = self.analyze_gaps_until_takeoff(fpath)
                        
                        if analysis['number_of_gaps'] > 0:
                            files_with_gaps += 1
                            print(f"\nFile: {analysis['file']}")
                            print(f"Athlete: {analysis['athlete']}")
                            print(f"Attempt: {analysis['attempt']}")
                            print(f"Points until takeoff: {analysis['total_points_to_takeoff']}")
                            print(f"Number of gaps: {analysis['number_of_gaps']}")
                            print("Gaps found:")
                            for gap in analysis['gaps']:
                                zone_str = ""
                                if gap['zone_6_1']:
                                    zone_str = "[6-1m: ROT]"
                                elif gap['zone_11_6']:
                                    zone_str = "[11-6m: GELB]"
                                print(f"  → Point {gap['index']}: {gap['start_value']:.1f} → {gap['end_value']:.1f} (Gap: {gap['difference']:.1f} mm) {zone_str}")
                            all_gaps.extend(analysis['gaps'])
                            all_gaps_11_6.extend(analysis['gaps_11_6'])
                            all_gaps_6_1.extend(analysis['gaps_6_1'])
                    
                    except Exception as e:
                        print(f"Error processing {fname}: {str(e)}")
        
        # Print summary statistics
        print("\n=== Summary Statistics ===")
        print(f"Total files analyzed: {total_files}")
        print(f"Files with gaps: {files_with_gaps} ({(files_with_gaps/total_files*100):.1f}%)")
        if all_gaps:
            gap_sizes = [gap['difference'] for gap in all_gaps]
            print(f"Total number of gaps: {len(all_gaps)}")
            print(f"Average gap size: {np.mean(gap_sizes):.1f} mm")
            print(f"Maximum gap size: {np.max(gap_sizes):.1f} mm")
            print(f"Minimum gap size: {np.min(gap_sizes):.1f} mm")
        print(f"\nNumber of gaps in 11-6m zone (GELB): {len(all_gaps_11_6)}")
        print(f"Number of gaps in 6-1m zone (ROT): {len(all_gaps_6_1)}")

# Example usage
if __name__ == "__main__":
    folders = [
        os.path.join("/Users/andreparduhn/Documents/OSP_New/Input files/Drei M"),
        os.path.join("/Users/andreparduhn/Documents/OSP_New/Input files/Drei W"),
        os.path.join("/Users/andreparduhn/Documents/OSP_New/Input files/Weit M"),
        os.path.join("/Users/andreparduhn/Documents/OSP_New/Input files/Weit W")
    ]
    
    analyzer = MovementDataAnalyzer(folders)
    analyzer.analyze_all_files()
    analyzer.analyze_all_files_gaps() 