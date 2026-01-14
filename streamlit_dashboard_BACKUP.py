import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from analyze_movement_data import MovementDataAnalyzer
from kalman_ssa_interpolator import KalmanSSAInterpolator
from neural_interpolator import NeuralInterpolator
from data_cleaner import DataCleaner

# Page config - MUST be first Streamlit command
st.set_page_config(
    page_title="OSP Anlaufanalyse",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# OSP Colors
OSP_COLORS = {
    'red': '#e30613',
    'black': '#000000',
    'white': '#ffffff',
    'gold': '#ffd700',
    'gray': '#f5f5f5',
    'light_gray': '#e8e8e8',
    'green': '#28a745',
    'yellow': '#ffc107',
}

# Custom CSS
st.markdown(f"""
<style>
    .main {{
        padding: 0rem 1rem;
    }}
    .stApp {{
        background-color: {OSP_COLORS['white']};
    }}
    h1 {{
        color: {OSP_COLORS['red']};
        font-weight: bold;
    }}
    h2, h3, h4 {{
        color: {OSP_COLORS['black']};
    }}
    .metric-card {{
        background-color: {OSP_COLORS['gray']};
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }}
    .status-badge {{
        padding: 8px 20px;
        border-radius: 20px;
        font-weight: bold;
        color: white;
        display: inline-block;
        margin: 10px 0;
    }}
    .status-green {{
        background-color: {OSP_COLORS['green']};
    }}
    .status-yellow {{
        background-color: {OSP_COLORS['yellow']};
        color: black;
    }}
    .status-red {{
        background-color: {OSP_COLORS['red']};
    }}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_analyzer():
    """Load the analyzer with all folders"""
    folders = [
        "Input files/Drei M",
        "Input files/Drei W",
        "Input files/Weit M",
        "Input files/Weit W"
    ]
    return MovementDataAnalyzer(folders)

@st.cache_resource
def load_kalman_interpolator():
    """Load the Kalman+SSA Interpolator (Olympic-grade) with global models"""
    return KalmanSSAInterpolator(sampling_rate=50, ssa_window=40, use_global_models=True)

@st.cache_resource
def load_neural_interpolator():
    """Load the Neural Network Interpolator"""
    interpolator = NeuralInterpolator(model_path="neural_interpolator_model.pt")
    if interpolator.is_trained:
        return interpolator
    else:
        return None

@st.cache_resource
def load_data_cleaner():
    """Load the Data Cleaner for removing measurement errors"""
    return DataCleaner(max_backward_mm=500, min_forward_velocity=2.0)

@st.cache_data
def load_file_list(_analyzer):
    """Load and cache the file list"""
    file_data = []
    for folder in _analyzer.folders:
        if not os.path.exists(folder):
            continue
        for fname in os.listdir(folder):
            if fname.lower().endswith('.dat'):
                fpath = os.path.join(folder, fname)
                try:
                    takeoff_point, data, athlete_name, attempt_num = _analyzer.read_data_file(fpath)
                    gap_analysis = _analyzer.analyze_gaps_until_takeoff(fpath)
                    
                    num_gaps = gap_analysis['number_of_gaps']
                    num_gaps_6_1 = len(gap_analysis['gaps_6_1'])
                    num_gaps_11_6 = len(gap_analysis['gaps_11_6'])
                    
                    if num_gaps_6_1 > 0:
                        status = 'rot'
                        quality = 'Kritisch'
                    elif num_gaps_11_6 > 0:
                        status = 'gelb'
                        quality = 'Achtung'
                    elif num_gaps > 0:
                        status = 'grün'
                        quality = 'OK'
                    else:
                        status = 'grün'
                        quality = 'Sehr gut'
                    
                    file_data.append({
                        'filepath': fpath,
                        'filename': fname,
                        'folder': os.path.basename(folder),
                        'Athlet': athlete_name,
                        'Versuch': attempt_num,
                        'Lücken': num_gaps,
                        'Qualität': quality,
                        'status': status,
                        'takeoff_point': takeoff_point
                    })
                except Exception as e:
                    continue
    
    return pd.DataFrame(file_data).sort_values(['folder', 'Athlet', 'Versuch'])

def calculate_quality_metrics(analyzer, data, takeoff_point, ssa_filled, ssa_ranges):
    """Calculate quality metrics - NUR bis zum Absprungpunkt!"""
    try:
        # Finde den Index des Absprungpunkts (wo Position >= takeoff_point)
        data_array = np.array(data)
        takeoff_indices = np.where(data_array >= takeoff_point)[0]
        if len(takeoff_indices) > 0:
            takeoff_idx = takeoff_indices[0]
        else:
            takeoff_idx = len(data)
        
        # NUR Daten bis zum Absprung verwenden!
        relevant_data = data_array[:takeoff_idx] if takeoff_idx > 0 else data_array
        relevant_ssa = np.array(ssa_filled)[:takeoff_idx] if takeoff_idx > 0 else np.array(ssa_filled)
        
        # Interpolation quality (nur relevante Bereiche)
        if ssa_ranges:
            # Nur Ranges bis zum Absprung berücksichtigen
            relevant_ranges = [(s, e) for s, e in ssa_ranges if s < takeoff_idx]
            
            if len(relevant_data) > 1:
                original_velocities = np.diff(relevant_data)
                interpolated_velocities = np.diff(relevant_ssa)
                smoothness_score = max(0, 1 - np.mean(np.abs(np.diff(interpolated_velocities))) / 100)
            else:
                smoothness_score = 1.0
            
            continuity_scores = []
            for start, end in relevant_ranges:
                if end < len(relevant_data):
                    continuity_scores.append(abs(relevant_ssa[end] - relevant_data[end]))
            
            if continuity_scores:
                continuity_score = max(0, 1 - np.mean(continuity_scores) / np.max(relevant_data))
            else:
                continuity_score = 1.0
            
            interpolation_quality = (smoothness_score + continuity_score) / 2
        else:
            interpolation_quality = 1.0
        
        # Data quality - NUR bis Absprung
        if len(relevant_data) > 1:
            noise_level = np.std(np.diff(relevant_data))
        else:
            noise_level = 0
        noise_score = max(0, 1 - min(noise_level / 1000, 1))
        
        gaps = analyzer.check_for_gaps(list(relevant_data), takeoff_point)
        gap_score = max(0, 1 - min(len(gaps) / 5, 1))
        
        velocities = analyzer.calculate_velocity(list(relevant_data))
        valid_velocities = [v for v in velocities if not np.isnan(v)]
        if valid_velocities:
            velocity_score = max(0, 1 - min(np.std(valid_velocities) / 5, 1))
        else:
            velocity_score = 0
        
        data_quality = (noise_score + gap_score + velocity_score) / 3
        
        # Technical stability - NUR bis Absprung
        step_pattern = analyzer.analyze_step_pattern(list(relevant_data))
        step_stability = max(0, 1 - min(step_pattern['std_step_size'] / 100, 1))
        
        if valid_velocities:
            velocity_stability = max(0, 1 - min(np.std(valid_velocities) / 5, 1))
        else:
            velocity_stability = 0
        
        accelerations = np.diff(velocities)
        valid_accelerations = [a for a in accelerations if not np.isnan(a)]
        if valid_accelerations:
            acceleration_stability = max(0, 1 - min(np.std(valid_accelerations) / 2, 1))
        else:
            acceleration_stability = 0
        
        technical_stability = (step_stability + velocity_stability + acceleration_stability) / 3
        
        return {
            'interpolation_quality': interpolation_quality,
            'data_quality': data_quality,
            'technical_stability': technical_stability,
            'noise_level': noise_level,
            'gap_count': len(gaps)
        }
    except:
        return {
            'interpolation_quality': 0.5,
            'data_quality': 0.5,
            'technical_stability': 0.5,
            'noise_level': 0,
            'gap_count': 0
        }

def analyze_zone(analyzer, data, takeoff_point, zone_start, zone_end):
    """Analyze a specific zone"""
    try:
        zone_start_idx = None
        zone_end_idx = None
        
        for i, d in enumerate(data):
            if d >= takeoff_point - zone_start and zone_start_idx is None:
                zone_start_idx = i
            if d >= takeoff_point - zone_end:
                zone_end_idx = i
                break
        
        if zone_start_idx is None or zone_end_idx is None or zone_start_idx >= zone_end_idx:
            return None
        
        zone_data = data[zone_start_idx:zone_end_idx]
        
        if len(zone_data) < 2:
            return None
        
        velocities = analyzer.calculate_velocity(zone_data)
        valid_velocities = [v for v in velocities if not np.isnan(v)]
        
        if len(valid_velocities) < 2:
            return None
        
        accelerations = np.diff(valid_velocities)
        step_lengths = np.diff(zone_data)
        
        gaps = []
        for gap in analyzer.check_for_gaps(data, takeoff_point):
            if zone_start_idx <= gap[0] <= zone_end_idx:
                gaps.append(gap)
        
        return {
            'mean_velocity': np.mean(valid_velocities),
            'velocity_std': np.std(valid_velocities),
            'step_length_mean': np.mean(step_lengths),
            'step_length_std': np.std(step_lengths),
            'acceleration_mean': np.mean(accelerations) if len(accelerations) > 0 else 0,
            'acceleration_std': np.std(accelerations) if len(accelerations) > 0 else 0,
            'data_points': len(zone_data),
            'gaps': len(gaps)
        }
    except:
        return None

def create_plot(analyzer, data, takeoff_point, athlete_name, attempt_num, interpolated_data, interpolation_ranges, gaps, show_interpolation=False):
    """Create interactive plotly figure with cleaned data"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Distanz-Profil', 'Geschwindigkeits-Profil'),
        vertical_spacing=0.12,
        row_heights=[0.5, 0.5]
    )
    
    # Show original data as light gray dashed line (for comparison)
    if show_interpolation:
        fig.add_trace(go.Scatter(
            y=[d/1000 for d in data],
            name='Original (mit Fehlern)', 
            line=dict(color='lightgray', width=1, dash='dot'),
            hovertemplate='Frame: %{x}<br>Original: %{y:.2f} m<extra></extra>',
            opacity=0.5
        ), row=1, col=1)
    
    # Data to plot (cleaned if available)
    plot_data = interpolated_data if show_interpolation else data
    data_label = 'Bereinigt' if show_interpolation else 'Messdaten'
    data_color = '#28a745' if show_interpolation else 'blue'
    
    fig.add_trace(go.Scatter(
        y=[d/1000 for d in plot_data],  # Convert to meters
        name=data_label, 
        line=dict(color=data_color, width=2),
        hovertemplate='Frame: %{x}<br>Distanz: %{y:.2f} m<extra></extra>'
    ), row=1, col=1)
    
    # Show interpolated/cleaned regions - IMPROVED VISUALIZATION
    if show_interpolation and interpolation_ranges:
        # Highlight interpolated points with markers
        interp_x = []
        interp_y = []
        for (start, end) in interpolation_ranges:
            for i in range(start, end):
                if i < len(plot_data):
                    interp_x.append(i)
                    interp_y.append(plot_data[i] / 1000)
        
        if interp_x:
            # Add interpolated points as distinct markers
            fig.add_trace(go.Scatter(
                x=interp_x,
                y=interp_y,
                mode='markers',
                marker=dict(
                    size=6,
                    color='#a259d9',
                    symbol='circle',
                    line=dict(width=1, color='white')
                ),
                name='Interpolierte Punkte',
                hovertemplate='Frame: %{x}<br>Interpoliert: %{y:.2f} m<extra></extra>'
            ), row=1, col=1)
        
        # Add subtle vertical shading (reduced opacity)
        for (start, end) in interpolation_ranges:
            # Get Y-range for this interpolation
            y_start = plot_data[start] / 1000 if start < len(plot_data) else 0
            y_end = plot_data[min(end-1, len(plot_data)-1)] / 1000 if end > 0 else 0
            
            # Add a shape that only covers the interpolated Y-range (plus some margin)
            y_min = min(y_start, y_end) - 2
            y_max = max(y_start, y_end) + 2
            
            fig.add_shape(
                type="rect",
                x0=start, x1=end,
                y0=y_min, y1=y_max,
                fillcolor='#a259d9',
                opacity=0.15,
                line=dict(width=2, color='#a259d9', dash='dot'),
                row=1, col=1
            )
            
            # Add annotation showing the interpolation details
            mid_x = (start + end) / 2
            mid_y = (y_start + y_end) / 2
            gap_size = abs(y_end - y_start)
            fig.add_annotation(
                x=mid_x,
                y=y_max + 1,
                text=f"↕ {gap_size:.1f}m interpoliert",
                showarrow=False,
                font=dict(size=10, color='#a259d9'),
                row=1, col=1
            )
    
    # Mark problem areas (only if NOT showing cleaning)
    if gaps and not show_interpolation:
        gap_legend_added = False
        
        for gap in gaps:
            idx = gap['index']
            gap_size_m = gap['difference'] / 1000
            
            fig.add_vline(
                x=idx, 
                line_dash="dash", 
                line_color='red',
                line_width=2,
                opacity=0.6,
                row=1, col=1,
                annotation_text=f"Fehler: {gap_size_m:.1f}m",
                annotation_position="top"
            )
            
            fig.add_vrect(
                x0=idx-0.5, x1=idx+1.5,
                fillcolor='red',
                opacity=0.1,
                line_width=0,
                row=1, col=1
            )
            
            if not gap_legend_added:
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode='lines',
                    line=dict(color='red', width=2, dash='dash'),
                    name='Messfehler',
                    showlegend=True
                ), row=1, col=1)
                gap_legend_added = True
    
    # Takeoff line
    fig.add_hline(
        y=takeoff_point/1000,  # Convert to meters
        line_dash="dash", 
        line_color=OSP_COLORS['red'],
        line_width=2,
        annotation_text=f"Absprung ({takeoff_point/1000:.2f}m)",
        annotation_position="right",
        row=1, col=1
    )
    
    # Critical zones (in meters)
    zone_11_6_start = (takeoff_point - 11000) / 1000
    zone_11_6_end = (takeoff_point - 6000) / 1000
    zone_6_1_start = (takeoff_point - 6000) / 1000
    zone_6_1_end = (takeoff_point - 1000) / 1000
    
    fig.add_hrect(
        y0=zone_11_6_start, y1=zone_11_6_end,
        fillcolor=OSP_COLORS['gold'], 
        opacity=0.15,
        line_width=1,
        line_color=OSP_COLORS['gold'],
        annotation_text="Zone 11-6m vor Absprung",
        annotation_position="left",
        row=1, col=1
    )
    
    fig.add_hrect(
        y0=zone_6_1_start, y1=zone_6_1_end,
        fillcolor=OSP_COLORS['red'], 
        opacity=0.12,
        line_width=1,
        line_color=OSP_COLORS['red'],
        annotation_text="Zone 6-1m vor Absprung (KRITISCH)",
        annotation_position="left",
        row=1, col=1
    )
    
    # Velocity profile (from interpolated data if available)
    velocities = analyzer.calculate_velocity(list(plot_data))
    v_plot = np.array(velocities)
    nans = np.isnan(v_plot)
    if np.any(~nans):
        if np.any(nans):
            v_plot[nans] = np.interp(
                np.flatnonzero(nans), 
                np.flatnonzero(~nans), 
                v_plot[~nans]
            )
        
        fig.add_trace(go.Scatter(
            y=v_plot, 
            name='Geschwindigkeit', 
            line=dict(color='green', width=2),
            hovertemplate='Frame: %{x}<br>Geschw.: %{y:.2f} m/s<extra></extra>'
        ), row=2, col=1)
        
        # Mark interpolated regions in velocity profile too
        if show_interpolation and interpolation_ranges:
            for (start, end) in interpolation_ranges:
                # Highlight velocity values in interpolated region
                if start < len(v_plot) and end <= len(v_plot):
                    interp_v_x = list(range(start, min(end, len(v_plot))))
                    interp_v_y = [v_plot[i] for i in interp_v_x if i < len(v_plot)]
                    
                    if interp_v_y:
                        fig.add_trace(go.Scatter(
                            x=interp_v_x,
                            y=interp_v_y,
                            mode='markers',
                            marker=dict(size=4, color='#a259d9', symbol='circle'),
                            name='Geschw. (interpoliert)',
                            showlegend=False,
                            hovertemplate='Frame: %{x}<br>Geschw. (interp.): %{y:.2f} m/s<extra></extra>'
                        ), row=2, col=1)
                
                # Add light shading
                fig.add_vrect(
                    x0=start, x1=end,
                    fillcolor='#a259d9',
                    opacity=0.1,
                    line_width=0,
                    row=2, col=1
                )
        
        # Mean velocity line
        mean_vel = np.nanmean(velocities)
        fig.add_hline(
            y=mean_vel,
            line_dash="dash",
            line_color='red',
            line_width=1,
            annotation_text=f"Ø {mean_vel:.2f} m/s",
            annotation_position="right",
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=f'{athlete_name} - Versuch {attempt_num}',
        title_font_size=20,
        title_font_color=OSP_COLORS['red'],
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            font=dict(size=12)
        ),
        height=700,
        hovermode='x unified',
        plot_bgcolor=OSP_COLORS['white'],
        paper_bgcolor=OSP_COLORS['white'],
        font=dict(family='Arial, sans-serif', size=12)
    )
    
    fig.update_xaxes(title_text='Messpunkt', row=1, col=1, gridcolor=OSP_COLORS['light_gray'])
    fig.update_xaxes(title_text='Messpunkt', row=2, col=1, gridcolor=OSP_COLORS['light_gray'])
    fig.update_yaxes(title_text='Distanz (m)', row=1, col=1, gridcolor=OSP_COLORS['light_gray'])
    fig.update_yaxes(title_text='Geschwindigkeit (m/s)', row=2, col=1, gridcolor=OSP_COLORS['light_gray'])
    
    return fig

def main():
    # Header
    col1, col2 = st.columns([1, 4])
    with col1:
        # Try to load local logo, fallback to online version
        import os
        logo_path = "osp_logo.png"
        if os.path.exists(logo_path):
            st.image(logo_path, width=200)
        else:
            st.image("https://osp-hessen.de/wp-content/uploads/2021/03/OSP_Logo_2021_RGB.png", width=150)
    with col2:
        st.title("Anlaufanalyse Dashboard")
    
    st.markdown("---")
    
    # Load data
    try:
        analyzer = load_analyzer()
        kalman_interpolator = load_kalman_interpolator()
        neural_interpolator = load_neural_interpolator()
        data_cleaner = load_data_cleaner()
        df = load_file_list(analyzer)
    except Exception as e:
        st.error(f"Fehler beim Laden der Daten: {e}")
        st.info("Stelle sicher, dass die 'Input files' Ordner vorhanden sind.")
        return
    
    # 2-Column Layout
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        st.subheader("📋 Versuche")
        
        # Filters
        with st.expander("🔍 Filter", expanded=False):
            quality_filter = st.multiselect(
                "Qualität",
                options=df['Qualität'].unique(),
                default=df['Qualität'].unique()
            )
            
            folder_filter = st.multiselect(
                "Disziplin",
                options=df['folder'].unique(),
                default=df['folder'].unique()
            )
        
        # Apply filters
        filtered_df = df[
            (df['Qualität'].isin(quality_filter)) &
            (df['folder'].isin(folder_filter))
        ]
        
        # Display table with selection
        display_df = filtered_df[['Athlet', 'Versuch', 'Lücken', 'Qualität']].copy()
        display_df.insert(0, 'Index', range(len(display_df)))
        
        # Color code quality
        def color_quality(row):
            if row['Qualität'] == 'Kritisch':
                return ['background-color: #f8d7da'] * len(row)
            elif row['Qualität'] == 'Achtung':
                return ['background-color: #fff3cd'] * len(row)
            elif row['Qualität'] == 'Sehr gut':
                return ['background-color: #d4edda'] * len(row)
            else:
                return [''] * len(row)
        
        styled_df = display_df.style.apply(color_quality, axis=1)
        
        # Show interactive dataframe with selection
        event = st.dataframe(
            styled_df,
            use_container_width=True,
            height=500,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row"
        )
        
        # Get selected row from dataframe interaction
        if event.selection and event.selection.rows:
            selected_row_idx = event.selection.rows[0]
            selected_file = filtered_df.iloc[selected_row_idx]
        else:
            # Default to first row if nothing selected
            if len(filtered_df) == 0:
                st.warning("Keine Versuche gefunden. Bitte Filter anpassen.")
                return
            selected_file = filtered_df.iloc[0]
        
        st.markdown("---")
        st.caption(f"📌 Ausgewählt: {selected_file['Athlet']} - Versuch {selected_file['Versuch']}")
    
    with col_right:
        # Header with interpolation method selection
        col_header1, col_header2 = st.columns([2, 1])
        with col_header1:
            st.subheader("📊 Detailanalyse")
        with col_header2:
            interpolation_method = st.selectbox(
                "Interpolationsmethode",
                options=[
                    "Keine", 
                    "Data Cleaner (linear)", 
                    "Data Cleaner + PCHIP",
                    "Data Cleaner + Kalman",
                    "Data Cleaner + SSA",
                    "Data Cleaner + Kalman+SSA", 
                    "Data Cleaner + Neural"
                ],
                index=5,  # Default: Data Cleaner + Kalman+SSA
                help="Alle Methoden bereinigen erst Messfehler, dann interpolieren. Fokus: Daten bis zum Absprung."
            )
            show_interpolation = interpolation_method != "Keine"
        
        try:
            # Load selected file
            filepath = selected_file['filepath']
            takeoff_point, data, athlete_name, attempt_num = analyzer.read_data_file(filepath)
            gap_analysis = analyzer.analyze_gaps_until_takeoff(filepath)
            gaps = gap_analysis['gaps']
            
            # Data Cleaning + Interpolation (removes measurement errors, then interpolates)
            interpolated_data = data
            interpolation_ranges = []
            interpolation_info = []
            
            if show_interpolation:
                try:
                    # SCHRITT 1: Immer zuerst Data Cleaner für Messfehler-Bereinigung
                    if data_cleaner:
                        cleaned_data, cleaner_gaps_info, removed_indices = data_cleaner.clean_and_interpolate(
                        data, sampling_rate=analyzer.sampling_rate or 50
                    )
                    else:
                        cleaned_data = np.array(data)
                        cleaner_gaps_info = []
                        removed_indices = []
                    
                    # SCHRITT 2: Je nach Methode interpolieren
                    # Methodenauswahl für Interpolator
                    method_map = {
                        "Data Cleaner (linear)": "linear",
                        "Data Cleaner + PCHIP": "pchip",
                        "Data Cleaner + Kalman": "kalman",
                        "Data Cleaner + SSA": "ssa",
                        "Data Cleaner + Kalman+SSA": "kalman_ssa",
                        "Data Cleaner + Neural": "neural"
                    }
                    
                    if interpolation_method == "Data Cleaner (linear)":
                        # Nur Data Cleaner (lineare Interpolation bereits gemacht)
                        interpolated_data = cleaned_data
                    
                        for gap in cleaner_gaps_info:
                            interpolation_info.append({
                                'index': gap['start_idx'],
                                'size_mm': gap['gap_size_mm'],
                                'size_m': gap['gap_size_mm'] / 1000,
                                'num_points': gap['end_idx'] - gap['start_idx'],
                                'confidence': 0.80,
                                'method': 'Linear (Data Cleaner)',
                                'start_idx': gap['start_idx'],
                                'end_idx': gap['end_idx'],
                                'removed_points': gap.get('removed_points', 0)
                            })
                            interpolation_ranges.append((gap['start_idx'], gap['end_idx']))
                    
                    elif interpolation_method in ["Data Cleaner + PCHIP", "Data Cleaner + Kalman", 
                                                   "Data Cleaner + SSA", "Data Cleaner + Kalman+SSA"]:
                        # Methode aus Map holen
                        interp_method = method_map.get(interpolation_method, "kalman_ssa")
                        method_display = {
                            "pchip": "PCHIP (Monoton)",
                            "kalman": "Kalman Filter",
                            "ssa": "SSA (Muster)",
                            "kalman_ssa": "Kalman+SSA Hybrid"
                        }.get(interp_method, interp_method)
                        
                        if not kalman_interpolator:
                            st.warning(f"Interpolator nicht verfügbar, verwende lineare Interpolation")
                            interpolated_data = cleaned_data
                            for gap in cleaner_gaps_info:
                                interpolation_info.append({
                                    'index': gap['start_idx'],
                                    'size_mm': gap['gap_size_mm'],
                                    'size_m': gap['gap_size_mm'] / 1000,
                                    'num_points': gap['end_idx'] - gap['start_idx'],
                                    'confidence': 0.70,
                                    'method': 'Linear (Fallback)',
                                    'start_idx': gap['start_idx'],
                                    'end_idx': gap['end_idx'],
                                    'removed_points': gap.get('removed_points', 0)
                                })
                                interpolation_ranges.append((gap['start_idx'], gap['end_idx']))
                        elif cleaner_gaps_info:
                            interpolated_data = np.array(cleaned_data)
                            
                            for gap in cleaner_gaps_info:
                                start_idx = gap['start_idx']
                                end_idx = gap['end_idx']
                                start_value = gap['start_value']
                                end_value = gap['end_value']
                                gap_size_mm = abs(end_value - start_value)
                                num_points = end_idx - start_idx
                                
                                # Interpolation mit gewählter Methode
                                # WICHTIG: num_points übergeben um Sampling-Dichte zu erhalten!
                                interp_result, confidence = kalman_interpolator.interpolate_gap(
                                    interpolated_data, 
                                    start_idx - 1,
                                    end_idx,
                                    gap_size_mm,
                                    method=interp_method,
                                    num_points_override=num_points  # Erhält ursprüngliche Punkt-Anzahl!
                                )
                                
                                # Ersetze interpolierte Werte
                                if len(interp_result) == num_points:
                                    interpolated_data[start_idx:end_idx] = interp_result
                                elif len(interp_result) > 1:
                                    from scipy.interpolate import interp1d
                                    x_old = np.linspace(0, 1, len(interp_result))
                                    x_new = np.linspace(0, 1, num_points)
                                    f = interp1d(x_old, interp_result, kind='linear')
                                    interpolated_data[start_idx:end_idx] = f(x_new)
                                
                                interpolation_info.append({
                                    'index': start_idx,
                                    'size_mm': gap_size_mm,
                                    'size_m': gap_size_mm / 1000,
                                    'num_points': num_points,
                                    'confidence': confidence,
                                    'method': method_display,
                                    'start_idx': start_idx,
                                    'end_idx': end_idx,
                                    'removed_points': gap.get('removed_points', 0)
                                })
                                interpolation_ranges.append((start_idx, end_idx))
                        else:
                            interpolated_data = cleaned_data
                    
                    elif interpolation_method == "Data Cleaner + Neural":
                        # Data Cleaner hat fehlerhafte Bereiche identifiziert
                        # JETZT: Neural ERSETZT die lineare Interpolation mit KI-basierter Interpolation
                        
                        if not neural_interpolator:
                            # Fallback: Use cleaned data with linear interpolation
                            st.warning("⚠️ Neural-Interpolator nicht verfügbar (Modell nicht trainiert), verwende lineare Interpolation")
                            interpolated_data = cleaned_data
                            for gap in cleaner_gaps_info:
                                interpolation_info.append({
                                    'index': gap['start_idx'],
                                    'size_mm': gap['gap_size_mm'],
                                    'size_m': gap['gap_size_mm'] / 1000,
                                    'num_points': gap['end_idx'] - gap['start_idx'],
                                    'confidence': 0.70,
                                    'method': 'Linear (Fallback)',
                                    'start_idx': gap['start_idx'],
                                    'end_idx': gap['end_idx'],
                                    'removed_points': gap.get('removed_points', 0)
                                })
                                interpolation_ranges.append((gap['start_idx'], gap['end_idx']))
                        elif cleaner_gaps_info:
                            # Starte mit bereinigten Daten
                            interpolated_data = np.array(cleaned_data)
                            
                            for gap in cleaner_gaps_info:
                                start_idx = gap['start_idx']
                                end_idx = gap['end_idx']
                                start_value = gap['start_value']
                                end_value = gap['end_value']
                                gap_size_mm = abs(end_value - start_value)
                                num_points = end_idx - start_idx
                                
                                # Verwende Neural-Interpolation für diesen Bereich
                                # Neural interpolate() erwartet: data, gap_start, gap_end
                                neural_interp, confidence = neural_interpolator.interpolate(
                                    list(interpolated_data), 
                                    start_idx - 1,  # Index vor der Lücke
                                    end_idx         # Index nach der Lücke
                                )
                                
                                # ERSETZE die linear interpolierten Werte mit Neural-Werten
                                # Bug fix: Check array length > 1 before interp1d
                                if len(neural_interp) == num_points:
                                    interpolated_data[start_idx:end_idx] = neural_interp
                                elif len(neural_interp) > 1:
                                    # Wenn Anzahl nicht passt, resize
                                    from scipy.interpolate import interp1d
                                    x_old = np.linspace(0, 1, len(neural_interp))
                                    x_new = np.linspace(0, 1, num_points)
                                    f = interp1d(x_old, neural_interp, kind='linear')
                                    interpolated_data[start_idx:end_idx] = f(x_new)
                                # else: keep cleaned_data values (already linearly interpolated)
                                
                                interpolation_info.append({
                                    'index': start_idx,
                                    'size_mm': gap_size_mm,
                                    'size_m': gap_size_mm / 1000,
                                    'num_points': num_points,
                                    'confidence': confidence,
                                    'method': 'Neural Network',
                                    'start_idx': start_idx,
                                    'end_idx': end_idx,
                                    'removed_points': gap.get('removed_points', 0)
                                })
                                interpolation_ranges.append((start_idx, end_idx))
                        else:
                            # Keine Fehler gefunden, verwende bereinigte Daten
                            interpolated_data = cleaned_data
                    
                except Exception as e:
                    st.warning(f"Interpolation fehlgeschlagen ({interpolation_method}): {e}")
                    import traceback
                    st.error(traceback.format_exc())
                    interpolated_data = data
            
            # Status badge
            status = selected_file['status']
            if status == 'rot':
                badge_class = 'status-red'
                badge_text = '🔴 KRITISCH'
            elif status == 'gelb':
                badge_class = 'status-yellow'
                badge_text = '🟡 ACHTUNG'
            else:
                badge_class = 'status-green'
                badge_text = '🟢 GUT'
            
            st.markdown(f'<div class="status-badge {badge_class}">{badge_text}</div>', unsafe_allow_html=True)
            
            # Gap badge with cleaning info
            num_gaps = len(gaps)
            if show_interpolation and interpolation_info:
                total_removed = sum(info.get('removed_points', 0) for info in interpolation_info)
                num_regions = len(interpolation_info)
                avg_confidence = np.mean([info.get('confidence', 0.8) for info in interpolation_info]) if interpolation_info else 0
                
                method_icons = {
                    "Data Cleaner (linear)": "🧹",
                    "Data Cleaner + PCHIP": "📈",
                    "Data Cleaner + Kalman": "🎯",
                    "Data Cleaner + SSA": "🔬",
                    "Data Cleaner + Kalman+SSA": "🏅",
                    "Data Cleaner + Neural": "🧠"
                }
                icon = method_icons.get(interpolation_method, "✨")
                
                # Zeige Info über Bereinigung UND Interpolation
                if "Kalman" in interpolation_method or "Neural" in interpolation_method or "PCHIP" in interpolation_method or "SSA" in interpolation_method:
                    st.success(f"{icon} **{interpolation_method}**: {total_removed} Fehler bereinigt, {num_regions} Region(en) interpoliert (Ø Konfidenz: {avg_confidence:.0%})")
                else:
                    st.success(f"{icon} **{interpolation_method}**: {total_removed} Messfehler entfernt, {num_regions} Region(en) interpoliert")
            elif num_gaps > 0:
                st.warning(f"⚠️ {num_gaps} Datenlücke(n) / Messfehler detektiert")
            else:
                st.success("✅ Keine Datenlücken oder Messfehler")
            
            cleaning_status = interpolation_method if show_interpolation else "Original"
            st.caption(f"Sampling Rate: {analyzer.sampling_rate} Hz | Absprung: {takeoff_point/1000:.2f}m | {cleaning_status}: {len(data)} Punkte")
            
            # Plot
            fig = create_plot(analyzer, data, takeoff_point, athlete_name, attempt_num, interpolated_data, interpolation_ranges, gaps, show_interpolation)
            st.plotly_chart(fig, use_container_width=True)
            
            # Metrics in 3 columns (use interpolated data if available)
            data_for_metrics = interpolated_data if show_interpolation else data
            metrics = calculate_quality_metrics(analyzer, data_for_metrics, takeoff_point, interpolated_data, interpolation_ranges)
            zone_11_6 = analyze_zone(analyzer, data_for_metrics, takeoff_point, 11000, 6000)
            zone_6_1 = analyze_zone(analyzer, data_for_metrics, takeoff_point, 6000, 1000)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### Qualität")
                st.metric("Interpolationsgüte", f"{metrics['interpolation_quality']:.1%}")
                st.metric("Datenqualität", f"{metrics['data_quality']:.1%}")
                st.metric("Stabilität", f"{metrics['technical_stability']:.1%}")
                st.metric("Rauschlevel", f"{metrics['noise_level']:.1f} mm")
                st.metric("Lücken", metrics['gap_count'])
            
            with col2:
                st.markdown("### Zone 11-6m")
                if zone_11_6:
                    st.metric("Ø Geschwindigkeit", f"{zone_11_6['mean_velocity']:.2f} m/s")
                    st.metric("Schrittlänge", f"{zone_11_6['step_length_mean']/1000:.3f} m")
                    st.metric("Beschleunigung", f"{zone_11_6['acceleration_mean']:.2f} m/s²")
                    st.metric("Lücken", zone_11_6['gaps'])
                else:
                    st.info("Keine Daten verfügbar")
            
            with col3:
                st.markdown("### Zone 6-1m")
                if zone_6_1:
                    st.metric("Ø Geschwindigkeit", f"{zone_6_1['mean_velocity']:.2f} m/s")
                    st.metric("Schrittlänge", f"{zone_6_1['step_length_mean']/1000:.3f} m")
                    st.metric("Beschleunigung", f"{zone_6_1['acceleration_mean']:.2f} m/s²")
                    st.metric("Lücken", zone_6_1['gaps'])
                else:
                    st.info("Keine Daten verfügbar")
            
            # Legende / Erklärungen
            with st.expander("Legende: Begriffe und Methoden"):
                st.markdown("""
### Qualitätsmetriken

Alle Metriken werden nur für den Anlauf **bis zum Absprungpunkt** berechnet.

| Metrik | Beschreibung |
|--------|--------------|
| **Interpolationsgüte** | Bewertet die Qualität der eingefügten Datenpunkte anhand von Glattheit und Kontinuität an den Übergängen. 100% = perfekt glatt und nahtlos. |
| **Datenqualität** | Gesamtbewertung der Rohdaten. Setzt sich zusammen aus: Rausch-Score (wie stark schwanken Schrittlängen), Lücken-Score (Anzahl Datenlücken), Geschwindigkeits-Score (Konstanz der Laufgeschwindigkeit). |
| **Stabilität** | Technische Laufstabilität des Athleten. Bewertet: Gleichmäßigkeit der Schrittlängen, Konstanz der Geschwindigkeit, Gleichmäßigkeit der Beschleunigung. |
| **Rauschlevel** | Standardabweichung der Schrittlängen in mm. Bei Schritt-Kontaktdaten misst dies die Variabilität der Schrittlängen, nicht Sensorrauschen. Niedrigere Werte = gleichmäßigere Schritte. |
| **Lücken** | Anzahl erkannter Datenlücken oder Messfehler im relevanten Bereich bis zum Absprung. |

---

### Interpolationsmethoden

| Methode | Beschreibung |
|---------|--------------|
| **Keine** | Zeigt die unverarbeiteten Originaldaten. Zum Vergleich und zur Beurteilung der Rohdatenqualität. |
| **Data Cleaner (linear)** | Erkennt und entfernt Messfehler. Lücken werden mit linearer Interpolation gefüllt - einer geraden Linie zwischen Start- und Endpunkt. Schnell, aber ohne Berücksichtigung von Bewegungsmustern. |
| **Data Cleaner + PCHIP** | Piecewise Cubic Hermite Interpolating Polynomial. Garantiert monotone (nur steigende) Werte ohne Überschwinger. Mathematisch optimal für Schrittdaten. |
| **Data Cleaner + Kalman** | Kalman Filter für physikbasierte Interpolation. Modelliert Position und Geschwindigkeit als physikalisches System. Bidirektionale Vorhersage (von beiden Seiten der Lücke). |
| **Data Cleaner + SSA** | Singular Spectrum Analysis. Extrahiert Schrittmuster aus dem Kontext vor der Lücke und wendet diese auf die Interpolation an. Gut für periodische Bewegungen. |
| **Data Cleaner + Kalman+SSA** | Kombiniert Kalman (Physik) mit SSA (Muster). Bei kleinen Lücken mehr Kalman, bei großen mehr SSA. Empfohlen als Standardmethode. |
| **Data Cleaner + Neural** | Verwendet ein trainiertes LSTM-Netz zur Interpolation. Lernt Bewegungsmuster aus allen vorhandenen Daten. |

---

### Technische Begriffe

**Data Cleaner**  
Bereinigt Bewegungsdaten von Messfehlern. Erkennt unrealistische Werte wie große Rückwärtssprünge (>500mm) und markiert diese zur Interpolation. Kleine Schwankungen werden als normales Messrauschen toleriert.

**PCHIP (Piecewise Cubic Hermite Interpolating Polynomial)**  
Mathematisch optimale Interpolation für monotone Daten. Garantiert, dass interpolierte Werte keine Überschwinger haben und immer zwischen Start- und Endwert liegen. Ideal für Schrittdaten, da Athleten nur vorwärts laufen.

**Kalman Filter**  
Mathematisches Verfahren zur Zustandsschätzung (entwickelt 1960, verwendet u.a. bei NASA). Modelliert Position und Geschwindigkeit als physikalisches System. Verwendet bidirektionale Vorhersage: Von vorne (vor der Lücke) und von hinten (nach der Lücke), dann gewichtete Kombination.

**SSA (Singular Spectrum Analysis)**  
Zeitreihenanalyse-Verfahren zur Extraktion wiederkehrender Muster. Analysiert die Schrittgrößen im Kontext vor der Lücke, extrahiert das Schrittmuster (Periodizität), und wendet dieses Muster auf die Interpolation an.

**Kalman+SSA Hybrid**  
Kombiniert beide Ansätze: Kalman liefert physikalisch plausible Grundvorhersage, SSA fügt das gelernte Schrittmuster hinzu. Gewichtung: Bei kleinen Lücken 70% Kalman / 30% SSA, bei großen Lücken 40% Kalman / 60% SSA.

**Neuronales Netz (LSTM)**  
Bidirektionales LSTM mit Attention-Mechanismus. Trainiert auf allen vorhandenen Anlaufdaten mit künstlich erzeugten Lücken (Self-Supervised Learning). Lernt typische Bewegungsmuster und kann diese zur Vorhersage nutzen.

**Konfidenz**  
Geschätzte Zuverlässigkeit der interpolierten Werte. Abhängig von Lückengröße und Methode: PCHIP ~95%, Kalman ~90%, SSA ~85%, große Lücken entsprechend niedriger.

---

### Zonen

| Zone | Bereich | Bedeutung |
|------|---------|-----------|
| **Zone 11-6m** | 11m bis 6m vor dem Absprung | Anlaufphase. Datenqualität hier beeinflusst die Analyse, aber weniger kritisch. |
| **Zone 6-1m** | 6m bis 1m vor dem Absprung | Kritische Absprungvorbereitung. Datenlücken hier sind besonders problematisch für die Analyse. |

---

### Adaptive Strategie bei Kalman+SSA

Die Hybrid-Methode passt die Gewichtung je nach Lückengröße an:

| Lückengröße | Gewichtung | Konfidenz |
|-------------|------------|-----------|
| < 3m | 70% Kalman, 30% SSA | ~90% |
| 3-6m | 50% Kalman, 50% SSA | ~80% |
| > 6m | 40% Kalman, 60% SSA | ~70% |

Bei kleinen Lücken ist die physikalische Vorhersage (Kalman) genauer. Bei großen Lücken hilft das Schrittmuster (SSA), realistische Bewegungen zu erzeugen.
                """)
            
            # Cleaning/Gaps details table
            if interpolation_info and show_interpolation:
                st.markdown(f"### Interpolationsdetails ({interpolation_method})")
                cleaning_data = []
                for info in interpolation_info:
                    row = {
                        'Start': f"Index {info.get('start_idx', info.get('index', 0))}",
                        'Ende': f"Index {info.get('end_idx', 0)}",
                        'Größe (m)': f"{info.get('size_m', 0):.2f}",
                        'Punkte eingefügt': f"{info.get('num_points', 0)}",
                        'Konfidenz': f"{info.get('confidence', 0.8):.0%}",
                        'Methode': info.get('method', interpolation_method)
                    }
                    # Zeige entfernte Punkte für alle Methoden
                    row['Fehler entfernt'] = f"{info.get('removed_points', 0)}"
                    cleaning_data.append(row)
                
                st.dataframe(pd.DataFrame(cleaning_data), use_container_width=True, hide_index=True)
            elif gaps:
                st.markdown("### ⚠️ Erkannte Fehler/Lücken")
                gap_data = []
                for g in gaps:
                    zone = ''
                    if g.get('zone_6_1'):
                        zone = '🔴 6-1m (kritisch)'
                    elif g.get('zone_11_6'):
                        zone = '🟡 11-6m'
                    else:
                        zone = '-'
                    
                    row = {
                        'Index': g['index'],
                        'Start (m)': f"{g['start_value']/1000:.2f}",
                        'Ende (m)': f"{g['end_value']/1000:.2f}",
                        'Größe (m)': f"{g['difference']/1000:.2f}",
                        'Zone': zone
                    }
                    gap_data.append(row)
                
                st.dataframe(pd.DataFrame(gap_data), use_container_width=True, hide_index=True)
            else:
                st.success("✅ Keine Fehler oder Lücken detektiert!")
        
        except Exception as e:
            st.error(f"Fehler beim Laden der Datei: {e}")
            st.exception(e)

if __name__ == "__main__":
    main()

