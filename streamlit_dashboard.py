import streamlit as st
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d
from analyze_movement_data import MovementDataAnalyzer
from kalman_ssa_interpolator import KalmanSSAInterpolator
from data_cleaner import DataCleaner
import tempfile
import io
from typing import Tuple, List, Dict, Optional

# --- Page config (must be first Streamlit call) ---
st.set_page_config(
    page_title="OSP Anlaufanalyse",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded"
)

OSP_RED    = '#e30613'
OSP_BLACK  = '#000000'
OSP_WHITE  = '#ffffff'
OSP_GOLD   = '#ffd700'
OSP_GRAY   = '#f5f5f5'
OSP_LGRAY  = '#e8e8e8'
OSP_GREEN  = '#28a745'
OSP_YELLOW = '#ffc107'
OSP_PURPLE = '#a259d9'

st.markdown(f"""
<style>
    .main {{ padding: 0rem 1rem; }}
    .stApp {{ background-color: {OSP_WHITE}; }}
    h1 {{ color: {OSP_RED}; font-weight: bold; }}
    h2, h3, h4 {{ color: {OSP_BLACK}; }}
    .status-badge {{
        padding: 8px 20px; border-radius: 20px; font-weight: bold;
        color: white; display: inline-block; margin: 10px 0;
    }}
    .status-green  {{ background-color: {OSP_GREEN}; }}
    .status-yellow {{ background-color: {OSP_YELLOW}; color: black; }}
    .status-red    {{ background-color: {OSP_RED}; }}
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def parse_dat(filepath: str) -> Tuple[float, List[float], str, int]:
    """Liest eine .dat Laveg-Datei und gibt (takeoff_mm, data_mm, name, versuch) zurück."""
    for enc in ['latin1', 'utf-8']:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                lines = f.readlines()
            break
        except UnicodeDecodeError:
            continue

    athlete = lines[1].strip() if len(lines) > 1 else "Unbekannt"
    versuch = int(lines[2].strip()) if len(lines) > 2 and lines[2].strip().isdigit() else 0

    takeoff_str = lines[3].strip().replace(',', '.') if len(lines) > 3 else "0"
    if '.' in takeoff_str:
        takeoff_mm = float(takeoff_str.replace('.', ''))
    else:
        takeoff_mm = float(takeoff_str) if takeoff_str else 0.0

    data = []
    for line in lines[8:]:
        v = line.strip().replace(',', '.')
        if v:
            try:
                data.append(float(v))
            except ValueError:
                pass

    return takeoff_mm, data, athlete, versuch


def parse_uploaded(uploaded_file) -> Tuple[float, List[float], str, int]:
    """Wie parse_dat, aber aus einem Streamlit UploadedFile."""
    content = uploaded_file.read()
    for enc in ['latin1', 'utf-8']:
        try:
            text = content.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    lines = text.splitlines()

    athlete = lines[1].strip() if len(lines) > 1 else "Unbekannt"
    versuch = int(lines[2].strip()) if len(lines) > 2 and lines[2].strip().isdigit() else 0

    takeoff_str = lines[3].strip().replace(',', '.') if len(lines) > 3 else "0"
    if '.' in takeoff_str:
        takeoff_mm = float(takeoff_str.replace('.', ''))
    else:
        takeoff_mm = float(takeoff_str) if takeoff_str else 0.0

    data = []
    for line in lines[8:]:
        v = line.strip().replace(',', '.')
        if v:
            try:
                data.append(float(v))
            except ValueError:
                pass

    return takeoff_mm, data, athlete, versuch


def classify_gaps(gaps_info: List[Dict], takeoff_mm: float) -> Tuple[str, str]:
    """Gibt (status, qualität) basierend auf Zonen zurück."""
    zone_6_1 = [g for g in gaps_info
                if g['start_value'] >= takeoff_mm - 6000 and g['start_value'] < takeoff_mm - 1000]
    zone_11_6 = [g for g in gaps_info
                 if g['start_value'] >= takeoff_mm - 11000 and g['start_value'] < takeoff_mm - 6000]

    if zone_6_1:
        return 'rot', 'Kritisch'
    if zone_11_6:
        return 'gelb', 'Achtung'
    if gaps_info:
        return 'grün', 'OK'
    return 'grün', 'Sehr gut'


# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource
def load_analyzer():
    folders = [
        "Input files/Drei M",
        "Input files/Drei W",
        "Input files/Weit M",
        "Input files/Weit W",
    ]
    return MovementDataAnalyzer(folders)


@st.cache_resource
def load_interpolator():
    return KalmanSSAInterpolator(sampling_rate=50, ssa_window=40)


@st.cache_resource
def load_cleaner():
    return DataCleaner(max_backward_mm=500, min_forward_velocity=2.0, gap_threshold_mm=1000.0)


@st.cache_data
def load_file_list(_cleaner):
    """Lädt alle .dat-Dateien, erkennt Gaps mit DataCleaner, klassifiziert nach Zonen."""
    folders = [
        "Input files/Drei M",
        "Input files/Drei W",
        "Input files/Weit M",
        "Input files/Weit W",
    ]
    rows = []
    for folder in folders:
        if not os.path.exists(folder):
            continue
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith('.dat'):
                continue
            fpath = os.path.join(folder, fname)
            try:
                takeoff_mm, data, athlete, versuch = parse_dat(fpath)
                _, gaps_info, _ = _cleaner.clean_and_interpolate(data, sampling_rate=50)
                status, quality = classify_gaps(gaps_info, takeoff_mm)
                rows.append({
                    'filepath': fpath,
                    'filename': fname,
                    'folder': os.path.basename(folder),
                    'Athlet': athlete,
                    'Versuch': versuch,
                    'Lücken': len(gaps_info),
                    'Qualität': quality,
                    'status': status,
                    'takeoff_mm': takeoff_mm,
                })
            except Exception:
                continue
    return pd.DataFrame(rows)


# ── Interpolation pipeline ────────────────────────────────────────────────────

METHOD_OPTIONS = [
    "Automatisch (Kalman+SSA)",
    "PCHIP",
    "Kalman Filter",
    "SSA",
    "Kalman+SSA Hybrid",
    "Linear (Basis)",
    "Keine",
]

METHOD_KEY = {
    "Automatisch (Kalman+SSA)": "kalman_ssa",
    "PCHIP":                    "pchip",
    "Kalman Filter":            "kalman",
    "SSA":                      "ssa",
    "Kalman+SSA Hybrid":        "kalman_ssa",
    "Linear (Basis)":           "linear",
}

METHOD_LABEL = {
    "kalman_ssa": "Kalman+SSA Hybrid",
    "pchip":      "PCHIP (Monoton)",
    "kalman":     "Kalman Filter",
    "ssa":        "SSA (Muster)",
    "linear":     "Linear",
}

METHOD_CONF = {
    "pchip":      0.95,
    "kalman_ssa": 0.90,
    "kalman":     0.90,
    "ssa":        0.85,
    "linear":     0.80,
}


def run_interpolation(
    data: List[float],
    sampling_rate: int,
    method_name: str,
    cleaner: DataCleaner,
    interpolator: KalmanSSAInterpolator,
) -> Tuple[np.ndarray, List[Dict], List[Tuple[int, int]]]:
    """
    Führt DataCleaner + Interpolation durch.
    Gibt (interpolated_data, interpolation_info, ranges) zurück.
    """
    cleaned, gaps_info, _ = cleaner.clean_and_interpolate(data, sampling_rate=sampling_rate)
    interpolated = np.array(cleaned, dtype=float)
    interp_info = []
    ranges = []

    if method_name == "Keine" or not gaps_info:
        if not gaps_info:
            pass
        else:
            # Keine Methode — aber trotzdem gap-info für Anzeige
            for g in gaps_info:
                interp_info.append({
                    'start_idx': g['start_idx'], 'end_idx': g['end_idx'],
                    'size_mm': g['gap_size_mm'], 'size_m': g['gap_size_mm'] / 1000,
                    'num_points': g['end_idx'] - g['start_idx'],
                    'confidence': 0.0, 'method': 'Keine',
                    'removed_points': g.get('removed_points', 0),
                })
                ranges.append((g['start_idx'], g['end_idx']))
        return np.array(data, dtype=float), interp_info, ranges

    interp_key = METHOD_KEY.get(method_name, "kalman_ssa")

    for gap in gaps_info:
        si, ei = gap['start_idx'], gap['end_idx']
        gsm = abs(gap['end_value'] - gap['start_value'])
        n_pts = ei - si
        if n_pts <= 0:
            continue

        if interp_key == "linear":
            result = np.linspace(gap['start_value'], gap['end_value'], n_pts + 2)[1:-1]
            conf = METHOD_CONF["linear"]
        else:
            try:
                result, conf = interpolator.interpolate_gap(
                    interpolated,
                    max(0, si - 1),
                    min(ei, len(interpolated) - 1),
                    gsm,
                    method=interp_key,
                    num_points_override=n_pts,
                )
            except Exception:
                result = np.linspace(gap['start_value'], gap['end_value'], n_pts + 2)[1:-1]
                conf = METHOD_CONF["linear"]

        if len(result) == n_pts:
            interpolated[si:ei] = result
        elif len(result) > 1:
            f = interp1d(np.linspace(0, 1, len(result)), result, kind='linear')
            interpolated[si:ei] = f(np.linspace(0, 1, n_pts))

        interp_info.append({
            'start_idx': si, 'end_idx': ei,
            'size_mm': gsm, 'size_m': gsm / 1000,
            'num_points': n_pts, 'confidence': conf,
            'method': METHOD_LABEL.get(interp_key, interp_key),
            'removed_points': gap.get('removed_points', 0),
        })
        ranges.append((si, ei))

    # Finale Monotonie-Erzwingung
    for i in range(1, len(interpolated)):
        if interpolated[i] < interpolated[i - 1]:
            interpolated[i] = interpolated[i - 1]

    return interpolated, interp_info, ranges


# ── Plotting ──────────────────────────────────────────────────────────────────

def create_plot(
    analyzer: MovementDataAnalyzer,
    data: List[float],
    takeoff_mm: float,
    athlete: str,
    versuch: int,
    interpolated: np.ndarray,
    ranges: List[Tuple[int, int]],
    show_interp: bool,
) -> go.Figure:

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Distanz-Profil', 'Geschwindigkeits-Profil'),
        vertical_spacing=0.12,
        row_heights=[0.5, 0.5],
    )

    # Original (grau, gestrichelt) nur wenn Interpolation aktiv
    if show_interp:
        fig.add_trace(go.Scatter(
            y=[d / 1000 for d in data],
            name='Original (Rohdaten)',
            line=dict(color='lightgray', width=1, dash='dot'),
            opacity=0.5,
            hovertemplate='Frame %{x}: %{y:.3f} m<extra></extra>',
        ), row=1, col=1)

    plot_data = interpolated if show_interp else np.array(data)
    fig.add_trace(go.Scatter(
        y=[d / 1000 for d in plot_data],
        name='Bereinigt' if show_interp else 'Messdaten',
        line=dict(color=OSP_GREEN if show_interp else 'royalblue', width=2),
        hovertemplate='Frame %{x}: %{y:.3f} m<extra></extra>',
    ), row=1, col=1)

    # Interpolierte Bereiche hervorheben
    if show_interp and ranges:
        ix, iy = [], []
        for s, e in ranges:
            for i in range(s, min(e, len(plot_data))):
                ix.append(i)
                iy.append(plot_data[i] / 1000)
        if ix:
            fig.add_trace(go.Scatter(
                x=ix, y=iy, mode='markers',
                marker=dict(size=5, color=OSP_PURPLE, symbol='circle',
                            line=dict(width=1, color='white')),
                name='Interpolierte Punkte',
                hovertemplate='Frame %{x}: %{y:.3f} m (interp.)<extra></extra>',
            ), row=1, col=1)
        for s, e in ranges:
            y0 = plot_data[s] / 1000 if s < len(plot_data) else 0
            y1 = plot_data[min(e - 1, len(plot_data) - 1)] / 1000
            fig.add_shape(type="rect",
                x0=s, x1=e,
                y0=min(y0, y1) - 2, y1=max(y0, y1) + 2,
                fillcolor=OSP_PURPLE, opacity=0.12,
                line=dict(width=1, color=OSP_PURPLE, dash='dot'),
                row=1, col=1)
            fig.add_annotation(
                x=(s + e) / 2, y=max(y0, y1) + 3,
                text=f"↕ {abs(y1-y0):.1f}m interpoliert",
                showarrow=False, font=dict(size=9, color=OSP_PURPLE),
                row=1, col=1)

    # Absprung-Linie
    fig.add_hline(
        y=takeoff_mm / 1000, line_dash="dash",
        line_color=OSP_RED, line_width=2,
        annotation_text=f"Absprung ({takeoff_mm/1000:.2f} m)",
        annotation_position="right", row=1, col=1)

    # Zonen
    fig.add_hrect(
        y0=(takeoff_mm - 11000) / 1000, y1=(takeoff_mm - 6000) / 1000,
        fillcolor=OSP_GOLD, opacity=0.12,
        line_width=1, line_color=OSP_GOLD,
        annotation_text="Zone 11-6m", annotation_position="left",
        row=1, col=1)
    fig.add_hrect(
        y0=(takeoff_mm - 6000) / 1000, y1=(takeoff_mm - 1000) / 1000,
        fillcolor=OSP_RED, opacity=0.10,
        line_width=1, line_color=OSP_RED,
        annotation_text="Zone 6-1m (kritisch)", annotation_position="left",
        row=1, col=1)

    # Geschwindigkeit
    velocities = analyzer.calculate_velocity(list(plot_data))
    v = np.array(velocities, dtype=float)
    nans = np.isnan(v)
    if np.any(~nans):
        if np.any(nans):
            v[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), v[~nans])
        fig.add_trace(go.Scatter(
            y=v, name='Geschwindigkeit',
            line=dict(color='seagreen', width=2),
            hovertemplate='Frame %{x}: %{y:.2f} m/s<extra></extra>',
        ), row=2, col=1)
        if show_interp and ranges:
            for s, e in ranges:
                fig.add_vrect(x0=s, x1=e,
                    fillcolor=OSP_PURPLE, opacity=0.08, line_width=0, row=2, col=1)
        mean_v = float(np.nanmean(v))
        fig.add_hline(
            y=mean_v, line_dash="dash", line_color='firebrick', line_width=1,
            annotation_text=f"Ø {mean_v:.2f} m/s",
            annotation_position="right", row=2, col=1)

    fig.update_layout(
        title=dict(text=f'{athlete} — Versuch {versuch}',
                   font=dict(size=20, color=OSP_RED)),
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=700, hovermode='x unified',
        plot_bgcolor=OSP_WHITE, paper_bgcolor=OSP_WHITE,
        font=dict(family='Arial, sans-serif', size=12),
    )
    fig.update_xaxes(title_text='Messpunkt', gridcolor=OSP_LGRAY)
    fig.update_yaxes(title_text='Distanz (m)',        row=1, col=1, gridcolor=OSP_LGRAY)
    fig.update_yaxes(title_text='Geschwindigkeit (m/s)', row=2, col=1, gridcolor=OSP_LGRAY)
    return fig


# ── Quality metrics ───────────────────────────────────────────────────────────

def quality_metrics(analyzer, data, takeoff_mm, interp_data, ranges):
    try:
        arr = np.array(data, dtype=float)
        ti = np.searchsorted(arr, takeoff_mm)
        rel = arr[:ti] if ti > 0 else arr
        rel_i = np.array(interp_data[:ti], dtype=float) if ti > 0 else np.array(interp_data)

        if ranges and len(rel_i) > 1:
            smooth = max(0.0, 1 - np.mean(np.abs(np.diff(np.diff(rel_i)))) / 100)
            cont_scores = [abs(rel_i[e] - rel[e]) for s, e in ranges if e < len(rel)]
            cont = max(0.0, 1 - np.mean(cont_scores) / max(rel)) if cont_scores else 1.0
            iq = (smooth + cont) / 2
        else:
            iq = 1.0

        noise = float(np.std(np.diff(rel))) if len(rel) > 1 else 0.0
        gap_count = len(analyzer.check_for_gaps(list(rel), takeoff_mm))
        vels = [v for v in analyzer.calculate_velocity(list(rel)) if not np.isnan(v)]
        vel_score = max(0.0, 1 - min(np.std(vels) / 5, 1)) if vels else 0.0
        dq = (max(0.0, 1 - min(noise / 1000, 1)) + max(0.0, 1 - min(gap_count / 5, 1)) + vel_score) / 3

        sp = analyzer.analyze_step_pattern(list(rel))
        step_stab = max(0.0, 1 - min(sp['std_step_size'] / 100, 1))
        accs = np.diff(vels) if len(vels) > 1 else []
        acc_stab = max(0.0, 1 - min(np.std(accs) / 2, 1)) if len(accs) else 0.0
        ts = (step_stab + vel_score + acc_stab) / 3

        return {'iq': iq, 'dq': dq, 'ts': ts, 'noise': noise, 'gaps': gap_count}
    except Exception:
        return {'iq': 0.5, 'dq': 0.5, 'ts': 0.5, 'noise': 0.0, 'gaps': 0}


def zone_stats(analyzer, data, takeoff_mm, zone_far_mm, zone_near_mm):
    try:
        arr = np.array(data, dtype=float)
        s = np.searchsorted(arr, takeoff_mm - zone_far_mm)
        e = np.searchsorted(arr, takeoff_mm - zone_near_mm)
        zd = arr[s:e]
        if len(zd) < 2:
            return None
        vels = [v for v in analyzer.calculate_velocity(list(zd)) if not np.isnan(v)]
        if not vels:
            return None
        steps = np.diff(zd)
        return {
            'mean_v': float(np.mean(vels)),
            'std_v':  float(np.std(vels)),
            'mean_step': float(np.mean(steps)),
            'mean_acc': float(np.mean(np.diff(vels))) if len(vels) > 1 else 0.0,
        }
    except Exception:
        return None


# ── Auth ──────────────────────────────────────────────────────────────────────

def check_password() -> bool:
    def _entered():
        try:
            correct = st.secrets.get("OSP_DASHBOARD_PASSWORD", "OSP2024")
        except Exception:
            correct = os.environ.get("OSP_DASHBOARD_PASSWORD", "OSP2024")
        st.session_state["auth_ok"] = (st.session_state["pw"] == correct)

    if "auth_ok" not in st.session_state:
        st.title("OSP Anlaufanalyse Dashboard")
        st.markdown("---")
        st.subheader("Zugriff geschützt")
        st.text_input("Passwort", type="password", on_change=_entered, key="pw")
        st.info("Kontakt: Administrator für Zugangsdaten.")
        return False
    if not st.session_state["auth_ok"]:
        st.title("OSP Anlaufanalyse Dashboard")
        st.markdown("---")
        st.subheader("Zugriff geschützt")
        st.text_input("Passwort", type="password", on_change=_entered, key="pw")
        st.error("Falsches Passwort.")
        return False
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not check_password():
        return

    # Header
    c1, c2 = st.columns([1, 4])
    with c1:
        if os.path.exists("osp_logo.png"):
            st.image("osp_logo.png", width=180)
    with c2:
        st.title("Anlaufanalyse Dashboard")
        st.caption("OSP Hessen — Laveg-Laser Positionsdaten | Weit- und Dreisprung")
    st.markdown("---")

    # Resources
    try:
        analyzer    = load_analyzer()
        interpolator = load_interpolator()
        cleaner     = load_cleaner()
        df          = load_file_list(cleaner)
    except Exception as ex:
        st.error(f"Fehler beim Laden: {ex}")
        df = pd.DataFrame()

    # Batch-Upload: einzelne oder mehrere .dat-Dateien
    with st.expander("Dateien hochladen (einzeln oder Batch)", expanded=False):
        uploaded_files = st.file_uploader(
            "Laveg .dat Dateien wählen (Mehrfachauswahl möglich)",
            type=['dat'],
            accept_multiple_files=True,
            help="Alle Dateien werden lokal verarbeitet — keine Daten verlassen das System.",
        )
        if uploaded_files:
            st.caption(f"{len(uploaded_files)} Datei(en) geladen.")

    # Hochgeladene Dateien in DataFrame umwandeln und mit lokalen Dateien zusammenführen
    upload_rows = []
    upload_cache: Dict[str, Tuple] = {}  # filename → (takeoff_mm, data, athlete, versuch)

    for uf in (uploaded_files or []):
        try:
            uf.seek(0)
            t, d, a, v = parse_uploaded(uf)
            _, gaps_info, _ = cleaner.clean_and_interpolate(d, sampling_rate=50)
            status, quality = classify_gaps(gaps_info, t)
            upload_rows.append({
                'filepath': None, 'filename': uf.name,
                'folder': 'Upload',
                'Athlet': a, 'Versuch': v,
                'Lücken': len(gaps_info), 'Qualität': quality,
                'status': status, 'takeoff_mm': t,
            })
            upload_cache[uf.name] = (t, d, a, v)
        except Exception as ex:
            st.warning(f"Konnte '{uf.name}' nicht lesen: {ex}")

    # Alle Dateien (lokal + Upload) in einer Liste
    if upload_rows:
        upload_df = pd.DataFrame(upload_rows)
        all_df = pd.concat([df, upload_df], ignore_index=True) if len(df) > 0 else upload_df
    else:
        all_df = df

    # State
    takeoff_mm: Optional[float] = None
    data: Optional[List[float]] = None
    athlete: Optional[str] = None
    versuch: Optional[int] = None
    selected: Optional[Dict] = None

    # Layout
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("Versuche")
        if len(all_df) == 0:
            st.info("Keine Dateien gefunden. Dateien hochladen oder 'Input files' Ordner befüllen.")
            return

        with st.expander("Filter", expanded=False):
            q_opts = sorted(all_df['Qualität'].unique())
            f_opts = sorted(all_df['folder'].unique())
            q_filter = st.multiselect("Qualität", q_opts, q_opts)
            f_filter = st.multiselect("Disziplin / Quelle", f_opts, f_opts)

        fdf = all_df[all_df['Qualität'].isin(q_filter) & all_df['folder'].isin(f_filter)]
        if len(fdf) == 0:
            st.warning("Keine Versuche — Filter anpassen.")
            return

        def row_color(row):
            colors = {'Kritisch': '#f8d7da', 'Achtung': '#fff3cd', 'Sehr gut': '#d4edda'}
            bg = colors.get(row['Qualität'], '')
            return [f'background-color: {bg}'] * len(row)

        disp = fdf[['Athlet', 'Versuch', 'Lücken', 'Qualität', 'folder']].copy()
        disp = disp.rename(columns={'folder': 'Quelle'})
        event = st.dataframe(
            disp.style.apply(row_color, axis=1),
            use_container_width=True, height=500,
            hide_index=True, on_select="rerun", selection_mode="single-row",
        )
        sel_idx = event.selection.rows[0] if event.selection and event.selection.rows else 0
        selected = fdf.iloc[sel_idx].to_dict()
        st.caption(f"Ausgewählt: {selected['Athlet']} — Versuch {selected['Versuch']}")

    with col_right:
        hc1, hc2 = st.columns([2, 1])
        with hc1:
            st.subheader("Detailanalyse")
        with hc2:
            method = st.selectbox(
                "Interpolationsmethode",
                options=METHOD_OPTIONS,
                index=0,
                help="Alle Methoden bereinigen zuerst Messfehler (DataCleaner), dann interpolieren.",
            )

        try:
            # Daten laden: Upload-Cache oder lokale Datei
            if selected is None:
                st.info("Versuch auswählen.")
                return
            if selected['filepath'] is None:
                # Aus Upload-Cache
                if selected['filename'] not in upload_cache:
                    st.error("Upload-Datei nicht mehr im Cache — bitte erneut hochladen.")
                    return
                takeoff_mm, data, athlete, versuch = upload_cache[selected['filename']]
            else:
                takeoff_mm, data, athlete, versuch = parse_dat(selected['filepath'])

            sampling_rate = analyzer.sampling_rate or 50

            # Status Badge
            st.markdown("---")
            badge_map = {
                'rot':   ('status-red',    'KRITISCH'),
                'gelb':  ('status-yellow', 'ACHTUNG'),
                'grün':  ('status-green',  'GUT'),
            }
            badge_cls, badge_txt = badge_map.get(selected['status'], ('status-green', 'GUT'))
            st.markdown(
                f'<div class="status-badge {badge_cls}">{badge_txt}</div>',
                unsafe_allow_html=True,
            )
            st.caption(
                f"Athlet: {athlete} | Versuch: {versuch} | "
                f"Absprung: {takeoff_mm/1000:.3f} m | Sampling: {sampling_rate} Hz"
            )

            # Interpolation ausführen
            show_interp = (method != "Keine")
            interp_data, interp_info, ranges = run_interpolation(
                data, sampling_rate, method, cleaner, interpolator
            )
            plot_data = interp_data if show_interp else np.array(data)

            # Zusammenfassung
            if show_interp and interp_info:
                total_removed = sum(i.get('removed_points', 0) for i in interp_info)
                avg_conf = np.mean([i['confidence'] for i in interp_info])
                st.success(
                    f"**{method}**: {total_removed} Messfehler entfernt, "
                    f"{len(interp_info)} Region(en) interpoliert "
                    f"(Ø Konfidenz: {avg_conf:.0%})"
                )
            elif not interp_info:
                st.success("Keine Messfehler oder Lücken gefunden.")

            # Plot
            fig = create_plot(
                analyzer, data, takeoff_mm, athlete, versuch,
                plot_data, ranges, show_interp,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Metriken
            m = quality_metrics(analyzer, list(plot_data), takeoff_mm, plot_data, ranges)
            z11 = zone_stats(analyzer, list(plot_data), takeoff_mm, 11000, 6000)
            z6  = zone_stats(analyzer, list(plot_data), takeoff_mm, 6000, 1000)

            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.markdown("#### Qualität")
                st.metric("Interpolationsgüte", f"{m['iq']:.1%}")
                st.metric("Datenqualität",       f"{m['dq']:.1%}")
                st.metric("Stabilität",          f"{m['ts']:.1%}")
                st.metric("Rauschlevel",         f"{m['noise']:.1f} mm")
                st.metric("Lücken",              m['gaps'])
            with mc2:
                st.markdown("#### Zone 11–6 m")
                if z11:
                    st.metric("Ø Geschwindigkeit", f"{z11['mean_v']:.2f} m/s")
                    st.metric("Ø Schrittlänge",    f"{z11['mean_step']/1000:.3f} m")
                    st.metric("Ø Beschleunigung",  f"{z11['mean_acc']:.2f} m/s²")
                else:
                    st.info("Keine Daten in dieser Zone.")
            with mc3:
                st.markdown("#### Zone 6–1 m (kritisch)")
                if z6:
                    st.metric("Ø Geschwindigkeit", f"{z6['mean_v']:.2f} m/s")
                    st.metric("Ø Schrittlänge",    f"{z6['mean_step']/1000:.3f} m")
                    st.metric("Ø Beschleunigung",  f"{z6['mean_acc']:.2f} m/s²")
                else:
                    st.info("Keine Daten in dieser Zone.")

            # Detailtabelle
            if interp_info and show_interp:
                st.markdown(f"#### Interpolationsdetails")
                tbl = [{
                    'Start':           f"Index {i['start_idx']}",
                    'Ende':            f"Index {i['end_idx']}",
                    'Größe (m)':       f"{i['size_m']:.2f}",
                    'Punkte':          i['num_points'],
                    'Fehler entfernt': i['removed_points'],
                    'Konfidenz':       f"{i['confidence']:.0%}",
                    'Methode':         i['method'],
                } for i in interp_info]
                st.dataframe(pd.DataFrame(tbl), use_container_width=True, hide_index=True)

            # Legende
            with st.expander("Methoden & Begriffe"):
                st.markdown("""
#### Interpolationsmethoden

| Methode | Beschreibung | Konfidenz |
|---------|-------------|-----------|
| **Automatisch (Kalman+SSA)** | Kombiniert Kalman-Filter (Physik) mit SSA (Schrittmuster). Empfohlen für alle Fälle. | ~90% |
| **PCHIP** | Piecewise Cubic Hermite. Mathematisch optimal, garantiert monoton. | ~95% |
| **Kalman Filter** | Bidirektionale Zustandsschätzung (Position + Geschwindigkeit). | ~90% |
| **SSA** | Singular Spectrum Analysis. Extrahiert Schrittmuster aus dem Kontext. | ~85% |
| **Linear (Basis)** | Gerade Linie zwischen Start- und Endpunkt. Schnell, aber ohne Musterkenntnis. | ~80% |
| **Keine** | Originaldaten ohne Verarbeitung. Zum Vergleich. | — |

#### Zonen

| Zone | Bereich | Bedeutung |
|------|---------|-----------|
| 11–6 m | Vor Absprung | Anlaufphase. Lücken = Warnung (gelb). |
| 6–1 m | Vor Absprung | Kritische Phase. Lücken = Kritisch (rot). |

#### Gap-Erkennung (DataCleaner)

- **Rückwärtssprünge > 500 mm** unter bisherigem Maximum → Messfehler
- **Vorwärtssprünge > 1000 mm** in einem Frame → Messausfall (Laveg-Threshold)
- Benachbarte Fehlerbereiche (< 3 Frames) werden zusammengeführt
                """)

        except Exception as ex:
            st.error(f"Fehler: {ex}")
            st.exception(ex)


if __name__ == "__main__":
    main()
