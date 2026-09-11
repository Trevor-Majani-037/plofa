"""
PLOFA 26/27 — Streamlit Analytics Dashboard
============================================
Professional player statistics comparison app with:
- Player Comparison Matrix
- Pizza Plot Radar Charts
- Table Style Comparisons
"""

import json
import sys
import os
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mplsoccer import Radar
import streamlit as st

matplotlib.rcParams["font.family"] = "Consolas"
matplotlib.rcParams["font.size"] = 11
plt.rcParams["font.family"] = "Consolas"
plt.rcParams["font.size"] = 11

# ─────────────────────────────────────────────
# PATH SETUP
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, ".."))
STATS_PATH = os.path.join(BASE_DIR, "..", "season_stats.json")

# ─────────────────────────────────────────────
# PLOFA BRAND THEME
# ─────────────────────────────────────────────
PLOFA_COLORS = {
    "bg_dark": "#0D0D0D",
    "bg_card": "#1A1A2E",
    "bg_panel": "#16213E",
    "accent_gold": "#F5C518",
    "accent_teal": "#00B4D8",
    "accent_red": "#E63946",
    "accent_green": "#2DC653",
    "text_primary": "#F0F0F0",
    "text_muted": "#8A8A8A",
    "pitch_green": "#2D5016",
    "pitch_line": "#FFFFFF",
}

PLAYER_COLORS = [
    "#00B4D8", "#F5C518", "#E63946", "#2DC653", "#FF6B6B",
    "#4ECDC4", "#FFE66D", "#95E1D3", "#F38181", "#AA96DA",
]

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    with open(STATS_PATH, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    return data

def get_player_df(data):
    players = data.get("players", {})
    rows = []
    for name, pdata in players.items():
        info = pdata.get("info", {})
        totals = pdata.get("totals", {})
        row = {
            "player": name,
            "team": info.get("team", "?"),
            "position": info.get("position", "?"),
            "archetype": info.get("archetype", "?"),
            "age": info.get("age"),
            "matches": totals.get("matches_played", 0),
            "minutes": totals.get("minutes", 0),
        }
        row.update(totals)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df

# ─────────────────────────────────────────────
# STAT CATEGORIES
# ─────────────────────────────────────────────
STAT_GROUPS = {
    "Attacking": [
        "goals", "assists", "xg", "xa", "shots_on_target", "shots_blocked_att", "shots",
        "shot_conversion", "big_chances_scored", "big_chances_created",
        "open_play_goals", "headed_goals", "left_foot_goals", "right_foot_goals", "pen_goals",
    ],
    "Passing & Creation": [
        "passes_completed", "passes_attempted", "pass_accuracy",
        "progressive_passes", "through_balls_att", "key_passes",
        "chances_created", "deep_completions", "zone14_entries",
        "switches_of_play", "line_breaking_passes",
        "chipped_passes", "headed_passes",
        "passes_right_foot", "passes_left_foot", "passes_head",
    ],
    "Advanced (xT/PVA/EPA)": [
        "xT", "gpa", "pva", "epa",
    ],
    "Dribbling & Carrying": [
        "dribbles_comp", "dribbles_att", "dribble_success_pct",
        "carries", "progressive_carries", "carry_distance",
        "progressive_carry_distance", "longest_progressive_carry",
        "final_third_carries",
    ],
    "Defending": [
        "tackles_won", "tackles_att", "tackle_success_pct",
        "interceptions", "clearances", "blocks",
        "aerial_duels_won", "aerial_duels_att",
        "recoveries", "ball_recoveries", "pressures",
    ],
    "Physical": [
        "sprints", "high_speed_sprints", "distance_covered",
        "touches", "touches_final_third", "touches_opp_box",
        "offsides", "fouls_committed", "fouls_won",
    ],
    "Goalkeeping": [
        "saves", "goals_conceded", "save_pct", "clean_sheets",
        "goals_prevented", "xgot_faced",
        "high_claims", "punches", "goalline_saves",
    ],
    "Overall": [
        "goals", "assists", "xg", "xa", "avg_rating",
        "pass_accuracy", "dribble_success_pct", "tackle_success_pct",
        "minutes", "matches",
    ],
}

PER90_STATS = [
    "goals_per90", "assists_per90", "xg_per90", "xa_per90",
    "tackles_won_per90", "interceptions_per90", "clearances_per90",
    "carries_per90", "progressive_carries_per90", "dribbles_comp_per90",
    "sprints_per90", "touches_per90", "passes_completed_per90",
    "recoveries_per90", "pressures_per90", "sca_per90", "gca_per90",
    "xT_per90", "gpa_per90", "pva_per90", "epa_per90",
]

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def safe_num(val):
    if val is None:
        return 0.0
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0

def normalize_stat(val, min_val, max_val):
    if max_val == min_val:
        return 50.0
    return max(0.0, min(100.0, ((val - min_val) / (max_val - min_val)) * 100.0))

def get_player_stats(df, player_name):
    row = df[df["player"] == player_name]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()

def build_comparison_matrix(df, players, stats):
    """Build a matrix DataFrame: rows=stats, cols=players."""
    data = {"Stat": []}
    for p in players:
        data[p] = []
    for stat in stats:
        data["Stat"].append(stat)
        vals = df[df["player"].isin(players)][stat].fillna(0)
        min_v = vals.min()
        max_v = vals.max() if vals.max() > min_v else min_v + 1
        for p in players:
            pval = safe_num(get_player_stats(df, p).get(stat, 0))
            data[p].append(pval)
    return pd.DataFrame(data)

def build_normalized_matrix(df, players, stats):
    """Build normalized matrix for heatmap (0-100 scale)."""
    data = {"Stat": []}
    for p in players:
        data[p] = []
    for stat in stats:
        data["Stat"].append(stat)
        vals = df[df["player"].isin(players)][stat].fillna(0)
        min_v = vals.min()
        max_v = vals.max() if vals.max() > min_v else min_v + 1
        for p in players:
            pval = safe_num(get_player_stats(df, p).get(stat, 0))
            norm = normalize_stat(pval, min_v, max_v)
            data[p].append(norm)
    return pd.DataFrame(data)

# ─────────────────────────────────────────────
# STYLING HELPERS
# ─────────────────────────────────────────────
def color_val(val, best_val, worst_val):
    """Return a color string based on value relative to best/worst."""
    if best_val == worst_val:
        return "background-color: transparent;"
    ratio = (val - worst_val) / (best_val - worst_val)
    if ratio >= 0.8:
        return "background-color: rgba(46, 198, 83, 0.3);"
    elif ratio >= 0.5:
        return "background-color: rgba(245, 197, 24, 0.2);"
    elif ratio <= 0.2:
        return "background-color: rgba(230, 57, 70, 0.2);"
    return "background-color: transparent;"

def style_table_cell(val, best_val, worst_val):
    if pd.isna(val):
        return ""
    if isinstance(val, str):
        return ""
    return color_val(val, best_val, worst_val)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PLOFA Analytics Hub",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown(f"""
<style>
    :root {{
        --plofa-bg: {PLOFA_COLORS['bg_dark']};
        --plofa-card: {PLOFA_COLORS['bg_card']};
        --plofa-panel: {PLOFA_COLORS['bg_panel']};
        --plofa-gold: {PLOFA_COLORS['accent_gold']};
        --plofa-teal: {PLOFA_COLORS['accent_teal']};
        --plofa-red: {PLOFA_COLORS['accent_red']};
        --plofa-green: {PLOFA_COLORS['accent_green']};
        --plofa-text: {PLOFA_COLORS['text_primary']};
        --plofa-muted: {PLOFA_COLORS['text_muted']};
    }}

    .stApp {{
        background-color: var(--plofa-bg);
        color: var(--plofa-text);
        font-family: 'Consolas', 'Courier New', monospace !important;
        font-size: 11px !important;
    }}

    body, div, span, p, h1, h2, h3, h4, h5, h6,
    .stMarkdown, .stText, .stSelectbox, .stMultiSelect,
    .stMetric, .stDataFrame, .stTable, .stTabs,
    [data-baseweb="tab"], [data-baseweb="select"],
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"],
    [data-testid="stSelectbox"], [data-testid="stMultiSelect"],
    [data-testid="stDataFrame"], [data-testid="data-testid"],
    button, input, textarea, select, option {{
        font-family: 'Consolas', 'Courier New', monospace !important;
        font-size: 11px !important;
    }}

    .main-header {{
        font-size: 2.5rem;
        font-weight: 900;
        background: linear-gradient(90deg, {PLOFA_COLORS['accent_gold']}, {PLOFA_COLORS['accent_teal']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }}

    .sub-header {{
        text-align: center;
        color: var(--plofa-muted);
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: var(--plofa-card);
        padding: 8px 12px 0;
        border-radius: 12px 12px 0 0;
    }}

    .stTabs [data-baseweb="tab"] {{
        background-color: var(--plofa-panel);
        color: var(--plofa-muted);
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
        border: 1px solid transparent;
        transition: all 0.2s;
    }}

    .stTabs [aria-selected="true"] {{
        background-color: var(--plofa-card);
        color: var(--plofa-gold);
        border-color: var(--plofa-gold);
    }}

    .stMetric {{
        background-color: var(--plofa-card);
        border: 1px solid rgba(245, 197, 24, 0.2);
        border-radius: 10px;
        padding: 10px;
    }}

    div[data-testid="stMetricValue"] {{
        color: var(--plofa-gold) !important;
        font-weight: 800;
    }}

    div[data-testid="stMetricLabel"] {{
        color: var(--plofa-muted) !important;
        font-size: 0.85rem;
    }}

    .stDataFrame {{
        border-radius: 10px;
        overflow: hidden;
    }}

    div[data-testid="stSelectbox"] label,
    div[data-testid="stMultiSelect"] label {{
        color: var(--plofa-text) !important;
        font-weight: 600;
        font-size: 0.9rem;
    }}

    .pizza-container {{
        display: flex;
        flex-wrap: wrap;
        gap: 20px;
        justify-content: center;
    }}

    .player-card {{
        background-color: var(--plofa-card);
        border: 1px solid rgba(245, 197, 24, 0.15);
        border-radius: 14px;
        padding: 16px;
        text-align: center;
        min-width: 280px;
        flex: 1 1 280px;
        max-width: 420px;
    }}

    .player-name {{
        font-size: 1.3rem;
        font-weight: 800;
        color: var(--plofa-gold);
        margin-bottom: 4px;
    }}

    .player-meta {{
        font-size: 0.85rem;
        color: var(--plofa-muted);
        margin-bottom: 12px;
    }}

    .stat-badge {{
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 2px;
    }}

    hr {{
        border-color: rgba(245, 197, 24, 0.15);
        margin: 1.5rem 0;
    }}

    .section-title {{
        font-size: 1.5rem;
        font-weight: 800;
        color: var(--plofa-text);
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--plofa-gold);
        display: inline-block;
    }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
def render_sidebar(df):
    st.sidebar.markdown("## ⚙️ Controls")

    # Team filter
    teams = sorted(df["team"].unique())
    selected_team = st.sidebar.selectbox("Filter by Team", ["All Teams"] + teams)

    if selected_team != "All Teams":
        df = df[df["team"] == selected_team]

    # Position filter
    positions = sorted(df["position"].unique())
    selected_pos = st.sidebar.multiselect(
        "Filter by Position",
        positions,
        default=positions,
    )
    if selected_pos:
        df = df[df["position"].isin(selected_pos)]

    # Min matches filter
    min_matches = st.sidebar.slider(
        "Minimum Matches Played",
        min_value=0,
        max_value=int(df["matches"].max()) if not df.empty else 0,
        value=0,
    )
    if min_matches > 0:
        df = df[df["matches"] >= min_matches]

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Player Comparison")
    player_options = sorted(df["player"].unique())
    selected_players = st.sidebar.multiselect(
        "Select Players to Compare (max 5)",
        player_options,
        max_selections=5,
    )

    stat_category = st.sidebar.selectbox(
        "Stat Category",
        list(STAT_GROUPS.keys()),
        index=0,
    )

    return df, selected_players, stat_category

# ─────────────────────────────────────────────
# TAB 1: COMPARISON MATRIX
# ─────────────────────────────────────────────
def render_comparison_matrix(df, players, stat_category):
    st.markdown('<p class="section-title">📊 Player Comparison Matrix</p>', unsafe_allow_html=True)

    if len(players) < 2:
        st.info("👈 Select 2+ players from the sidebar to compare.")
        return

    # ── Available stat pools ────────────────────────────────────────
    total_pool = {
        "Games": "matches_played",
        "Minutes": "minutes",
        "Goals": "goals",
        "Assists": "assists",
        "Passes Attempted": "passes_attempted",
        "Passes Completed": "passes_completed",
        "Tackles Won": "tackles_won",
        "Interceptions": "interceptions",
        "Clearances": "clearances",
        "Aerial Duels Won": "aerial_duels_won",
        "Sprints": "sprints",
        "Distance Covered": "distance_covered",
        "Shots On Target": "shots_on_target",
        "Big Chances Created": "big_chances_created",
        "Through Balls": "through_balls_att",
        "Crosses": "crosses_open_play_att",
        "Dribbles": "dribbles_att",
        "Progressive Carries": "progressive_carries",
        "Progressive Carry Distance": "progressive_carry_distance",
        "Longest Progressive Carry": "longest_progressive_carry",
        "Pressures": "pressures",
        "Ball Recoveries": "recoveries",
        "Fouls Committed": "fouls_committed",
        "Yellow Cards": "yellow_cards",
        "Red Cards": "red_cards",
        "Saves": "saves",
        "Clean Sheets": "clean_sheets",
        "Goals Conceded": "goals_conceded",
        "Goals Prevented": "goals_prevented",
        "xGOT Faced": "xgot_faced",
        "xA": "xa",
        "xG": "xg",
        "Chipped Passes": "chipped_passes",
        "Headed Passes": "headed_passes",
        "Left Foot Goals": "left_foot_goals",
        "Right Foot Goals": "right_foot_goals",
        "Headed Goals": "headed_goals",
    }

    per90_pool = {
        "Goals/90": "goals_per90",
        "Assists/90": "assists_per90",
        "Passes/90": "passes_completed_per90",
        "Tackles/90": "tackles_won_per90",
        "Interceptions/90": "interceptions_per90",
        "Clearances/90": "clearances_per90",
        "Aerial/90": "aerial_duels_won_per90",
        "Sprints/90": "sprints_per90",
        "Distance/90": "distance_covered_per90",
        "Shots OT/90": "shots_on_target_per90",
        "Through Balls/90": "through_balls_att_per90",
        "Dribbles/90": "dribbles_att_per90",
        "Pressures/90": "pressures_per90",
        "Recoveries/90": "recoveries_per90",
        "xA/90": "xa_per90",
        "xG/90": "xg_per90",
        "SCA/90": "sca_per90",
    }

    # ── Total / Per 90 toggle ───────────────────────────────────────
    c1, c2, c3 = st.columns([1, 1, 3])
    with c1:
        mode_total = st.toggle("Total", value=True, key="cmp_mode_total")
    with c2:
        mode_per90 = st.toggle("Per 90", value=False, key="cmp_mode_per90")

    if mode_total and mode_per90:
        mode_per90 = False
        st.rerun()

    active_pool = per90_pool if mode_per90 else total_pool

    # Build available options with readable names
    available_stats = []
    for label, key in active_pool.items():
        if key in df.columns:
            available_stats.append((label, key))

    if not available_stats:
        st.warning("No stats available for this mode.")
        return

    # ── Stat selector ───────────────────────────────────────────────
    default_labels = [label for label, key in available_stats[:12]]
    selected_labels = st.multiselect(
        "Choose stats to compare",
        [label for label, _ in available_stats],
        default=default_labels,
        key="cmp_stat_selector",
    )

    if not selected_labels:
        st.info("Pick at least one stat.")
        return

    label_to_key = {label: key for label, key in available_stats}
    selected_keys = [label_to_key[l] for l in selected_labels]

    # ── Player header cards ─────────────────────────────────────────
    card_cols = st.columns(len(players))
    purple_dark = "#2D1B4E"
    purple_light = "#7C3AED"
    purple_bg = "#1E1030"

    for idx, p in enumerate(players):
        with card_cols[idx]:
            pstats = get_player_stats(df, p)
            team = pstats.get("team", "?")
            pos = pstats.get("position", "?")
            matches = pstats.get("matches_played", 0)
            mins = pstats.get("minutes", 0)
            goals = pstats.get("goals", 0)
            assists = pstats.get("assists", 0)

            st.markdown(f"""
            <div style="
                background: linear-gradient(180deg, {purple_light}22, {purple_bg});
                border: 2px solid {purple_light};
                border-radius: 12px;
                padding: 14px;
                text-align: center;
            ">
                <div style="
                    width: 52px; height: 52px; border-radius: 50%;
                    background: {purple_light};
                    margin: 0 auto 8px;
                    display: flex; align-items: center; justify-content: center;
                    font-size: 1.3rem; font-weight: 900; color: #fff;
                ">{p[0]}</div>
                <div style="font-size: 1rem; font-weight: 800; color: #F5C518;">{p}</div>
                <div style="color: #8A8A8A; font-size: 0.8rem;">{team} • {pos}</div>
                <div style="margin-top: 8px; display: flex; gap: 6px; justify-content: center;">
                    <span style="background:{PLOFA_COLORS['accent_teal']}22; color:{PLOFA_COLORS['accent_teal']}; padding: 3px 8px; border-radius: 10px; font-size: 0.75rem; font-weight: bold;">
                        ⚽ {goals}
                    </span>
                    <span style="background:{PLOFA_COLORS['accent_gold']}22; color:{PLOFA_COLORS['accent_gold']}; padding: 3px 8px; border-radius: 10px; font-size: 0.75rem; font-weight: bold;">
                        🅰️ {assists}
                    </span>
                </div>
                <div style="color: #8A8A8A; font-size: 0.75rem; margin-top: 6px;">
                    {matches} matches • {mins} mins
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # ── Stat bars ───────────────────────────────────────────────────
    invert_stats = {
        "goals_conceded", "red_cards", "yellow_cards", "offsides",
        "fouls_committed", "turnovers", "bad_touches", "dispossessed",
    }

    bar_container = st.container()

    for stat_label in selected_labels:
        stat_key = label_to_key[stat_label]
        values = {}
        for p in players:
            values[p] = safe_num(get_player_stats(df, p).get(stat_key, 0))

        valid_vals = [v for v in values.values() if v > 0]
        max_val = max(valid_vals) if valid_vals else 1.0
        if max_val == 0:
            max_val = 1.0

        is_invert = stat_key in invert_stats
        if is_invert:
            best_val = min(values.values())
        else:
            best_val = max(values.values())

        winners = {p for p, v in values.items() if abs(v - best_val) < 1e-6}

        # Stat label row
        label_col, *player_cols = st.columns([2] + [3] * len(players))
        with label_col:
            st.markdown(f"**{stat_label}**")

        for p_idx, p in enumerate(players):
            with player_cols[p_idx]:
                v = values[p]
                bar_pct = max((v / max_val) * 100, 1.0) if max_val > 0 else 0.0
                is_winner = p in winners
                bar_color = PLOFA_COLORS["accent_green"] if is_winner else PLOFA_COLORS["accent_red"]
                track_color = "#333355"

                st.markdown(f"""
                <div style="
                    background: {track_color};
                    border-radius: 4px;
                    height: 14px;
                    width: 100%;
                    position: relative;
                    overflow: hidden;
                ">
                    <div style="
                        background: {bar_color};
                        height: 100%;
                        width: {bar_pct:.1f}%;
                        border-radius: 4px;
                        transition: width 0.3s;
                    "></div>
                </div>
                <div style="
                    font-size: 0.8rem;
                    font-weight: bold;
                    color: {bar_color};
                    margin-top: 2px;
                    text-align: right;
                ">{v:.0f}</div>
                """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TAB 2: PIZZA PLOTS
# ─────────────────────────────────────────────
def draw_pizza_plot(fig, ax, stat_labels, values, color, bg_colors):
    """Draw a single pizza plot (coxcomb / rose chart) using Wedge patches."""
    from matplotlib.patches import Wedge

    n = len(stat_labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    bar_width = 2 * np.pi / n * 0.75

    bg_dark, bg_panel, bg_card, text_primary, text_muted, accent_gold = bg_colors

    # Concentric reference circles (25, 50, 75, 100)
    for pct in [25, 50, 75, 100]:
        theta = np.linspace(0, 2 * np.pi, 200)
        x = pct * np.cos(theta)
        y = pct * np.sin(theta)
        ax.plot(x, y, color="#333355", linewidth=0.8, linestyle="--", alpha=0.6, zorder=1)

    # Wedges for each stat
    for angle, val in zip(angles, values):
        theta1 = np.degrees(angle - bar_width / 2)
        theta2 = np.degrees(angle + bar_width / 2)
        wedge = Wedge(
            (0, 0),
            val,
            theta1,
            theta2,
            color=color,
            alpha=0.82,
            zorder=3,
        )
        ax.add_patch(wedge)

    # Percentile labels: exact value on the slice, stat name outside
    label_radius = 108
    for i, (angle, label, val) in enumerate(zip(angles, stat_labels, values)):
        ha = "left" if np.cos(angle) >= 0 else "right"
        lx = np.cos(angle) * label_radius
        ly = np.sin(angle) * label_radius

        ax.text(
            lx,
            ly,
            f"{val:.0f}",
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color=accent_gold,
            zorder=5,
        )

        ax.text(
            lx + np.cos(angle) * 10,
            ly + np.sin(angle) * 10,
            label,
            ha=ha,
            va="center",
            fontsize=7.5,
            fontweight="600",
            color=text_primary,
            zorder=5,
        )

    ax.set_xlim(-135, 135)
    ax.set_ylim(-135, 135)
    ax.set_aspect("equal")
    ax.axis("off")


def render_pizza_plots(df, players, stat_category):
    st.markdown('<p class="section-title">🍕 Pizza Plot Comparison</p>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color: var(--plofa-muted); margin-bottom: 1rem;'>"
        "Opta-style pizza plots: percentile 0–100 per stat vs all players in the filtered dataset. "
        "Values shown on each slice."
        "</p>",
        unsafe_allow_html=True,
    )

    if len(players) < 1:
        st.info("👈 Select 1+ players from the sidebar to view pizza plots.")
        return

    pizza_groups = {
        "Attacking": ["goals", "assists", "xg", "xa", "shots_on_target", "shot_conversion", "big_chances_created"],
        "Passing & Creation": ["pass_accuracy", "progressive_passes", "through_balls_att", "deep_completions", "chances_created"],
        "Dribbling & Carrying": ["dribbles_comp", "dribble_success_pct", "progressive_carries", "progressive_carry_distance", "longest_progressive_carry", "carries"],
        "Defending": ["tackles_won", "interceptions", "clearances", "aerial_duels_won", "recoveries"],
        "Physical": ["sprints", "distance_covered", "touches", "fouls_won"],
        "Goalkeeping": ["saves", "save_pct", "clean_sheets", "goals_conceded", "high_claims", "goals_prevented"],
        "Overall": ["goals", "assists", "xg", "xa", "avg_rating", "pass_accuracy", "dribble_success_pct", "tackle_success_pct"],
    }

    group = stat_category if stat_category in pizza_groups else "Attacking"
    pizza_stats = pizza_groups.get(group, pizza_groups["Attacking"])
    pizza_stats = [s for s in pizza_stats if s in df.columns]

    if not pizza_stats:
        st.warning("No pizza stats available for this category.")
        return

    label_names = [s.replace("_", " ").title() for s in pizza_stats]
    sub_all = df[pizza_stats].fillna(0)
    mins = sub_all.min()
    maxs = sub_all.max()
    maxs = maxs.replace(0, 1.0)

    def to_percentile(player_name):
        pstats = get_player_stats(df, player_name)
        vals = []
        for s in pizza_stats:
            v = safe_num(pstats.get(s, 0))
            pct = ((v - mins[s]) / (maxs[s] - mins[s])) * 100
            vals.append(max(0.0, min(100.0, pct)))
        return vals

    bg_colors = (
        PLOFA_COLORS["bg_dark"],
        PLOFA_COLORS["bg_panel"],
        PLOFA_COLORS["bg_card"],
        PLOFA_COLORS["text_primary"],
        PLOFA_COLORS["text_muted"],
        PLOFA_COLORS["accent_gold"],
    )

    # ── Side-by-side pizza plots ──────────────────────────────────
    if len(players) == 1:
        cols = st.columns(1)
    elif len(players) == 2:
        cols = st.columns(2)
    else:
        cols = st.columns(min(len(players), 3))

    player_pct = {}

    for idx, p in enumerate(players):
        col = cols[idx % len(cols)]
        with col:
            pstats = get_player_stats(df, p)
            team = pstats.get("team", "?")
            pos = pstats.get("position", "?")
            values = to_percentile(p)
            player_pct[p] = dict(zip(pizza_stats, values))
            color = PLAYER_COLORS[idx % len(PLAYER_COLORS)]

            fig, ax = plt.subplots(figsize=(6, 6))
            fig.patch.set_facecolor(PLOFA_COLORS["bg_dark"])
            ax.set_facecolor(PLOFA_COLORS["bg_panel"])

            draw_pizza_plot(fig, ax, label_names, values, color, bg_colors)

            ax.set_title(
                f"{p}\n{team} • {pos}",
                color=PLOFA_COLORS["accent_gold"],
                fontsize=12,
                fontweight="bold",
                pad=20,
            )

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=PLOFA_COLORS["bg_dark"])
            buf.seek(0)
            st.image(buf, width="stretch")
            plt.close(fig)

    # ── Opta-style percentile table ───────────────────────────────
    if len(players) >= 2:
        st.markdown("#### 📊 Percentile Comparison Table")

        table_rows = []
        for s in pizza_stats:
            row = {"Stat": s.replace("_", " ").title()}
            for p in players:
                v = player_pct[p][s]
                row[p] = f"{v:.0f}"
            table_rows.append(row)

        tbl_df = pd.DataFrame(table_rows)
        display_cols = ["Stat"] + players

        def highlight_pct_row(row):
            styles = [""] * len(row)
            vals = []
            for p in players:
                try:
                    vals.append(float(row[p]))
                except (ValueError, TypeError):
                    vals.append(-1.0)
            if vals:
                best_idx = int(np.argmax(vals))
                for i, p in enumerate(players):
                    ci = display_cols.index(p)
                    if i == best_idx:
                        styles[ci] = f"background-color: rgba(46, 198, 83, 0.25); font-weight: bold; color: {PLOFA_COLORS['text_primary']};"
                    else:
                        styles[ci] = f"color: {PLOFA_COLORS['text_primary']};"
            return styles

        styled_tbl = (
            tbl_df[display_cols]
            .style
            .apply(highlight_pct_row, axis=1)
            .set_table_styles([
                {"selector": "th", "props": [
                    ("background-color", PLOFA_COLORS["bg_panel"]),
                    ("color", PLOFA_COLORS["accent_gold"]),
                    ("font-weight", "800"),
                    ("font-size", "0.85rem"),
                    ("padding", "10px"),
                ]},
                {"selector": "td", "props": [
                    ("padding", "10px"),
                    ("font-size", "0.9rem"),
                ]},
            ])
        )

        st.dataframe(styled_tbl, use_container_width=True, hide_index=True)

        # Raw values
        with st.expander("🔢 Raw Values"):
            raw_rows = []
            for s in pizza_stats:
                row = {"Stat": s.replace("_", " ").title()}
                for p in players:
                    pstats = get_player_stats(df, p)
                    row[p] = safe_num(pstats.get(s, 0))
                raw_rows.append(row)
            raw_df = pd.DataFrame(raw_rows)
            raw_display = raw_df.copy()
            for p in players:
                raw_display[p] = raw_display[p].apply(
                    lambda x: f"{x:.0f}" if isinstance(x, float) else str(int(x)) if isinstance(x, (int, float)) else str(x)
                )
            st.dataframe(
                raw_display.style.set_table_styles([
                    {"selector": "th", "props": [
                        ("background-color", PLOFA_COLORS["bg_panel"]),
                        ("color", PLOFA_COLORS["accent_gold"]),
                        ("font-weight", "800"),
                    ]},
                ]),
                use_container_width=True,
                hide_index=True,
            )

# ─────────────────────────────────────────────
# TAB 3: TABLE COMPARISON
# ─────────────────────────────────────────────
def render_table_comparison(df, players, stat_category):
    st.markdown('<p class="section-title">📋 Detailed Table Comparison</p>', unsafe_allow_html=True)

    if len(players) < 1:
        st.info("👈 Select 1+ players from the sidebar.")
        return

    stats = STAT_GROUPS[stat_category]
    valid_stats = [s for s in stats if s in df.columns]

    if not valid_stats:
        st.warning(f"No stats available for category: {stat_category}")
        return

    # Build table
    table_data = {"Stat": []}
    for p in players:
        table_data[p] = []
        table_data[f"{p}_rank"] = []

    for stat in valid_stats:
        table_data["Stat"].append(stat.replace("_", " ").title())
        vals = []
        for p in players:
            v = safe_num(get_player_stats(df, p).get(stat, 0))
            vals.append(v)
            table_data[p].append(v)

        # Rank (higher is better — for defensive stats like goals_conceded, invert)
        invert = stat in ["goals_conceded", "red_cards", "yellow_cards", "offsides", "fouls_committed", "turnovers", "bad_touches", "dispossessed"]
        if invert:
            ranked = sorted(range(len(vals)), key=lambda i: vals[i])
        else:
            ranked = sorted(range(len(vals)), key=lambda i: vals[i], reverse=True)
        rank_map = {idx: i + 1 for i, idx in enumerate(ranked)}
        for i in range(len(players)):
            table_data[f"{players[i]}_rank"].append(rank_map[i])

    table_df = pd.DataFrame(table_data)

    # Format display
    display_df = table_df.copy()
    for p in players:
        display_df[p] = display_df[p].apply(
            lambda x: f"{x:.0f}" if isinstance(x, float) else str(int(x)) if isinstance(x, (int, float)) else str(x)
        )
        display_df[p] = display_df[p] + " " + display_df[f"{p}_rank"].apply(
            lambda r: f"({'🥇' if r==1 else '🥈' if r==2 else '🥉' if r==3 else ''})"
        )

    display_df = display_df[[col for col in display_df.columns if "_rank" not in col]]

    # Style
    def highlight_row(row):
        styles = [""] * len(row)
        stat_name = row["Stat"].lower().replace(" ", "_")
        invert = stat_name in ["goals_conceded", "red_cards", "yellow_cards", "offsides", "fouls_committed", "turnovers", "bad_touches", "dispossessed"]

        vals = []
        for p in players:
            try:
                vals.append(float(row[p].split()[0]))
            except (ValueError, IndexError):
                vals.append(0)

        if invert:
            best_idx = vals.index(min(vals)) if any(v > 0 for v in vals) else -1
            worst_idx = vals.index(max(vals)) if any(v > 0 for v in vals) else -1
        else:
            best_idx = vals.index(max(vals)) if any(v > 0 for v in vals) else -1
            worst_idx = vals.index(min(vals)) if any(v > 0 for v in vals) else -1

        for i, p in enumerate(players):
            col_idx = display_df.columns.get_loc(p)
            if i == best_idx and best_idx >= 0:
                styles[col_idx] = f"background-color: rgba(46, 198, 83, 0.25); font-weight: bold;"
            elif i == worst_idx and worst_idx >= 0 and len(players) > 1:
                styles[col_idx] = f"background-color: rgba(230, 57, 70, 0.15);"
        return styles

    styled = display_df.style.apply(highlight_row, axis=1).set_table_styles([
        {"selector": "th", "props": [
            ("background-color", PLOFA_COLORS["bg_panel"]),
            ("color", PLOFA_COLORS["accent_gold"]),
            ("font-weight", "800"),
            ("font-size", "0.85rem"),
            ("padding", "10px"),
        ]},
        {"selector": "td", "props": [
            ("padding", "10px"),
            ("font-size", "0.85rem"),
        ]},
        {"selector": "tr:hover", "props": [
            ("background-color", "rgba(245, 197, 24, 0.05)"),
        ]},
    ])

    st.dataframe(
        styled,
        use_container_width=True,
        height=min(700, len(valid_stats) * 42 + 50),
        hide_index=True,
    )

    # Per-90 toggle
    st.markdown("#### ⏱️ Per 90 Minutes Comparison")
    per90_stats = [s for s in valid_stats if f"{s}_per90" in df.columns]
    if per90_stats:
        per90_data = {"Stat": []}
        for p in players:
            per90_data[p] = []
        for s in per90_stats:
            per90_data["Stat"].append(s.replace("_per90", "").replace("_", " ").title())
            for p in players:
                v = safe_num(get_player_stats(df, p).get(f"{s}_per90", 0))
                per90_data[p].append(v)
        per90_df = pd.DataFrame(per90_data)
        for p in players:
            per90_df[p] = per90_df[p].apply(lambda x: f"{x:.3f}")

        st.dataframe(
            per90_df.style.background_gradient(cmap="RdYlGn", subset=players, vmin=0, vmax=100)
            .set_table_styles([
                {"selector": "th", "props": [
                    ("background-color", PLOFA_COLORS["bg_panel"]),
                    ("color", PLOFA_COLORS["accent_gold"]),
                    ("font-weight", "800"),
                ]},
            ]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("No per-90 data available for selected stats.")

# ─────────────────────────────────────────────
# TAB 4: PLAYER PROFILES
# ─────────────────────────────────────────────
def render_player_profiles(df, players):
    st.markdown('<p class="section-title">🎯 Player Profile Cards</p>', unsafe_allow_html=True)

    if not players:
        st.info("👈 Select players from the sidebar.")
        return

    if len(players) <= 3:
        cols = st.columns(len(players))
    else:
        cols = st.columns(4)

    for idx, p in enumerate(players):
        col = cols[idx % 4]
        with col:
            pstats = get_player_stats(df, p)
            team = pstats.get("team", "?")
            pos = pstats.get("position", "?")
            matches = pstats.get("matches_played", 0)
            mins = pstats.get("minutes", 0)
            rating = pstats.get("avg_rating", 0)
            goals = pstats.get("goals", 0)
            assists = pstats.get("assists", 0)

            color = PLAYER_COLORS[idx % len(PLAYER_COLORS)]

            st.markdown(f"""
            <div style="
                background: {PLOFA_COLORS['bg_card']};
                border: 1px solid {color}33;
                border-radius: 14px;
                padding: 18px;
                text-align: center;
            ">
                <div style="
                    width: 60px; height: 60px; border-radius: 50%;
                    background: linear-gradient(135deg, {color}, {PLOFA_COLORS['bg_panel']});
                    margin: 0 auto 10px; display: flex; align-items: center; justify-content: center;
                    font-size: 1.5rem; font-weight: 900; color: #fff;
                ">{p[0]}</div>
                <div style="font-size: 1.1rem; font-weight: 800; color: {PLOFA_COLORS['accent_gold']};">
                    {p}
                </div>
                <div style="color: {PLOFA_COLORS['text_muted']}; font-size: 0.85rem; margin-bottom: 10px;">
                    {team} • {pos}
                </div>
                <div style="display: flex; justify-content: center; gap: 8px; flex-wrap: wrap; margin-bottom: 10px;">
                    <span class="stat-badge" style="background:{PLOFA_COLORS['accent_teal']}22; color:{PLOFA_COLORS['accent_teal']}; border: 1px solid {PLOFA_COLORS['accent_teal']}44;">
                        ⚽ {goals}
                    </span>
                    <span class="stat-badge" style="background:{PLOFA_COLORS['accent_gold']}22; color:{PLOFA_COLORS['accent_gold']}; border: 1px solid {PLOFA_COLORS['accent_gold']}44;">
                        🅰️ {assists}
                    </span>
                    <span class="stat-badge" style="background:{PLOFA_COLORS['accent_green']}22; color:{PLOFA_COLORS['accent_green']}; border: 1px solid {PLOFA_COLORS['accent_green']}44;">
                        ⭐ {rating:.1f}
                    </span>
                </div>
                <div style="color: {PLOFA_COLORS['text_muted']}; font-size: 0.8rem;">
                    {matches} matches • {mins} mins
                </div>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TAB 5: LEADERBOARDS
# ─────────────────────────────────────────────
def render_leaderboards(data, df):
    st.markdown('<p class="section-title">🏆 Season Leaderboards</p>', unsafe_allow_html=True)

    leaderboards = data.get("leaderboards", {})
    if not leaderboards:
        st.warning("No leaderboard data available.")
        return

    board_options = {
        "Top Scorers": ("top_scorers", "goals"),
        "Top Assisters": ("top_assisters", "assists"),
        "Best Pass Accuracy": ("best_pass_accuracy", "pass_accuracy"),
        "Most Tackles Won": ("most_tackles", "tackles_won"),
        "Most Interceptions": ("most_interceptions", "interceptions"),
        "Most Clean Sheets": ("most_clean_sheets", "clean_sheets"),
        "Highest Avg Rating": ("highest_avg_rating", "avg_rating"),
        "Most Saves": ("most_saves", "saves"),
        "Most Dribbles": ("most_dribbles", "dribbles_att"),
        "Most Pressures": ("most_pressures", "pressures"),
        "Most Recoveries": ("most_recoveries", "recoveries"),
        "Most Carries": ("most_carries", "carries"),
        "Best Dribble Success": ("best_dribble_success", "dribble_success_pct"),
        "Best Tackle Success": ("best_tackle_success", "tackle_success_pct"),
        "Best Press Success": ("best_press_success", "press_success_pct"),
        "Most xG": ("top_xG", "xg"),
        "Most xA": ("top_xA", "xa"),
        "Most Shots on Target": ("most_shots_on_target", "shots_on_target"),
    }

    selected_board = st.selectbox("Select Leaderboard", list(board_options.keys()))

    if selected_board in leaderboards:
        board_data = leaderboards[selected_board]
        if isinstance(board_data, list):
            board_df = pd.DataFrame(board_data)
            if not board_df.empty:
                # Add team color indicators
                if "team" in board_df.columns:
                    board_df = board_df[["player", "team", "position", "matches", "minutes", "value"]].head(15)
                else:
                    board_df = board_df[["player", "position", "matches", "minutes", "value"]].head(15)

                st.dataframe(
                    board_df.style.background_gradient(cmap="RdYlGn", subset=["value"])
                    .set_table_styles([
                        {"selector": "th", "props": [
                            ("background-color", PLOFA_COLORS["bg_panel"]),
                            ("color", PLOFA_COLORS["accent_gold"]),
                            ("font-weight", "800"),
                        ]},
                    ]),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            # Build from df
            key = board_options[selected_board][1]
            lb = df[["player", "team", "position", "matches", "minutes", key]].dropna(subset=[key])
            lb = lb.sort_values(key, ascending=False).head(15)
            st.dataframe(
                lb.style.background_gradient(cmap="RdYlGn", subset=[key])
                .set_table_styles([
                    {"selector": "th", "props": [
                        ("background-color", PLOFA_COLORS["bg_panel"]),
                        ("color", PLOFA_COLORS["accent_gold"]),
                        ("font-weight", "800"),
                    ]},
                ]),
                use_container_width=True,
                hide_index=True,
            )
    else:
        # Fallback: build from df
        key = board_options[selected_board][1]
        lb = df[["player", "team", "position", "matches", "minutes", key]].dropna(subset=[key])
        lb = lb.sort_values(key, ascending=False).head(15)
        if not lb.empty:
            st.dataframe(
                lb.style.background_gradient(cmap="RdYlGn", subset=[key])
                .set_table_styles([
                    {"selector": "th", "props": [
                        ("background-color", PLOFA_COLORS["bg_panel"]),
                        ("color", PLOFA_COLORS["accent_gold"]),
                        ("font-weight", "800"),
                    ]},
                ]),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.warning(f"No data available for: {selected_board}")

# ─────────────────────────────────────────────
# TAB 6: MATCH TIMELINE / STATS EXPLORER
# ─────────────────────────────────────────────
def render_stats_explorer(df):
    st.markdown('<p class="section-title">📈 Stats Explorer</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        x_stat = st.selectbox(
            "X Axis Stat",
            options=[c for c in df.columns if df[c].dtype in ["float64", "int64"] and c not in ["age"]],
            index=0,
            format_func=lambda x: x.replace("_", " ").title(),
        )
    with col2:
        y_stat = st.selectbox(
            "Y Axis Stat",
            options=[c for c in df.columns if df[c].dtype in ["float64", "int64"] and c not in ["age"]],
            index=1,
            format_func=lambda x: x.replace("_", " ").title(),
        )

    color_by = st.selectbox("Color By", ["team", "position", "archetype"], index=0)

    fig = px.scatter(
        df,
        x=x_stat,
        y=y_stat,
        color=color_by,
        hover_name="player",
        hover_data=["team", "position", "matches", "minutes"],
        template="plotly_dark",
        color_discrete_sequence=PLAYER_COLORS,
        height=600,
    )

    fig.update_layout(
        paper_bgcolor=PLOFA_COLORS["bg_dark"],
        plot_bgcolor=PLOFA_COLORS["bg_panel"],
        font=dict(color=PLOFA_COLORS["text_primary"]),
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)", title_font=dict(color=PLOFA_COLORS["accent_teal"])),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)", title_font=dict(color=PLOFA_COLORS["accent_teal"])),
        legend=dict(font=dict(color=PLOFA_COLORS["text_primary"])),
        margin=dict(l=40, r=40, t=40, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show top performers
    st.markdown(f"#### Top 10 by {x_stat.replace('_', ' ').title()}")
    top_x = df.nlargest(10, x_stat)[["player", "team", "position", x_stat, "matches"]]
    st.dataframe(
        top_x.style.background_gradient(cmap="RdYlGn", subset=[x_stat])
        .set_table_styles([
            {"selector": "th", "props": [
                ("background-color", PLOFA_COLORS["bg_panel"]),
                ("color", PLOFA_COLORS["accent_gold"]),
                ("font-weight", "800"),
            ]},
        ]),
        use_container_width=True,
        hide_index=True,
    )

# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
def main():
    # Header
    st.markdown('<p class="main-header">PLOFA ANALYTICS HUB</p>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="sub-header">Season 26/27 — Professional Player Statistics Dashboard</p>',
        unsafe_allow_html=True,
    )

    # Load data
    try:
        data = load_data()
        df = get_player_df(data)
    except Exception as e:
        st.error(f"❌ Failed to load data: {e}")
        st.info("Make sure `season_stats.json` exists in the parent directory.")
        st.stop()

    # Sidebar
    filtered_df, selected_players, stat_category = render_sidebar(df)

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Comparison Matrix",
        "🍕 Pizza Plots",
        "📋 Table Comparison",
        "🎯 Player Profiles",
        "🏆 Leaderboards",
    ])

    with tab1:
        render_comparison_matrix(filtered_df, selected_players, stat_category)

    with tab2:
        render_pizza_plots(filtered_df, selected_players, stat_category)

    with tab3:
        render_table_comparison(filtered_df, selected_players, stat_category)

    with tab4:
        render_player_profiles(filtered_df, selected_players)

    with tab5:
        render_leaderboards(data, filtered_df)

    # Footer
    st.markdown("---")
    st.markdown(
        f"<div style='text-align:center; color: {PLOFA_COLORS['text_muted']}; font-size: 0.8rem;'>"
        "PLOFA 26/27 Analytics • Built with Streamlit & Plotly"
        "</div>",
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
