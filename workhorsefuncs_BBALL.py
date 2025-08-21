import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
import schedule, time, datetime
import subprocess
import os

# ---- Helpers ----
def parse_Time(series: pd.Series) -> pd.Series:
    try:
        ts = pd.to_datetime(series.astype(str).str.strip(),
                            format="mixed", errors="coerce")
    except TypeError:
        ts = pd.to_datetime(series.astype(str).str.strip(),
                            errors="coerce", infer_datetime_format=True)
    return ts
def to_Num(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
def update_PPMPlot(datapath, plotpath):
    df = pd.read_csv(datapath)
    df.columns = df.columns.str.strip()
    ts_col = "Timestamp" if "Timestamp" in df.columns else ("Stoptime" if "Stoptime" in df.columns else None)
    if ts_col is None:
        raise ValueError(f"Could not find a timestamp column. Available columns: {list(df.columns)}")

    # Parse timestamps and numerics
    df[ts_col] = parse_Time(df[ts_col])
    to_Num(df, ["PPM", "Temp", "GoalNum", "MinsPlayed"])
    df = df.dropna(subset=[ts_col, "PPM"]).copy()
    df = df.sort_values(ts_col)

    # Add small padding to x-axis
    x_pad = (df[ts_col].max() - df[ts_col].min()) * 0.02
    x_range = [df[ts_col].min() - x_pad, df[ts_col].max() + x_pad]

    fig = go.Figure()

    # --- White lines ---
    for i in range(len(df) - 1):
        row, next_row = df.iloc[i], df.iloc[i + 1]
        fig.add_trace(go.Scatter(
            x=[row[ts_col], next_row[ts_col]],
            y=[row["PPM"], next_row["PPM"]],
            mode="lines",
            line=dict(color="white", width=2),
            hoverinfo="skip",
            showlegend=False,
            customdata=[(row.get('Court?'), row.get('Playlist'))]
        ))

    # --- Small colored circles ---
    fig.add_trace(go.Scatter(
        x=df[ts_col],
        y=df["PPM"],
        mode="markers",
        marker=dict(
            size=28,
            color=df["Temp"],
            colorscale="Inferno",
            cmin=df["Temp"].min(),
            cmax=df["Temp"].max(),
            line=dict(width=1, color="white"),
            symbol="circle",
            opacity=0.7
        ),
        hoverinfo="skip",
        showlegend=False,
        customdata=list(zip(df.get('Court?'), df.get('Playlist')))
    ))

    # --- Emoji markers ---
    fig.add_trace(go.Scatter(
        x=df[ts_col],
        y=df["PPM"],
        mode="text",
        text=["🏀"] * len(df),
        textfont=dict(size=18),
        hovertext=df.apply(lambda r:
                           f"# Points Made: {r.get('GoalNum', '')}<br>"
                           f"Court: {r.get('Court?', '')}<br>"
                           f"# Mins Played: {r.get('MinsPlayed', '')}<br>"
                           f"Temp: {r.get('Temp', '')}<br>"
                           f"Playlist: {r.get('Playlist', '')}<br>"
                           f"Notes: {r.get('Notes', '')}", axis=1),
        hoverinfo="text",
        showlegend=False,
        customdata=list(zip(df.get('Court?'), df.get('Playlist')))
    ))

    # --- Temperature colorbar ---
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(
            colorscale="Inferno",
            showscale=True,
            color=df["Temp"],
            size=0,
            colorbar=dict(
                title="Temperature (deg)",
                x=1.03,
                y=0.05,
                len=0.3,
                yanchor="bottom"
            )
        ),
        showlegend=False
    ))

    # --- Helper for dropdown filtering ---
    def make_visibility(selected_court, selected_playlist):
        vis = []
        for trace in fig.data[:-1]:
            if hasattr(trace, "customdata") and trace.customdata is not None:
                cd = trace.customdata[0]
                trace_court = cd[0] if cd is not None else None
                trace_playlist = cd[1] if cd is not None else None
                visible = True
                if selected_court != "All" and trace_court != selected_court:
                    visible = False
                if selected_playlist != "All" and trace_playlist != selected_playlist:
                    visible = False
                vis.append(visible)
            else:
                vis.append(False)
        return vis + [True]

    # --- Dropdown menus ---
    updatemenus = []

    if "Court?" in df.columns:
        court_vals = ["All"] + sorted(df["Court?"].dropna().unique())
        buttons = [dict(label=str(c),
                        method="update",
                        args=[{"visible": make_visibility(c, "All")},
                              {"title": f"Hannah's Hoops - Court: {c}"}]) for c in court_vals]
        updatemenus.append(dict(buttons=buttons, direction="down",
                                x=1.02, y=1.0, xanchor="left", yanchor="top", showactive=True))

    if "Playlist" in df.columns:
        playlist_vals = ["All"] + sorted(df["Playlist"].dropna().unique())
        buttons = [dict(label=str(p),
                        method="update",
                        args=[{"visible": make_visibility("All", p)},
                              {"title": f"Hannah's Hoops - Playlist: {p}"}]) for p in playlist_vals]
        updatemenus.append(dict(buttons=buttons, direction="down",
                                x=1.02, y=0.9, xanchor="left", yanchor="top", showactive=True))

    # --- Annotations ---
    annotations = [
        dict(text="Court?", x=1.07, y=1.03, xref="paper", yref="paper",
             showarrow=False, font=dict(color="white"), align="center"),
        dict(text="Playlist", x=1.07, y=0.93, xref="paper", yref="paper",
             showarrow=False, font=dict(color="white"), align="center")
    ]

    # --- Layout ---
    fig.update_layout(
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white", family="Courier New"),
        xaxis=dict(showgrid=True, gridcolor="gray", gridwidth=0.3,
                   tickformat="%m/%d", zeroline=False, linecolor="white",
                   linewidth=0.5, ticks="outside", range=x_range),
        yaxis=dict(title="PPM", showgrid=True, gridcolor="gray", gridwidth=0.3,
                   zeroline=False, linecolor="white", linewidth=2, ticks="outside"),
        title="365 Days of Hannah's Hoops",
        updatemenus=updatemenus,
        annotations=annotations
    )

    # Save HTML
    fig.write_html("ppm-time.html", auto_open=True)
    print(f"✅ Plot saved as {"ppm-time.html"}")
def pushto_Git(file_path, repo_path, commit_message="Update PPM plot"):
    file_path = os.path.abspath(file_path)
    repo_path = os.path.abspath(repo_path)
    cwd = os.getcwd()
    os.chdir(repo_path)

    try:
        subprocess.run(["git", "add", file_path], check=True)
        subprocess.run(["git", "commit", "-m", commit_message], check=True)
        subprocess.run(["git", "push"], check=True)
        print(f"✅ Successfully pushed {file_path} to GitHub.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Git command failed: {e}")
    finally:
        os.chdir(cwd)
