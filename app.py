"""
Interactive Ridgeline Plot for Overheating Metrics - Render.com Deployment
Displays distributions of all_main_zones overheating hours with interactive filters
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
import os

# Get script directory and construct paths relative to it
script_dir = Path(__file__).parent


dtype_map = {
    "all_main_zones_above_26_percent": "float32",
    "all_main_zones_above_26_percent_july": "float32",
    "all_main_zones_above_26_percent_peak_week": "float32",
    "all_main_zones_above_26_percent_peak_day": "float32",
    "fdwr": "float32",
    "house_peak_summer_temp": "float32",
}


# Load data files
print("Loading data files...")
df = pd.read_csv(script_dir / 'parametric_results.csv.gz', compression="gzip", dtype=dtype_map)

with open(script_dir / 'climate_zone_cities.json', 'r') as f:
    climate_zone_cities = json.load(f)

climate_data = pd.read_csv(script_dir / 'cwec_climate_data.csv')

print(f"Loaded {len(df)} simulation results")

# Create mapping from city to weather_file using climate_zone_cities.json
city_to_weather_file = {}
for cz, cities in climate_zone_cities.items():
    for city_info in cities:
        city_name = city_info['city']
        # Handle the space before weather_file key (typo in JSON)
        weather_file = city_info.get(' weather_file', city_info.get('weather_file', ''))
        city_to_weather_file[city_name] = weather_file.strip()

# Create mapping from weather_file to climate data
weather_to_climate = {}
for _, row in climate_data.iterrows():
    station_name = row['Station_Name'].strip()
    weather_to_climate[station_name] = {
        'CDD10': row.get('CDD10', np.nan),
        'Tdb2.5': row.get('Tdb', np.nan)
    }
    # Also store without suffix for partial matching
    base_name = station_name
    for suffix in [' AP', ' CS', ' RCS', ' AgDM', ' AWOS', ' AUT', ' A']:
        if base_name.endswith(suffix):
            base_name = base_name[:-len(suffix)]
            break
    if base_name != station_name:
        weather_to_climate[base_name] = weather_to_climate[station_name]

# Map city -> climate data via weather_file
def get_climate_data(city_name):
    weather_file = city_to_weather_file.get(city_name, '')
    if not weather_file:
        return {'CDD10': np.nan, 'Tdb2.5': np.nan}
    
    if weather_file in weather_to_climate:
        return weather_to_climate[weather_file]
    
    for station_name, climate_vals in weather_to_climate.items():
        if station_name.startswith(weather_file):
            return climate_vals
    
    return {'CDD10': np.nan, 'Tdb2.5': np.nan}

# Add climate data to simulation dataframe
df['CDD10'] = df['city'].apply(lambda x: get_climate_data(x)['CDD10'])
df['Tdb2.5'] = df['city'].apply(lambda x: get_climate_data(x)['Tdb2.5'])

# Extract all_main_zones metrics and convert percentages to hours
df['all_zones_hours_summer'] = df['all_main_zones_above_26_percent'] * 36.72
df['all_zones_hours_july'] = df['all_main_zones_above_26_percent_july'] * 7.44
df['all_zones_hours_peak_week'] = df['all_main_zones_above_26_percent_peak_week'] * 1.68
df['all_zones_hours_peak_day'] = df['all_main_zones_above_26_percent_peak_day'] * 0.24

# Round FDWR to create discrete categories
df['fdwr_rounded'] = df['fdwr'].round(3)

print("Data preparation complete")
print(f"CDD10 range: {df['CDD10'].min():.1f} - {df['CDD10'].max():.1f}")
print(f"Tdb2.5 range: {df['Tdb2.5'].min():.1f} - {df['Tdb2.5'].max():.1f}")
print(f"FDWR values: {sorted(df['fdwr_rounded'].unique())}")

# Create the interactive ridgeline plot
def create_ridgeline_plot(df_filtered):
    """Create ridgeline plot with four distributions using complementary CDF (survival function)"""
    
    metrics = [
        {'col': 'all_zones_hours_summer', 'label': 'Summer (May-Sep)', 'color': 'rgba(255, 100, 100, 0.6)'},
        {'col': 'all_zones_hours_july', 'label': 'July Only', 'color': 'rgba(255, 165, 0, 0.6)'},
        {'col': 'all_zones_hours_peak_week', 'label': 'Peak Week', 'color': 'rgba(255, 200, 50, 0.6)'},
        {'col': 'all_zones_hours_peak_day', 'label': 'Peak Day', 'color': 'rgba(255, 235, 100, 0.6)'}
    ]
    
    # Create subplots - one row for each metric
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.08,
        subplot_titles=[m['label'] for m in metrics],
        row_heights=[0.25, 0.25, 0.25, 0.25]
    )
    
    for i, metric in enumerate(metrics, start=1):
        data = df_filtered[metric['col']].dropna()
        
        if len(data) > 0:
            sorted_data = np.sort(data)
            n = len(sorted_data)
            ccdf = np.arange(n, 0, -1) / n
            
            # Add baseline at 0%
            fig.add_trace(go.Scatter(
                x=[sorted_data[0], sorted_data[-1]],
                y=[0, 0],
                mode='lines',
                line=dict(color='gray', width=1),
                showlegend=False,
                hoverinfo='skip',
                fill=None
            ), row=i, col=1)
            
            # Add complementary CDF curve
            fig.add_trace(go.Scatter(
                x=sorted_data,
                y=ccdf,
                fill='tonexty',
                fillcolor=metric['color'],
                line=dict(color=metric['color'].replace('0.6', '1.0'), width=2),
                name=metric['label'],
                showlegend=(i == 1),
                hovertemplate='<b>%{fullData.name}</b><br>Hours: %{x:.1f}<br>Proportion ≥ x: %{y:.1%}<extra></extra>'
            ), row=i, col=1)
            
            # Add statistics as annotation
            median_val = np.median(data)
            mean_val = np.mean(data)
            p99 = np.percentile(data, 99)
            p99_9 = np.percentile(data, 99.9)
            p99_998 = np.percentile(data, 99.998)
            
            
            fig.add_annotation(
                xref=f'x{i}', yref=f'y{i}',
                x=sorted_data[-1],
                y=0.5,
                text=f"n={len(data)}, μ={mean_val:.1f}h, m={median_val:.1f}h, 99th={p99:.1f}h, 99.9th={p99_9:.1f}h, 99.998th={p99_998:.1f}h",
                showarrow=False,
                xanchor='right',
                font=dict(size=9, color='gray')
            )
            
            # Update axes
            fig.update_xaxes(
                title_text='Hours Above 26°C' if i == 4 else '',
                showgrid=True,
                zeroline=False,
                row=i, col=1
            )
            
            fig.update_yaxes(
                title_text='Proportion',
                showticklabels=True,
                showgrid=True,
                zeroline=False,
                tickformat='.0%',
                range=[-0.05, 1.05],
                row=i, col=1
            )
    
    fig.update_layout(
        title=dict(
            text=f'Overheating Hours Distribution - All Main Zones Above 26°C<br><sub>Complementary CDF: Proportion of simulations with X or more hours | n={len(df_filtered)} simulations</sub>',
            x=0.5,
            xanchor='center'
        ),
        height=900,
        width=1200,
        template='plotly_white',
        showlegend=False
    )
    
    return fig

# Create initial plot with all data
initial_fig = create_ridgeline_plot(df)

# Create interactive sliders and multi-select dropdown
cdd10_min, cdd10_max = df['CDD10'].min(), df['CDD10'].max()
tdb_min, tdb_max = df['Tdb2.5'].min(), df['Tdb2.5'].max()
fdwr_values = sorted(df['fdwr_rounded'].unique())

# Create Dash app for interactive filtering
print("\nCreating interactive dashboard...")

from dash import Dash, dcc, html, Input, Output

app = Dash(__name__)
server = app.server  # Expose server for Gunicorn

app.layout = html.Div([
    html.H1("Overheating Analysis - Interactive Ridgeline Plot", 
            style={'textAlign': 'center', 'color': '#2c3e50'}),
    
    html.Div([
        html.Div([
            html.Div([
                html.Label('CDD10 Range (Cooling Degree Days):', 
                          style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                dcc.RangeSlider(
                    id='cdd10-slider',
                    min=float(cdd10_min),
                    max=float(cdd10_max),
                    step=10,
                    value=[float(cdd10_min), float(cdd10_max)],
                    marks={int(v): f'{int(v)}' for v in np.linspace(cdd10_min, cdd10_max, 8)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'width': '100%', 'padding': '20px'}),
            
            html.Div([
                html.Label('Tdb 2.5% Range (Design Temperature °C):', 
                          style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                dcc.RangeSlider(
                    id='tdb-slider',
                    min=float(tdb_min),
                    max=float(tdb_max),
                    step=0.5,
                    value=[float(tdb_min), float(tdb_max)],
                    marks={int(v): f'{int(v)}°C' for v in np.linspace(tdb_min, tdb_max, 8)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'width': '100%', 'padding': '20px'}),
            
            html.Div([
                html.Label('FDWR Values (Facade to Wall Ratio):', 
                          style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                dcc.Dropdown(
                    id='fdwr-dropdown',
                    options=[{'label': f'{v:.3f}', 'value': v} for v in fdwr_values],
                    value=fdwr_values,
                    multi=True,
                    placeholder="Select FDWR values..."
                )
            ], style={'width': '100%', 'padding': '20px'}),
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        
        html.Div([
            dcc.Graph(id='cdd-tdb-plot', style={'height': '500px'})
        ], style={'width': '65%', 'display': 'inline-block', 'verticalAlign': 'top'})
    ], style={'display': 'flex', 'width': '100%'}),
    
    html.Div(id='filter-summary', 
            style={'textAlign': 'center', 'padding': '10px', 'fontStyle': 'italic', 'color': '#7f8c8d'}),
    
    dcc.Graph(id='ridgeline-plot', style={'height': '950px'}),
    
    dcc.Graph(id='temp-distribution-plot', style={'height': '500px'})
])

@app.callback(
    [Output('ridgeline-plot', 'figure'),
     Output('filter-summary', 'children'),
     Output('cdd-tdb-plot', 'figure'),
     Output('temp-distribution-plot', 'figure')],
    [Input('cdd10-slider', 'value'),
     Input('tdb-slider', 'value'),
     Input('fdwr-dropdown', 'value')]
)
def update_plot(cdd10_range, tdb_range, fdwr_selected):
    df_filtered = df.copy()

    # Apply CDD10 OR Tdb2.5 filter
    df_filtered = df_filtered[
        ((df_filtered['CDD10'] >= cdd10_range[0]) & 
         (df_filtered['CDD10'] <= cdd10_range[1])) |
        ((df_filtered['Tdb2.5'] >= tdb_range[0]) & 
         (df_filtered['Tdb2.5'] <= tdb_range[1]))
    ]
    
    # Apply FDWR filter (AND with above)
    if fdwr_selected:
        df_filtered = df_filtered[df_filtered['fdwr_rounded'].isin(fdwr_selected)]
    
    summary = f"Showing {len(df_filtered):,} of {len(df):,} simulations | "
    summary += f"CDD10: {cdd10_range[0]:.0f}-{cdd10_range[1]:.0f} | "
    summary += f"Tdb2.5: {tdb_range[0]:.1f}°C-{tdb_range[1]:.1f}°C | "
    summary += f"FDWR: {len(fdwr_selected) if fdwr_selected else 0}/{len(fdwr_values)} values selected"
    
    if len(df_filtered) > 0:
        fig = create_ridgeline_plot(df_filtered)
    else:
        fig = go.Figure()
        fig.add_annotation(
            text="No data matches the current filter criteria",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20, color="red")
        )
        fig.update_layout(height=900, width=1200)
    
    # Create CDD vs Tdb scatter plot with selection highlighting
    # Plot unique climate points only (not all simulations)
    fig_scatter = go.Figure()
    
    # Determine which climate points are selected
    climate_in_selection = (
        ((climate_data['CDD10'] >= cdd10_range[0]) & (climate_data['CDD10'] <= cdd10_range[1])) |
        ((climate_data['Tdb'] >= tdb_range[0]) & (climate_data['Tdb'] <= tdb_range[1]))
    )
    
    # Add unselected climate points
    climate_unselected = climate_data[~climate_in_selection]
    if len(climate_unselected) > 0:
        fig_scatter.add_trace(go.Scatter(
            x=climate_unselected['CDD10'],
            y=climate_unselected['Tdb'],
            mode='markers',
            marker=dict(size=8, color='rgba(200, 200, 200, 0.3)', line=dict(width=1, color='gray')),
            text=climate_unselected['Station_Name'],
            textposition="top center",
            textfont=dict(size=8, color='rgba(200, 200, 200, 0.5)'),
            name='Unselected',
            hovertemplate='<b>%{text}</b><br>CDD10: %{x:.0f}<br>Tdb2.5: %{y:.1f}°C<extra></extra>'
        ))
    
    # Add selected climate points
    climate_selected = climate_data[climate_in_selection]
    if len(climate_selected) > 0:
        fig_scatter.add_trace(go.Scatter(
            x=climate_selected['CDD10'],
            y=climate_selected['Tdb'],
            mode='markers',
            marker=dict(size=10, color='rgba(0, 100, 255, 0.7)', line=dict(width=2, color='darkblue')),
            text=climate_selected['Station_Name'],
            textposition="top center",
            textfont=dict(size=9, color='darkblue'),
            name='Selected',
            hovertemplate='<b>%{text}</b><br>CDD10: %{x:.0f}<br>Tdb2.5: %{y:.1f}°C<extra></extra>'
        ))
    
    # Add vertical lines for CDD10 range
    fig_scatter.add_vline(x=cdd10_range[0], line_dash="dash", line_color="red", line_width=2,
                          annotation_text="CDD min", annotation_position="top left")
    fig_scatter.add_vline(x=cdd10_range[1], line_dash="dash", line_color="red", line_width=2,
                          annotation_text="CDD max", annotation_position="top right")
    
    # Add horizontal lines for Tdb range
    fig_scatter.add_hline(y=tdb_range[0], line_dash="dash", line_color="green", line_width=2,
                          annotation_text="Tdb min", annotation_position="left")
    fig_scatter.add_hline(y=tdb_range[1], line_dash="dash", line_color="green", line_width=2,
                          annotation_text="Tdb max", annotation_position="right")
    
    fig_scatter.update_layout(
        title='Climate Zones - CDD10 vs Tdb2.5',
        xaxis_title='CDD10 (Cooling Degree Days)',
        yaxis_title='Tdb2.5 (Design Temperature °C)',
        height=500,
        width=900,
        template='plotly_white',
        hovermode='closest'
    )
    
    # Create peak summer temperature distribution
    fig_temp = go.Figure()
    data = df_filtered['house_peak_summer_temp'].dropna()
    
    if len(data) > 0:
        sorted_data = np.sort(data)
        n = len(sorted_data)
        ccdf = np.arange(n, 0, -1) / n
        
        # Add baseline at 0%
        fig_temp.add_trace(go.Scatter(
            x=[sorted_data[0], sorted_data[-1]],
            y=[0, 0],
            mode='lines',
            line=dict(color='gray', width=1),
            showlegend=False,
            hoverinfo='skip',
            fill=None
        ))
        
        # Add complementary CDF curve
        fig_temp.add_trace(go.Scatter(
            x=sorted_data,
            y=ccdf,
            fill='tonexty',
            fillcolor='rgba(100, 200, 255, 0.6)',
            line=dict(color='rgba(0, 100, 200, 1.0)', width=2),
            name='House Peak Summer Temperature',
            hovertemplate='<b>Temperature</b><br>°C: %{x:.1f}<br>Proportion ≥ x: %{y:.1%}<extra></extra>'
        ))
        
        # Add statistics
        median_val = np.median(data)
        mean_val = np.mean(data)
        p95 = np.percentile(data, 95)
        
        fig_temp.add_annotation(
            xref='x', yref='y',
            x=sorted_data[-1],
            y=0.5,
            text=f"n={len(data)}, μ={mean_val:.1f}°C, m={median_val:.1f}°C, 95th={p95:.1f}°C",
            showarrow=False,
            xanchor='right',
            font=dict(size=10, color='gray')
        )
    
    fig_temp.update_layout(
        title='House Peak Summer Temperature Distribution',
        xaxis_title='Temperature (°C)',
        yaxis_title='Proportion',
        height=500,
        width=1200,
        template='plotly_white',
        showlegend=False,
        yaxis=dict(tickformat='.0%', range=[-0.05, 1.05])
    )
    
    return fig, summary, fig_scatter, fig_temp

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    app.run(host='0.0.0.0', port=port, debug=False)
