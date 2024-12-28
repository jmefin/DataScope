import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import tempfile
import base64
import os
import time
from datetime import datetime, timedelta

class DataImporter:
    def __init__(self):
        self.supported_formats = {
            'zulu': self._parse_zulu_format
        }
    
    def import_csv(self, file_path: str, format_type: str = 'zulu') -> pd.DataFrame:
        if format_type not in self.supported_formats:
            raise ValueError(f"Unsupported format type: {format_type}")
        return self.supported_formats[format_type](file_path)
    
    def _parse_zulu_format(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path, na_values=[''])
        df['dt'] = pd.to_datetime(df['dt'], utc=True)
        
        # Convert PM columns to numeric
        pm_columns = [col for col in df.columns if col.startswith('pm')]
        for col in pm_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        return df

# Default metric limits and target ranges
DEFAULT_LIMITS = {
    'temperature': {'target_low': 20.5, 'target_high': 26.0, 'alert_low': 20.0, 'alert_high': 27.0},
    'co2': {'target_low': 0, 'target_high': 750, 'alert_low': None, 'alert_high': 950},
    'humidity': {'target_low': 25, 'target_high': 60, 'alert_low': 15, 'alert_high': 75},
    'pm2p5': {'target_low': 0, 'target_high': 10, 'alert_low': None, 'alert_high': 25},
    'tvoc': {'target_low': 0, 'target_high': 250, 'alert_low': None, 'alert_high': 750},
    'laeqx': {'target_low': 0, 'target_high': 53, 'alert_low': None, 'alert_high': 58}
}

# Prioritized metrics order
PRIORITY_METRICS = ['temperature', 'co2', 'humidity', 'pm2p5', 'tvoc', 'laeqx', 'pm1p0', 'pm4p0', 'pm10p0']

# Set up Streamlit app
st.set_page_config(layout="wide")
st.title("DataScope")
st.sidebar.header("Configuration")

# Multiple file upload
uploaded_files = st.sidebar.file_uploader("Upload CSV files", type=['csv'], accept_multiple_files=True)

# Dictionary to store imported dataframes
if 'imported_data' not in st.session_state:
    st.session_state.imported_data = {}

# Import data from uploaded files
if uploaded_files:
    importer = DataImporter()
    message_placeholder = st.empty()
    
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        if file_name not in st.session_state.imported_data:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            data = importer.import_csv(tmp_file_path)
            st.session_state.imported_data[file_name] = data
            os.unlink(tmp_file_path)
    
    message_placeholder.success(f"Loaded {len(uploaded_files)} files successfully!")
    time.sleep(1)
    message_placeholder.empty()

# Check if we have imported data
if st.session_state.imported_data:
    # Get available metrics from all files
    all_metrics = set()
    for df in st.session_state.imported_data.values():
        all_metrics.update([col for col in df.columns if col != 'dt'])
    
    # Create ordered list of metrics with priority metrics first
    available_metrics = (
        [metric for metric in PRIORITY_METRICS if metric in all_metrics] +
        sorted([metric for metric in all_metrics if metric not in PRIORITY_METRICS])
    )
    
    # Sidebar: Select metrics to plot
    st.sidebar.subheader("Metrics Selection")
    primary_metrics = st.sidebar.multiselect(
        "Primary metrics (main y-axis):",
        available_metrics,
        default=['temperature'] if 'temperature' in available_metrics else [available_metrics[0]]
    )

    # Option to enable secondary y-axis
    use_secondary_axis = st.sidebar.checkbox("Use secondary y-axis", value=False)

    secondary_metric = None
    if use_secondary_axis:
        # Filter out the primary metrics from available options
        secondary_metrics = [m for m in available_metrics if m not in primary_metrics]
        if secondary_metrics:  # Check if there are any metrics left to choose from
            secondary_metric = st.sidebar.selectbox(
                "Secondary metric (right y-axis):",
                secondary_metrics
            )
        else:
            st.sidebar.warning("No available metrics for secondary axis")
            use_secondary_axis = False

    # Limit values setup
    limit_values = {}
    metrics_to_process = primary_metrics.copy()
    if use_secondary_axis and secondary_metric:
        metrics_to_process.append(secondary_metric)

    for metric in metrics_to_process:
        if metric in DEFAULT_LIMITS:
            defaults = DEFAULT_LIMITS[metric]
            st.sidebar.subheader(f"Limits for {metric}")
            
            show_limits = st.sidebar.checkbox(
                f"Show limit lines for {metric}",
                value=False,
                key=f"show_limits_{metric}"
            )
            
            show_target_area = st.sidebar.checkbox(
                f"Show target area for {metric}",
                value=False,
                key=f"target_area_{metric}"
            )
            
            col1, col2 = st.sidebar.columns(2)
            with col1:
                target_low = st.number_input(
                    f"Target low for {metric}",
                    value=float(defaults['target_low']),
                    step=0.1
                )
            with col2:
                target_high = st.number_input(
                    f"Target high for {metric}",
                    value=float(defaults['target_high']),
                    step=0.1
                )
            
            col3, col4 = st.sidebar.columns(2)
            with col3:
                alert_low = st.number_input(
                    f"Alert low for {metric}",
                    value=float(defaults['alert_low']) if defaults['alert_low'] is not None else target_low,
                    step=0.1
                )
            with col4:
                alert_high = st.number_input(
                    f"Alert high for {metric}",
                    value=float(defaults['alert_high']) if defaults['alert_high'] is not None else target_high,
                    step=0.1
                )
            
            limit_values[metric] = {
                'target_low': target_low,
                'target_high': target_high,
                'alert_low': alert_low,
                'alert_high': alert_high,
                'show_target_area': show_target_area,
                'show_limits': show_limits
            }
        else:
            st.sidebar.subheader(f"Limits for {metric}")
            
            show_limits = st.sidebar.checkbox(
                f"Show limit lines for {metric}",
                value=False,
                key=f"show_limits_{metric}"
            )
            
            show_target_area = st.sidebar.checkbox(
                f"Show target area for {metric}",
                value=False,
                key=f"target_area_{metric}"
            )
            
            lower_limit = st.sidebar.number_input(f"Lower limit for {metric}", value=0.0, step=0.1)
            upper_limit = st.sidebar.number_input(f"Upper limit for {metric}", value=100.0, step=0.1)
            
            limit_values[metric] = {
                'target_low': lower_limit,
                'target_high': upper_limit,
                'alert_low': lower_limit,
                'alert_high': upper_limit,
                'show_target_area': show_target_area,
                'show_limits': show_limits
            }

    # Plot settings
    st.sidebar.subheader("Plot Settings")
    plot_height = st.sidebar.slider(
        'Plot height',
        min_value=300,
        max_value=3000,
        value=400,
        step=50
    )
    
    # Select data sources
    st.sidebar.subheader("Select Data Sources")
    selected_files = st.sidebar.multiselect(
        "Select files to display:",
        list(st.session_state.imported_data.keys()),
        default=list(st.session_state.imported_data.keys())
    )

    # Create plot
    if primary_metrics:
        fig = go.Figure()

        # Plot data
        for file_name, data in st.session_state.imported_data.items():
            if file_name in selected_files:
                short_name = file_name.split('_')[2].replace('.csv', '')
                if short_name.isdigit():
                    short_name = short_name[-4:]
                
                # Primary metrics
                for metric in primary_metrics:
                    if metric in data.columns:
                        plot_data = data.dropna(subset=[metric])
                        fig.add_trace(go.Scatter(
                            x=plot_data['dt'],
                            y=plot_data[metric],
                            mode='lines',
                            name=f"{short_name} - {metric}",
                            hovertemplate=f"<b>{short_name} - {metric}</b><br>" +
                                        "Value: %{y:.2f}<br>" +
                                        "Time: %{x}<extra></extra>"
                        ))
                
                # Secondary metric if enabled
                if use_secondary_axis and secondary_metric and secondary_metric in data.columns:
                    plot_data = data.dropna(subset=[secondary_metric])
                    fig.add_trace(go.Scatter(
                        x=plot_data['dt'],
                        y=plot_data[secondary_metric],
                        mode='lines',
                        name=f"{short_name} - {secondary_metric}",
                        yaxis="y2",
                        hovertemplate=f"<b>{short_name} - {secondary_metric}</b><br>" +
                                    "Value: %{y:.2f}<br>" +
                                    "Time: %{x}<extra></extra>"
                    ))

        # Add limit lines and target areas if they exist
        if limit_values:
            for metric in metrics_to_process:
                if metric in limit_values:
                    limits = limit_values[metric]
                    yaxis = "y2" if metric == secondary_metric else "y"
                    
                    if limits['show_target_area']:
                        fig.add_trace(go.Scatter(
                            x=[fig.data[0].x[0], fig.data[0].x[-1]],
                            y=[limits['target_high'], limits['target_high']],
                            fill=None,
                            mode='lines',
                            line=dict(color='rgba(0,255,0,0.2)', width=0),
                            showlegend=False,
                            name=f"{metric} target high",
                            yaxis=yaxis
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=[fig.data[0].x[0], fig.data[0].x[-1]],
                            y=[limits['target_low'], limits['target_low']],
                            fill='tonexty',
                            mode='lines',
                            line=dict(color='rgba(0,255,0,0.2)', width=0),
                            fillcolor='rgba(0,255,0,0.2)',
                            showlegend=False,
                            name=f"{metric} target low",
                            yaxis=yaxis
                        ))
                    
                    if limits['show_limits']:
                        fig.add_hline(
                            y=limits['target_low'],
                            line_dash="dash",
                            line_color="green",
                            opacity=0.5,
                            name=f"{metric} target low",
                            layer="below"
                        )
                        fig.add_hline(
                            y=limits['target_high'],
                            line_dash="dash",
                            line_color="green",
                            opacity=0.5,
                            name=f"{metric} target high",
                            layer="below"
                        )
                        
                        if limits['alert_low'] is not None:
                            fig.add_hline(
                                y=limits['alert_low'],
                                line_dash="dash",
                                line_color="red",
                                opacity=0.5,
                                name=f"{metric} alert low",
                                layer="below"
                            )
                        
                        if limits['alert_high'] is not None:
                            fig.add_hline(
                                y=limits['alert_high'],
                                line_dash="dash",
                                line_color="red",
                                opacity=0.5,
                                name=f"{metric} alert high",
                                layer="below"
                            )

        # Update layout based on axis configuration
        layout_args = {
            'height': plot_height,
            'hovermode': 'x unified',
            'yaxis': dict(
                title=', '.join(primary_metrics),
                side="left"
            ),
        }

        if use_secondary_axis and secondary_metric:
            layout_args.update({
                'yaxis2': dict(
                    title=secondary_metric,
                    side="right",
                    overlaying="y",
                    showgrid=False
                )
            })

        fig.update_layout(**layout_args)
        
        # Display the plot
        st.plotly_chart(fig, use_container_width=True)

        # Statistical summary
        if st.checkbox("Show Statistical Summary"):
            st.subheader("Statistical Summary")
            for file_name in selected_files:
                st.write(f"Statistics for {file_name}:")
                metrics_to_show = metrics_to_process
                st.dataframe(st.session_state.imported_data[file_name][metrics_to_show].describe())
