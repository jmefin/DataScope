import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import tempfile
import base64
import os
import time
import numpy as np
from datetime import datetime, timedelta
from database_manager import DatabaseManager
from analysis import AdvancedAnalysis

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

def setup_limit_values(metric, defaults, key_suffix=""):
   """Helper function to set up limit values for a metric"""
   st.sidebar.subheader(f"Limits for {metric}")
   
   show_limits = st.sidebar.checkbox(
       f"Show limit lines for {metric}", 
       value=False,
       key=f"show_limits_{metric}_{key_suffix}"
   )
   
   show_target_area = st.sidebar.checkbox(
       f"Show target area for {metric}", 
       value=False,
       key=f"show_target_area_{metric}_{key_suffix}"
   )
   
   col1, col2 = st.sidebar.columns(2)
   with col1:
       target_low = st.number_input(
           f"Target low for {metric}", 
           value=float(defaults['target_low']), 
           step=0.1,
           key=f"target_low_{metric}_{key_suffix}"
       )
   with col2:
       target_high = st.number_input(
           f"Target high for {metric}", 
           value=float(defaults['target_high']), 
           step=0.1,
           key=f"target_high_{metric}_{key_suffix}"
       )
   
   col3, col4 = st.sidebar.columns(2)
   with col3:
       alert_low = st.number_input(
           f"Alert low for {metric}", 
           value=float(defaults['alert_low']) if defaults['alert_low'] is not None else target_low, 
           step=0.1,
           key=f"alert_low_{metric}_{key_suffix}"
       )
   with col4:
       alert_high = st.number_input(
           f"Alert high for {metric}", 
           value=float(defaults['alert_high']) if defaults['alert_high'] is not None else target_high, 
           step=0.1,
           key=f"alert_high_{metric}_{key_suffix}"
       )
   
   return {
       'target_low': target_low,
       'target_high': target_high,
       'alert_low': alert_low,
       'alert_high': alert_high,
       'show_target_area': show_target_area,
       'show_limits': show_limits
   }

# Set up Streamlit app
st.set_page_config(layout="wide")
st.title("DataScope")
st.sidebar.header("Configuration")

# Initialize database and analysis managers in session state
if 'db_manager' not in st.session_state:
   st.session_state.db_manager = DatabaseManager()
   st.session_state.analysis = AdvancedAnalysis(st.session_state.db_manager)
   st.session_state.uploaded_files = {}

# Multiple file upload
uploaded_files = st.sidebar.file_uploader("Upload CSV files", type=['csv'], accept_multiple_files=True)

# Import data from uploaded files
if uploaded_files:
   message_placeholder = st.empty()
   
   for uploaded_file in uploaded_files:
       file_name = uploaded_file.name
       if file_name not in st.session_state.uploaded_files:
           with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
               tmp_file.write(uploaded_file.getvalue())
               
               df = pd.read_csv(tmp_file.name, na_values=[''])
               df['dt'] = pd.to_datetime(df['dt'], utc=True)
               
               dataset_id = st.session_state.db_manager.store_dataset(file_name, df)
               st.session_state.uploaded_files[file_name] = dataset_id
               
               os.unlink(tmp_file.name)
   
   message_placeholder.success(f"Loaded {len(uploaded_files)} files successfully!")
   time.sleep(1)
   message_placeholder.empty()
# Check if we have uploaded files
if st.session_state.uploaded_files:
    # Get available metrics
    all_metrics = st.session_state.db_manager.get_available_metrics()
    
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
    use_secondary_axis = st.sidebar.checkbox(
        "Use secondary y-axis", 
        value=False,
        key="use_secondary_axis_main"
    )

    secondary_metric = None
    if use_secondary_axis:
        secondary_metrics = [m for m in available_metrics if m not in primary_metrics]
        if secondary_metrics:
            secondary_metric = st.sidebar.selectbox(
                "Secondary metric (right y-axis):",
                secondary_metrics
            )
        else:
            st.sidebar.warning("No available metrics for secondary axis")
            use_secondary_axis = False

    # Option to show difference plot
    st.sidebar.subheader("Difference Plot")
    show_differences = st.sidebar.checkbox(
        "Show difference plot", 
        value=False,
        key="show_differences_main"
    )

    if show_differences and len(st.session_state.uploaded_files) >= 2:
        difference_metrics = st.sidebar.multiselect(
            "Select metrics for difference calculation:",
            primary_metrics,
            default=[primary_metrics[0]] if primary_metrics else []
        )
        
        reference_file = st.sidebar.selectbox(
            "Select reference file:",
            list(st.session_state.uploaded_files.keys()),
            index=0,
            key="reference_file_main"
        )
    elif show_differences and len(st.session_state.uploaded_files) < 2:
        st.sidebar.warning("Need at least 2 files selected to calculate differences")
        show_differences = False

    # Limit values setup
    limit_values = {}
    metrics_to_process = primary_metrics.copy()
    if use_secondary_axis and secondary_metric:
        metrics_to_process.append(secondary_metric)

    for metric in metrics_to_process:
        if metric in DEFAULT_LIMITS:
            limit_values[metric] = setup_limit_values(metric, DEFAULT_LIMITS[metric], "main")

    # Statistical analysis - Moved here before difference plot
    st.sidebar.subheader("Statistical Analysis")
    stats_level = st.sidebar.radio(
        "Select statistics level:",
        ["Basic", "Advanced"],
        index=0,
        key="stats_level_main"
    )
    show_stats = st.sidebar.checkbox("Show Statistical Summary", key="show_stats_main")

    # Plot settings
    st.sidebar.subheader("Plot Settings")
    plot_height = st.sidebar.slider(
        'Plot height', 
        min_value=300, 
        max_value=2000, 
        value=400, 
        step=50,
        key="plot_height_main"
    )
    
    # Select data sources
    st.sidebar.subheader("Select Data Sources")
    selected_files = st.sidebar.multiselect(
        "Select files to display:",
        list(st.session_state.uploaded_files.keys()),
        default=list(st.session_state.uploaded_files.keys()),
        key="selected_files_main"
    )

    # Get selected dataset IDs
    selected_dataset_ids = [st.session_state.uploaded_files[file] for file in selected_files]

    # Create plot if we have data to show
    if primary_metrics and selected_dataset_ids:
        # Fetch data from database
        plot_data = st.session_state.db_manager.get_dataset_data(
            selected_dataset_ids,
            metrics_to_process
        )

        # Main plot
        fig = go.Figure()

        # Plot data for each file
        for file_name in selected_files:
            file_data = plot_data[plot_data['filename'] == file_name]
            short_name = file_name.split('_')[2].replace('.csv', '')
            if short_name.isdigit():
                short_name = short_name[-4:]
            
            # Primary metrics
            for metric in primary_metrics:
                if metric in file_data.columns:
                    fig.add_trace(go.Scatter(
                        x=file_data['timestamp'],
                        y=file_data[metric],
                        mode='lines',
                        name=f"{short_name} - {metric}",
                        hovertemplate=f"<b>{short_name} - {metric}</b><br>" +
                                    "Value: %{y:.2f}<br>" +
                                    "Time: %{x}<extra></extra>"
                    ))
            
            # Secondary metric if enabled
            if use_secondary_axis and secondary_metric and secondary_metric in file_data.columns:
                fig.add_trace(go.Scatter(
                    x=file_data['timestamp'],
                    y=file_data[secondary_metric],
                    mode='lines',
                    name=f"{short_name} - {secondary_metric}",
                    yaxis="y2",
                    hovertemplate=f"<b>{short_name} - {secondary_metric}</b><br>" +
                                "Value: %{y:.2f}<br>" +
                                "Time: %{x}<extra></extra>"
                ))

        # Add limit lines and target areas
        if limit_values and len(fig.data) > 0:
            for metric in metrics_to_process:
                if metric in limit_values:
                    limits = limit_values[metric]
                    yaxis = "y2" if metric == secondary_metric else "y"
                    
                    if limits['show_target_area']:
                        fig.add_trace(go.Scatter(
                            x=[plot_data['timestamp'].min(), plot_data['timestamp'].max()],
                            y=[limits['target_high'], limits['target_high']],
                            fill=None,
                            mode='lines',
                            line=dict(color='rgba(0,255,0,0.2)', width=0),
                            showlegend=False,
                            name=f"{metric} target high",
                            yaxis=yaxis
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=[plot_data['timestamp'].min(), plot_data['timestamp'].max()],
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
                        fig.add_hline(y=limits['target_low'], line_dash="dash",
                                    line_color="green", opacity=0.5,
                                    name=f"{metric} target low")
                        fig.add_hline(y=limits['target_high'], line_dash="dash",
                                    line_color="green", opacity=0.5,
                                    name=f"{metric} target high")
                        fig.add_hline(y=limits['alert_low'], line_dash="dash",
                                    line_color="red", opacity=0.5,
                                    name=f"{metric} alert low")
                        fig.add_hline(y=limits['alert_high'], line_dash="dash",
                                    line_color="red", opacity=0.5,
                                    name=f"{metric} alert high")

        # Update layout
        layout_args = {
            'height': plot_height,
            'hovermode': 'x unified',
            'yaxis': dict(title=', '.join(primary_metrics), side="left")
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
        
        # Display the main plot
        st.plotly_chart(fig, use_container_width=True)

        # Show statistical analysis if enabled
        if show_stats:
            st.subheader("Statistical Summary")
            
            if stats_level == "Basic":
                # Simplified basic stats with only min, max, and mean
                basic_stats = {}
                for metric in metrics_to_process:
                    stats = st.session_state.analysis.calculate_basic_stats(
                        selected_dataset_ids,
                        [metric]
                    )[metric]
                    basic_stats[metric] = {
                        'Min': stats['min'],
                        'Max': stats['max'],
                        'Mean': stats['mean']
                    }
                    st.write(f"Statistics for {metric}:")
                    st.write(pd.DataFrame([basic_stats[metric]]).round(2))
            
            else:  # Advanced
                # More detailed statistics (previous basic stats)
                advanced_stats = st.session_state.analysis.calculate_basic_stats(
                    selected_dataset_ids,
                    metrics_to_process
                )
                for metric, stats in advanced_stats.items():
                    st.write(f"Statistics for {metric}:")
                    st.write(pd.DataFrame([stats]).round(2))
                
                # New heatmap visualization
                st.subheader("Value Distribution Heatmap")
                for metric in metrics_to_process:
                    if st.checkbox(f"Show distribution heatmap for {metric}", key=f"heatmap_{metric}"):
                        # Get data for the metric
                        metric_data = plot_data[metric].dropna()
                        
                        if len(metric_data) > 0:
                            # Create bins for the heatmap
                            bins = np.linspace(metric_data.min(), metric_data.max(), 20)
                            
                            # Create timestamp bins (hourly)
                            plot_data['hour'] = pd.to_datetime(plot_data['timestamp']).dt.hour
                            
                            # Calculate 2D histogram
                            hist, x_edges, y_edges = np.histogram2d(
                                plot_data['hour'],
                                plot_data[metric],
                                bins=[24, bins]
                            )
                            
                            # Create heatmap
                            fig_heat = go.Figure(data=go.Heatmap(
                                z=hist.T,
                                x=np.arange(24),  # Hours
                                y=[(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)],
                                colorscale='Viridis',
                                name=metric
                            ))
                            
                            fig_heat.update_layout(
                                title=f"Distribution Heatmap for {metric}",
                                xaxis_title="Hour of Day",
                                yaxis_title=metric,
                                height=400
                            )
                            
                            st.plotly_chart(fig_heat, use_container_width=True)
                        else:
                            st.warning(f"No valid data available for {metric} heatmap")