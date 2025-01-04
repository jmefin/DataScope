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

# Add version info
st.sidebar.markdown("---")
st.sidebar.markdown("**Version:** 0.0.1")

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

  difference_metrics = []
  if show_differences and len(st.session_state.uploaded_files) >= 2:
      difference_metrics = st.sidebar.multiselect(
          "Select metrics for difference calculation:",
          primary_metrics,
          default=[primary_metrics[0]] if primary_metrics else []
      )
      reference_file = st.sidebar.selectbox(
          "Select reference file:",
          list(st.session_state.uploaded_files.keys()),
          index=0
      )

  # Limit values setup
  limit_values = {}
  metrics_to_process = primary_metrics.copy()
  if use_secondary_axis and secondary_metric:
      metrics_to_process.append(secondary_metric)

  for metric in metrics_to_process:
      if metric in DEFAULT_LIMITS:
          limit_values[metric] = setup_limit_values(metric, DEFAULT_LIMITS[metric], "main")

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

      # Statistical analysis
      st.sidebar.subheader("Statistical Analysis")
      stats_level = st.sidebar.radio(
          "Select statistics level:",
          ["Basic", "Advanced"],
          index=0,
          key="stats_level_main"
      )
      show_stats = st.checkbox("Show Statistical Summary", key="show_stats_main")
      
      if show_stats:
          st.subheader("Statistical Summary")
          
          if stats_level == "Basic":
              # Simplified basic statistics showing only min, max, and average
              basic_stats = st.session_state.analysis.calculate_basic_stats(
                  selected_dataset_ids,
                  metrics_to_process
              )
              
              # Create a simplified DataFrame with only min, max, and average
              simplified_stats = {}
              for metric, stats in basic_stats.items():
                  simplified_stats[metric] = {
                      'min': stats['min'],
                      'max': stats['max'],
                      'average': stats['mean']
                  }
              
              # Display the simplified statistics
              st.write("Basic Statistics (min, max, average):")
              st.write(pd.DataFrame(simplified_stats).round(2))
          
          else:  # Advanced
              # Correlation analysis
              if len(metrics_to_process) > 1:
                  st.subheader("Correlation Analysis")
                  corr_matrix = st.session_state.analysis.calculate_correlations(
                      selected_dataset_ids,
                      metrics_to_process
                  )
                  st.write(corr_matrix.round(3))
              
              # Heatmap visualization
              st.subheader("Heatmap Visualization")
              for metric in metrics_to_process:
                  if st.checkbox(f"Show heatmap for {metric}", key=f"show_heatmap_{metric}"):
                      # Get data for the selected metric
                      metric_data = st.session_state.db_manager.get_dataset_data(
                          selected_dataset_ids,
                          [metric]
                      )
                      
                      # Extract hour from timestamp
                      metric_data['hour'] = pd.to_datetime(metric_data['timestamp']).dt.hour
                      
                      # Create heatmap data
                      heatmap_data = metric_data.groupby(['hour', 'filename'])[metric].mean().unstack()
                      
                      # Create heatmap figure with swapped axes
                      fig_heatmap = go.Figure(data=go.Heatmap(
                          z=heatmap_data.values.T,
                          x=list(range(24)),
                          y=heatmap_data.columns,
                          colorscale='Viridis'
                      ))
                      
                      # Update layout with swapped axes
                      fig_heatmap.update_layout(
                          title=f"{metric} Heatmap",
                          xaxis_title="Hour of Day",
                          yaxis_title="Files",
                          xaxis=dict(
                              tickmode='array',
                              tickvals=list(range(24)),
                              ticktext=[f"{h:02d}:00" for h in range(24)]
                          )
                      )
                      
                      st.plotly_chart(fig_heatmap, use_container_width=True)
              
              # Full statistics (replacing Trend analysis)
              st.subheader("Detailed Statistics")
              full_stats = st.session_state.analysis.calculate_basic_stats(
                  selected_dataset_ids,
                  metrics_to_process
              )
              for metric, stats in full_stats.items():
                  st.write(f"Detailed statistics for {metric}:")
                  st.write(pd.DataFrame([stats]).round(2))

      # Difference plot
      if show_differences and difference_metrics and len(selected_files) >= 2:
          st.subheader("Difference Plot")
          
          fig_diff = go.Figure()
          
          # Convert reference data to hourly averages
          reference_data = plot_data[plot_data['filename'] == reference_file].copy()
          reference_data['hour'] = pd.to_datetime(reference_data['timestamp']).dt.floor('h')
          numeric_columns = reference_data.select_dtypes(include=[np.number]).columns
          reference_data = reference_data.groupby('hour')[numeric_columns].mean().reset_index()
          reference_data = reference_data.rename(columns={'hour': 'timestamp'})
          
          for file_name in selected_files:
              if file_name != reference_file:
                  # Convert comparison data to hourly averages
                  comparison_data = plot_data[plot_data['filename'] == file_name].copy()
                  comparison_data['hour'] = pd.to_datetime(comparison_data['timestamp']).dt.floor('h')
                  comparison_data = comparison_data.groupby('hour')[numeric_columns].mean().reset_index()
                  comparison_data = comparison_data.rename(columns={'hour': 'timestamp'})
                  
                  # Create merged dataframe with nearest timestamp matching
                  merged = pd.merge_asof(
                      reference_data.sort_values('timestamp'),
                      comparison_data.sort_values('timestamp'),
                      on='timestamp',
                      direction='nearest',
                      tolerance=pd.Timedelta('5min')
                  )
                  
                  if len(merged) > 0:
                      for metric in difference_metrics:
                          if f"{metric}_x" in merged.columns and f"{metric}_y" in merged.columns:
                              difference = merged[f"{metric}_y"] - merged[f"{metric}_x"]
                              if len(difference) > 0:
                                  fig_diff.add_trace(go.Scatter(
                                      x=merged['timestamp'],
                                      y=difference,
                                      mode='lines',
                                      name=f"{file_name} - {metric} diff",
                                      hovertemplate=f"<b>{file_name} - {metric} difference</b><br>" +
                                                  "Difference: %{y:.2f}<br>" + "Time: %{x}<extra></extra>"
                                  ))
                              else:
                                  st.warning(f"No valid difference data for {metric} between {reference_file} and {file_name}")
                      
                  else:
                      st.warning(f"Could not align timestamps between {reference_file} and {file_name} within 5 minute tolerance")
                  
          # Add zero line for reference
          fig_diff.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
          
          # Update difference plot layout
          fig_diff.update_layout(
              height=plot_height,
              hovermode='x unified',
              yaxis=dict(title='Difference'),
              showlegend=True
          )
          
          # Display the difference plot
          st.plotly_chart(fig_diff, use_container_width=True)
