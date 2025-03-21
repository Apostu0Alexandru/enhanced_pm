import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create directories for documentation and figures if they do not exist
docs_path = 'docs/figures'
os.makedirs(docs_path, exist_ok=True)

# Generate sample RUL prediction data (replace with your actual predictions)
def generate_sample_rul_data(num_engines=5, cycles_per_engine=100):
    data = []
    for engine in range(1, num_engines + 1):
        # Set random initial RUL value between 120-200
        initial_rul = np.random.randint(120, 200)
        
        # Generate cycles with decreasing RUL and some noise
        for cycle in range(1, cycles_per_engine + 1):
            # Linear degradation with noise
            rul = max(0, initial_rul - cycle + np.random.normal(0, 3))
            
            # Uncertainty increases as RUL decreases
            uncertainty = max(2, 5 + 0.1 * rul)
            
            data.append({
                'Engine': f'Engine #{engine}',
                'Cycle': cycle,
                'RUL': rul,
                'Uncertainty': uncertainty
            })
    
    return pd.DataFrame(data)

# Generate sample data
rul_predictions = generate_sample_rul_data(num_engines=5, cycles_per_engine=100)

# Create basic RUL prediction plot
fig = px.line(rul_predictions, x='Cycle', y='RUL', 
              color='Engine', error_y='Uncertainty')

fig.update_layout(
    title='Engine Remaining Useful Life Predictions',
    xaxis_title='Operating Cycles',
    yaxis_title='Remaining Useful Life (cycles)',
    legend_title='Engine ID',
    hovermode='closest',
    height=600,
    width=900
)

# Add threshold line for maintenance planning
fig.add_shape(
    type="line", line=dict(dash='dash', width=2, color="red"),
    y0=30, y1=30, x0=0, x1=100,
    name="Maintenance Threshold"
)
fig.add_annotation(
    x=10, y=30,
    text="Maintenance Threshold",
    showarrow=False,
    yshift=10,
    font=dict(color="red")
)

# Save as interactive HTML
fig.write_html('docs/figures/prognosis.html')

# Create advanced multi-panel dashboard
def create_advanced_dashboard(predictions_df):
    # Create multi-panel figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("RUL Predictions", "Degradation Rate", 
                       "Uncertainty Analysis", "Time-to-Threshold"),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "box"}, {"type": "bar"}]]
    )
    
    # Panel 1: RUL Predictions
    for engine in predictions_df['Engine'].unique():
        engine_data = predictions_df[predictions_df['Engine'] == engine]
        fig.add_trace(
            go.Scatter(
                x=engine_data['Cycle'], y=engine_data['RUL'],
                name=engine, mode='lines',
                line=dict(width=2),
                hovertemplate="Cycle: %{x}<br>RUL: %{y:.1f}"
            ),
            row=1, col=1
        )
    
    # Panel 2: Degradation Rate
    for engine in predictions_df['Engine'].unique():
        engine_data = predictions_df[predictions_df['Engine'] == engine]
        # Calculate degradation rate
        engine_data['Degradation'] = engine_data['RUL'].diff() * -1
        fig.add_trace(
            go.Scatter(
                x=engine_data['Cycle'][1:], y=engine_data['Degradation'][1:],
                name=engine, mode='lines',
                hovertemplate="Cycle: %{x}<br>Rate: %{y:.2f}"
            ),
            row=1, col=2
        )
    
    # Panel 3: Uncertainty Analysis
    fig.add_trace(
        go.Box(
            x=predictions_df['Engine'],
            y=predictions_df['Uncertainty'],
            name="Prediction Uncertainty"
        ),
        row=2, col=1
    )
    
    # Panel 4: Time to Threshold
    # Calculate remaining cycles to threshold (30 cycles)
    time_to_threshold = []
    for engine in predictions_df['Engine'].unique():
        engine_data = predictions_df[predictions_df['Engine'] == engine]
        # Find first cycle where RUL goes below 30
        threshold_cycle = engine_data[engine_data['RUL'] < 30]['Cycle'].min()
        if pd.isna(threshold_cycle):  # If never crosses threshold
            threshold_cycle = engine_data['Cycle'].max()
        time_to_threshold.append({
            'Engine': engine,
            'Cycles to Threshold': threshold_cycle
        })
    
    threshold_df = pd.DataFrame(time_to_threshold)
    fig.add_trace(
        go.Bar(
            x=threshold_df['Engine'],
            y=threshold_df['Cycles to Threshold'],
            name="Cycles to Maintenance"
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title="Comprehensive RUL Prognosis Dashboard",
        height=800,
        width=1200,
        showlegend=False,
        hovermode="closest"
    )
    
    # Add threshold line to RUL plot
    fig.add_shape(
        type="line", line=dict(dash='dash', width=2, color="red"),
        y0=30, y1=30, x0=0, x1=100, 
        row=1, col=1
    )
    
    return fig

# Create and save advanced dashboard
advanced_fig = create_advanced_dashboard(rul_predictions)
advanced_fig.write_html('docs/figures/advanced_prognosis_dashboard.html')

print("Interactive prognosis dashboards created and saved to 'docs/figures/'")
