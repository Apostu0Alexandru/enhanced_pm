# visualization/dashboard.py
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from data_processing.data_loader import NASADataProcessor
from data_processing.hybrid_generator import HybridDataEngine
from models.fusion_model import HybridPredictiveModel
from monitoring.alerts import MaintenanceAlertSystem

# Ensure output directory exists
os.makedirs('output', exist_ok=True)

# Initialize data processor and model
processor = NASADataProcessor('nasa_data/train_FD001.txt')
features, labels = processor.preprocess()

# Load or train model
try:
    model = HybridPredictiveModel(input_dim=features.shape[1])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Try to load weights if model file exists
    if os.path.exists('models/hybrid_model.h5'):
        model.load_weights('models/hybrid_model.h5')
        print("Loaded existing model from models/hybrid_model.h5")
    else:
        print("No existing model found, using untrained model")
except Exception as e:
    print(f"Error loading model: {e}")
    # Create a simple fallback model for demonstration
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    model = Sequential([
        Dense(10, activation='relu', input_shape=(features.shape[1],)),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')

# Initialize alert system
alert_system = MaintenanceAlertSystem(features)

def get_real_data():
    """Get real data from NASA dataset and model predictions"""
    # Get unique unit IDs from the dataset
    unit_ids = np.unique(processor.data['unit'])[:3]  # Get first 3 units
    equipment_ids = [f'FD001-Unit{int(unit_id)}' for unit_id in unit_ids]
    
    # Current time for timestamps
    current_time = datetime.now()
    timestamps = [current_time - timedelta(minutes=i) for i in range(30)]
    timestamps.reverse()
    
    # Get RUL predictions for each unit
    rul_data = {}
    health_scores = {}
    alerts_data = {}
    
    for i, eq_id in enumerate(equipment_ids):
        unit_id = int(eq_id.split('-Unit')[1])
        
        # Get data for this unit
        unit_data = processor.data[processor.data['unit'] == unit_id]
        
        if len(unit_data) > 0:
            # Get features for this unit
            unit_features = unit_data.drop(columns=['unit', 'cycle', 'RUL']).values
            
            # Get predictions
            predictions = model.predict(unit_features).flatten()
            
            # Store last 30 predictions (or fewer if not enough data)
            rul_values = predictions[-30:] if len(predictions) >= 30 else np.pad(
                predictions, (30 - len(predictions), 0), 'constant', constant_values=predictions[0]
            )
            rul_data[eq_id] = rul_values.tolist()
            
            # Calculate health scores based on RUL
            min_rul = np.min(rul_values)
            max_rul = np.max(rul_values)
            health_pct = min(100, max(0, min_rul / 100 * 100))
            
            health_scores[eq_id] = {
                'sensor': int(health_pct * 0.9 + np.random.randint(0, 10)),
                'maintenance': int(health_pct * 0.8 + np.random.randint(0, 20)),
                'integrated': int(health_pct * 0.85 + np.random.randint(0, 15))
            }
            
            # Generate alert status based on health
            status = 'Critical' if health_scores[eq_id]['integrated'] < 70 else \
                     'Warning' if health_scores[eq_id]['integrated'] < 85 else 'Normal'
                     
            # Last maintenance would be start of data collection
            last_maintenance = (current_time - timedelta(days=len(unit_data))).strftime('%Y-%m-%d')
            
            # Next scheduled based on RUL
            next_days = max(1, int(min_rul / 2))
            next_scheduled = (current_time + timedelta(days=next_days)).strftime('%Y-%m-%d')
            
            alerts_data[eq_id] = {
                'status': status,
                'last_maintenance': last_maintenance,
                'next_scheduled': next_scheduled,
                'forecasted_days': next_days,
                'deviation': np.random.randint(-5, 5)  # Random deviation for demo
            }
        else:
            # Fallback if no data for this unit
            rul_data[eq_id] = [100 - i for i in range(30)]
            health_scores[eq_id] = {'sensor': 80, 'maintenance': 75, 'integrated': 78}
            alerts_data[eq_id] = {
                'status': 'Normal',
                'last_maintenance': (current_time - timedelta(days=30)).strftime('%Y-%m-%d'),
                'next_scheduled': (current_time + timedelta(days=15)).strftime('%Y-%m-%d'),
                'forecasted_days': 15,
                'deviation': 0
            }
    
    return {
        'timestamps': timestamps,
        'rul_data': rul_data,
        'health_scores': health_scores,
        'alerts_data': alerts_data
    }

def calculate_mtbf(unit_id):
    """Calculate MTBF from NASA dataset for given unit"""
    # Extract unit number
    unit_num = int(unit_id.split('-Unit')[1])
    
    # Filter for this unit
    unit_data = processor.data[processor.data['unit'] == unit_num]
    
    # In NASA dataset, MTBF can be represented by total cycles
    return unit_data['cycle'].max() if len(unit_data) > 0 else 200

def calculate_mttr(unit_id):
    """Calculate MTTR (Mean Time To Repair)"""
    # For simulation, use a random value between 2-8 hours
    return np.random.randint(2, 8)

def calculate_oee(unit_id):
    """Calculate OEE (Overall Equipment Effectiveness)"""
    # Extract unit number
    unit_num = int(unit_id.split('-Unit')[1])
    
    # Filter for this unit
    unit_data = processor.data[processor.data['unit'] == unit_num]
    
    # Calculate OEE based on RUL (higher RUL = higher OEE)
    if len(unit_data) > 0:
        min_rul = unit_data['RUL'].min()
        max_rul = unit_data['RUL'].max()
        oee = min(95, max(60, int(min_rul / max_rul * 100)))
        return oee
    return 85  # Default value

def calculate_pmp(unit_id):
    """Calculate PMP (Predictive Maintenance Performance)"""
    # For simulation, use a value between 70-90%
    return np.random.randint(70, 90)

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("NASA Turbofan Predictive Maintenance Dashboard", style={'textAlign': 'center'}),
        html.P(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", id='update-time')
    ], className='header'),
    
    # Equipment selector
    html.Div([
        html.Label("Select Equipment:"),
        dcc.Dropdown(
            id='equipment-dropdown',
            options=[
                {'label': f'FD001-Unit{i}', 'value': f'FD001-Unit{i}'} 
                for i in range(1, 4)  # First 3 units from NASA dataset
            ],
            value='FD001-Unit1'
        )
    ], style={'width': '50%', 'margin': '20px auto'}),
    
    # Main dashboard content
    html.Div([
        # Left column - Health Scores
        html.Div([
            html.H3("Equipment Health Scores"),
            html.Div(id='health-score-indicators'),
            html.Div(id='maintenance-recommendation', className='recommendation-box')
        ], className='dashboard-column'),
        
        # Right column - RUL Prediction
        html.Div([
            html.H3("Remaining Useful Life Prediction"),
            dcc.Graph(id='rul-prediction-graph'),
            html.Div([
                html.H4("Maintenance Schedule"),
                html.Div(id='maintenance-schedule')
            ])
        ], className='dashboard-column')
    ], className='dashboard-row'),
    
    # Bottom section - Metrics & KPIs
    html.Div([
        html.H3("Performance Metrics"),
        html.Div([
            # MTBF, MTTR, OEE metrics
            html.Div([
                html.H4("MTBF"),
                html.Div(id='mtbf-value', className='metric-value')
            ], className='metric-box'),
            html.Div([
                html.H4("MTTR"),
                html.Div(id='mttr-value', className='metric-value')
            ], className='metric-box'),
            html.Div([
                html.H4("OEE"),
                html.Div(id='oee-value', className='metric-value')
            ], className='metric-box'),
            html.Div([
                html.H4("PMP"),
                html.Div(id='pmp-value', className='metric-value')
            ], className='metric-box')
        ], className='metrics-container')
    ], className='dashboard-row'),
    
    # Alert history table
    html.Div([
        html.H3("Recent Alerts"),
        dash_table.DataTable(
            id='alerts-table',
            columns=[
                {'name': 'Timestamp', 'id': 'timestamp'},
                {'name': 'Equipment', 'id': 'equipment'},
                {'name': 'Alert Type', 'id': 'type'},
                {'name': 'Severity', 'id': 'severity'},
                {'name': 'Status', 'id': 'status'}
            ],
            style_data_conditional=[
                {
                    'if': {'filter_query': '{severity} = "Critical"'},
                    'backgroundColor': '#ffcccc',
                    'color': '#990000'
                },
                {
                    'if': {'filter_query': '{severity} = "Warning"'},
                    'backgroundColor': '#ffffcc',
                    'color': '#999900'
                }
            ]
        )
    ], className='dashboard-row'),
    
    # Refresh interval
    dcc.Interval(
        id='refresh-interval',
        interval=10000,  # 10 seconds
        n_intervals=0
    )
], className='dashboard-container')

# Callbacks
@app.callback(
    [Output('rul-prediction-graph', 'figure'),
     Output('health-score-indicators', 'children'),
     Output('maintenance-recommendation', 'children'),
     Output('maintenance-schedule', 'children'),
     Output('mtbf-value', 'children'),
     Output('mttr-value', 'children'),
     Output('oee-value', 'children'),
     Output('pmp-value', 'children'),
     Output('alerts-table', 'data'),
     Output('update-time', 'children')],
    [Input('refresh-interval', 'n_intervals'),
     Input('equipment-dropdown', 'value')]
)
def update_dashboard(n_intervals, equipment_id):
    # Get data from NASA dataset and model
    data = get_real_data()
    
    # 1. RUL Prediction Graph
    rul_figure = {
        'data': [
            go.Scatter(
                x=[t.strftime('%H:%M:%S') for t in data['timestamps']],
                y=data['rul_data'][equipment_id],
                mode='lines+markers',
                name='RUL Prediction',
                line=dict(color='#2c3e50')
            ),
            # Add threshold line
            go.Scatter(
                x=[data['timestamps'][0].strftime('%H:%M:%S'), data['timestamps'][-1].strftime('%H:%M:%S')],
                y=[50, 50],
                mode='lines',
                name='Critical Threshold',
                line=dict(color='red', dash='dash')
            )
        ],
        'layout': go.Layout(
            title='Remaining Useful Life Trend',
            xaxis=dict(title='Time'),
            yaxis=dict(title='RUL (Cycles)'),
            height=400,
            margin=dict(l=40, r=40, t=40, b=40)
        )
    }
    
    # 2. Health Score Indicators
    health_scores = data['health_scores'][equipment_id]
    health_indicators = html.Div([
        html.Div([
            html.H4("Sensor Health"),
            html.Div(f"{health_scores['sensor']}%", className=f"health-score {'good' if health_scores['sensor'] >= 85 else 'warning' if health_scores['sensor'] >= 70 else 'critical'}")
        ], className='health-indicator'),
        html.Div([
            html.H4("Maintenance Health"),
            html.Div(f"{health_scores['maintenance']}%", className=f"health-score {'good' if health_scores['maintenance'] >= 85 else 'warning' if health_scores['maintenance'] >= 70 else 'critical'}")
        ], className='health-indicator'),
        html.Div([
            html.H4("Integrated Health"),
            html.Div(f"{health_scores['integrated']}%", className=f"health-score {'good' if health_scores['integrated'] >= 85 else 'warning' if health_scores['integrated'] >= 70 else 'critical'}")
        ], className='health-indicator')
    ])
    
    # 3. Maintenance Recommendation
    alert_status = data['alerts_data'][equipment_id]['status']
    recommendation = html.Div([
        html.H4(f"Status: {alert_status}", className=f"status-{alert_status.lower()}"),
        html.P(
            "Immediate maintenance recommended. Schedule downtime within 48 hours." if alert_status == 'Critical' else
            "Monitor closely and prepare for maintenance in the next 2 weeks." if alert_status == 'Warning' else
            "Equipment operating normally. Follow regular maintenance schedule."
        )
    ])
    
    # 4. Maintenance Schedule
    schedule = html.Div([
        html.Div([
            html.Strong("Last Maintenance: "),
            html.Span(data['alerts_data'][equipment_id]['last_maintenance'])
        ]),
        html.Div([
            html.Strong("Next Scheduled: "),
            html.Span(data['alerts_data'][equipment_id]['next_scheduled'])
        ]),
        html.Div([
            html.Strong("Forecasted Days to Maintenance: "),
            html.Span(f"{data['alerts_data'][equipment_id]['forecasted_days']} days")
        ]),
        html.Div([
            html.Strong("Deviation from Schedule: "),
            html.Span(f"{data['alerts_data'][equipment_id]['deviation']} days", 
                     style={'color': 'red' if data['alerts_data'][equipment_id]['deviation'] < 0 else 'green'})
        ])
    ])
    
    # 5-8. KPI Metrics
   # In the update_dashboard callback:
# Replace random metrics with actual calculations
    mtbf = f"{calculate_mtbf(equipment_id)} hours"  # Mean Time Between Failures
    mttr = f"{calculate_mttr(equipment_id)} hours"  # Mean Time To Repair
    oee = f"{calculate_oee(equipment_id)}%"  # Overall Equipment Effectiveness
    pmp = f"{calculate_pmp(equipment_id)}%"  # Predictive Maintenance Performance

    
    # 9. Alerts Table
    alerts = []
    for i in range(5):
        severity = "Critical" if i == 0 and alert_status == "Critical" else "Warning" if i < 2 and alert_status in ["Warning", "Critical"] else "Normal"
        alerts.append({
            'timestamp': (datetime.now() - timedelta(hours=i*4)).strftime('%Y-%m-%d %H:%M'),
            'equipment': equipment_id,
            'type': np.random.choice(['Vibration', 'Temperature', 'Pressure', 'Oil Level']),
            'severity': severity,
            'status': 'Open' if severity != 'Normal' else 'Closed'
        })
    
    # 10. Update time
    update_time = f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    return rul_figure, health_indicators, recommendation, schedule, mtbf, mttr, oee, pmp, alerts, update_time

# Add CSS for styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        <title>Predictive Maintenance Dashboard</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        {%css%}
        <style>
            .dashboard-container {
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            .header {
                margin-bottom: 20px;
                border-bottom: 1px solid #ddd;
                padding-bottom: 10px;
            }
            .dashboard-row {
                display: flex;
                flex-wrap: wrap;
                margin-bottom: 20px;
                background-color: #f9f9f9;
                border-radius: 5px;
                padding: 15px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .dashboard-column {
                flex: 1;
                min-width: 300px;
                padding: 10px;
            }
            .health-indicator {
                display: inline-block;
                width: 30%;
                text-align: center;
                margin: 10px;
            }
            .health-score {
                font-size: 24px;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            .good {
                background-color: #d4edda;
                color: #155724;
            }
            .warning {
                background-color: #fff3cd;
                color: #856404;
            }
            .critical {
                background-color: #f8d7da;
                color: #721c24;
            }
            .recommendation-box {
                margin-top: 20px;
                padding: 15px;
                border-radius: 5px;
                background-color: #e2e3e5;
            }
            .status-critical {
                color: #721c24;
            }
            .status-warning {
                color: #856404;
            }
            .status-normal {
                color: #155724;
            }
            .metrics-container {
                display: flex;
                justify-content: space-between;
            }
            .metric-box {
                flex: 1;
                text-align: center;
                padding: 10px;
                margin: 0 10px;
                background-color: white;
                border-radius: 5px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            .metric-value {
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
            }
        </style>
        {%metas%}
        {%favicon%}
        {
            %app_entry%
        }
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
 </html>
 '''

if __name__ == '__main__':
    app.run_server(debug=True)
