import dash
from dash import dcc, html, dash_table, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load data
df = pd.read_csv('transformer_health_dataset.csv')
# Ensure label integrity if present
if 'fault' in df.columns:
    df['fault'] = pd.to_numeric(df['fault'], errors='coerce').fillna(0).round().astype(int)

# Optionally cap dataset size to improve responsiveness
MAX_ROWS = 2000
if len(df) > MAX_ROWS and 'fault' in df.columns:
    counts = df['fault'].value_counts().sort_index()
    # compute proportional per-class sample sizes
    sizes = (counts / counts.sum() * MAX_ROWS).round().astype(int)
    # adjust to ensure the total equals MAX_ROWS
    diff = MAX_ROWS - sizes.sum()
    if diff != 0:
        # distribute the remainder by adding to the largest classes first (or subtracting)
        order = sizes.sort_values(ascending=(diff < 0)).index
        for cls in order:
            if diff == 0:
                break
            sizes.loc[cls] += 1 if diff > 0 else -1
            diff += -1 if diff > 0 else 1
    parts = []
    for cls, n in sizes.items():
        part = df[df['fault'] == cls].sample(n=n, random_state=42, replace=False)
        parts.append(part)
    df = pd.concat(parts, axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)

feature_cols = [c for c in df.columns if c != 'fault']

# Train or load model (from notebook logic)
def train_or_load_model():
    models_dir = 'models'
    scaler_path = os.path.join(models_dir, 'scaler.joblib')
    model_path = os.path.join(models_dir, 'rf_transformer_health.joblib')

    os.makedirs(models_dir, exist_ok=True)

    scaler = None
    model = None

    if os.path.exists(scaler_path) and os.path.exists(model_path):
        scaler = joblib.load(scaler_path)
        model = joblib.load(model_path)
    else:
        X = df[feature_cols]
        y = df['fault']

        # Choose stratification only if every class has at least 2 samples
        use_stratify = False
        if 'fault' in df.columns:
            vc = y.value_counts()
            use_stratify = (vc.min() >= 2)

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                stratify=y if use_stratify else None,
                test_size=0.2,
                random_state=42
            )
        except ValueError:
            # Fallback: non-stratified split if stratification fails
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Hyperparameter tuning with cross-validation
        base_model = RandomForestClassifier(random_state=42, class_weight='balanced')
        param_grid = {
            'n_estimators': [150, 300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'max_features': ['sqrt', 'log2']
        }
        # Pick CV strategy robustly: prefer stratified if feasible
        cv = None
        if use_stratify:
            n_splits = 5
            if y_train.value_counts().min() < n_splits:
                n_splits = max(2, int(y_train.value_counts().min()))
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        else:
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
        search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring='roc_auc',
            n_jobs=-1,
            cv=cv,
            refit=True,
            verbose=0
        )
        search.fit(X_train_scaled, y_train)
        model = search.best_estimator_

        joblib.dump(scaler, scaler_path)
        joblib.dump(model, model_path)

    # Compute evaluation artifacts on a held-out test split (avoid training data)
    X_all = df[feature_cols]
    y_all = df['fault']
    # Evaluation split: try to stratify, otherwise fall back
    try:
        vc_all = y_all.value_counts()
        strat = y_all if (vc_all.min() >= 2) else None
        X_train_ev, X_test_ev, y_train_ev, y_test_ev = train_test_split(
            X_all, y_all, stratify=strat, test_size=0.2, random_state=42
        )
    except ValueError:
        X_train_ev, X_test_ev, y_train_ev, y_test_ev = train_test_split(
            X_all, y_all, test_size=0.2, random_state=42
        )
    # IMPORTANT: don't refit the scaler here; only transform using the trained scaler
    X_test_ev_scaled = scaler.transform(X_test_ev)
    y_pred_ev = model.predict(X_test_ev_scaled)
    acc = accuracy_score(y_test_ev, y_pred_ev)
    cm = confusion_matrix(y_test_ev, y_pred_ev)

    importances = getattr(model, 'feature_importances_', np.zeros(len(feature_cols)))
    feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False)

    return model, scaler, acc, cm, feat_imp

model, scaler, model_accuracy, cmatrix, feature_importance = train_or_load_model()

# Helper: Key metrics (adapted to dataset columns)
metrics = [
    {'label': 'Total Records', 'value': len(df)},
    {'label': 'Fault Cases', 'value': int(df['fault'].sum())},
    {'label': 'No Fault Cases', 'value': int((df['fault'] == 0).sum())},
    {'label': 'Avg. Oil Temp', 'value': f"{df.get('oil_temp', pd.Series(dtype=float)).mean():.1f}" if 'oil_temp' in df.columns else 'N/A'},
    {'label': 'Avg. Load Current', 'value': f"{df.get('load_current', pd.Series(dtype=float)).mean():.1f}" if 'load_current' in df.columns else 'N/A'},
]

# App initialization
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = 'Transight Dashboard'

# Styles
CONTENT_STYLE = {
    'margin': '40px',
    'background-color': '#f8f9fa',
    'minHeight': '100vh',
}
CARD_STYLE = {
    'background': '#fff', 'border-radius': '8px', 'box-shadow': '0 2px 8px rgba(0,0,0,0.07)',
    'padding': '18px 20px', 'margin': '10px', 'display': 'inline-block', 'minWidth': '160px',
    'textAlign': 'center', 'color': '#22223b', 'transition': 'transform 0.3s',
}
FOOTER_STYLE = {
    'width': '100%', 'background': '#22223b', 'color': '#fff', 'textAlign': 'center',
    'padding': '18px 0 10px 0', 'position': 'relative', 'bottom': 0, 'marginTop': '40px',
    'fontSize': '15px', 'letterSpacing': '1px', 'boxShadow': '0 -2px 8px rgba(0,0,0,0.07)'
}

# Main dashboard layout
def serve_layout():
    return html.Div([
        # Main content
        html.Div([
            html.H1('Transight Dashboard', style={'color': '#22223b', 'marginBottom': '10px'}),
            html.P('Monitor and analyze transformer sensor data for predictive maintenance.', style={'color': '#4a4e69', 'marginBottom': '30px'}),

            # Key metrics
            html.Div([
                html.Div([
                    html.Div([
                        html.H4(m['label'], style={'fontSize': '16px', 'marginBottom': '8px'}),
                        html.H2(m['value'], style={'fontWeight': 'bold', 'fontSize': '28px'})
                    ], style=CARD_STYLE)
                ], style={'display': 'inline-block'}) for m in metrics
            ], style={'marginBottom': '30px'}),

            # Fault distribution
            html.Div([
                html.H3('Fault Distribution', style={'color': '#22223b'}),
                dcc.Graph(
                    id='fault-dist',
                    figure=px.histogram(
                        df, x='fault', nbins=2, labels={'fault': 'Fault'},
                        title='Fault Distribution',
                        category_orders={'fault': [0, 1]},
                        color='fault',
                        color_discrete_map={0: '#4caf50', 1: '#e63946'}
                    ).update_layout(
                        plot_bgcolor='#f8f9fa',
                        paper_bgcolor='#f8f9fa',
                        font_color='#22223b',
                        margin=dict(l=0, r=0, t=40, b=0)
                    )
                ),
            ], style={'marginBottom': '40px'}),

            # Parameter visualization
            html.Div([
                html.H3('Sensor Parameter Visualization', style={'color': '#22223b'}),
                html.Label('Select Parameter:', style={'fontWeight': 'bold', 'marginRight': '10px'}),
                dcc.Dropdown(
                    id='param-dropdown',
                    options=[{'label': col.title().replace('_', ' '), 'value': col} for col in feature_cols],
                    value=feature_cols[0] if feature_cols else None,
                    clearable=False,
                    style={'width': '250px', 'display': 'inline-block'}
                ),
                dcc.Graph(id='param-graph'),
            ], style={'marginBottom': '40px'}),

            # Model insights
            html.Div([
                html.H3('Model Insights', style={'color': '#22223b'}),
                html.Div([
                    dcc.Graph(
                        id='feature-importance',
                        figure=px.bar(
                            feature_importance.sort_values(ascending=True),
                            x=feature_importance.sort_values(ascending=True).values,
                            y=feature_importance.sort_values(ascending=True).index,
                            orientation='h',
                            title='Feature Importance (RandomForest)'
                        ).update_layout(
                            plot_bgcolor='#f8f9fa', paper_bgcolor='#f8f9fa', font_color='#22223b',
                            margin=dict(l=0, r=0, t=40, b=0)
                        )
                    )
                ], style={'marginBottom': '20px'}),
                html.Div([
                    dcc.Graph(
                        id='confusion-matrix',
                        figure=px.imshow(
                            cmatrix,
                            text_auto=True,
                            x=['Healthy', 'Faulty'], y=['Healthy', 'Faulty'],
                            color_continuous_scale='Blues',
                            title=f'Confusion Matrix (Accuracy: {model_accuracy:.2f})'
                        ).update_layout(
                            plot_bgcolor='#f8f9fa', paper_bgcolor='#f8f9fa', font_color='#22223b',
                            margin=dict(l=0, r=0, t=40, b=0)
                        )
                    )
                ])
            ], style={'marginBottom': '40px'}),

            # Inference form
            html.Div([
                html.H3('Predict Transformer Health', style={'color': '#22223b'}),
                html.Div([
                    html.Div([
                        html.Label(col.title().replace('_', ' ')),
                        dcc.Input(
                            id={'type': 'feature-input', 'index': col},
                            type='number',
                            value=float(df[col].median()) if pd.api.types.is_numeric_dtype(df[col]) else None,
                            debounce=False,
                            style={'width': '100%'}
                        )
                    ], style={'display': 'inline-block', 'width': '180px', 'marginRight': '10px', 'marginBottom': '10px'})
                    for col in feature_cols
                ]),
                html.Button('Predict', id='predict-btn', n_clicks=0, style={'marginTop': '10px'}),
                html.Div(id='prediction-output', style={'marginTop': '15px', 'fontWeight': 'bold', 'color': '#22223b'}),
                html.Div(id='prediction-prob', style={'marginTop': '5px', 'color': '#4a4e69'}),
                dcc.Graph(id='prob-gauge'),
                dcc.Graph(id='input-zscores'),
            ], style={'marginBottom': '40px'}),

            # Data explorer
            html.Div([
                html.H3('Data Explorer', style={'color': '#22223b'}),
                html.Label('Filter by Fault Status:', style={'fontWeight': 'bold', 'marginRight': '10px'}),
                dcc.RadioItems(
                    id='fault-filter',
                    options=[{'label': 'All', 'value': 'all'}, {'label': 'No Fault', 'value': 0}, {'label': 'Fault', 'value': 1}],
                    value='all',
                    inline=True,
                    style={'marginBottom': '10px'}
                ),
                dash_table.DataTable(
                    id='filtered-table',
                    columns=[{"name": i.title().replace('_', ' '), "id": i} for i in df.columns],
                    data=df.to_dict('records'),
                    page_size=10,
                    style_table={'overflowX': 'auto'},
                    style_header={'backgroundColor': '#22223b', 'color': 'white', 'fontWeight': 'bold'},
                    style_data={'backgroundColor': '#fff', 'color': '#22223b'},
                    style_cell={'padding': '8px', 'textAlign': 'center'},
                    style_data_conditional=[
                        {
                            'if': {'column_id': 'fault', 'filter_query': '{fault} eq 1'},
                            'backgroundColor': '#ffe5e5',
                            'color': '#e63946',
                        },
                        {
                            'if': {'column_id': 'fault', 'filter_query': '{fault} eq 0'},
                            'backgroundColor': '#e5ffe5',
                            'color': '#4caf50',
                        },
                    ],
                )
            ], style={'marginBottom': '40px'}),

            # About section
            html.Div([
                html.Hr(),
                html.H4('About', style={'color': '#22223b'}),
                html.P('This dashboard helps monitor transformer health using sensor data. Use the visualizations and filters to identify patterns and potential faults for predictive maintenance.', style={'color': '#4a4e69'}),
            ], style={'marginTop': '40px', 'marginBottom': '20px'}),

            # Footer
            html.Footer([
                html.Span('© 2025 Transformer Health Monitoring | Designed with '),
                html.Span(style={'color': '#e63946', 'fontWeight': 'bold', 'fontSize': '18px'}),
                html.Span(' by Transight'),
            ], style=FOOTER_STYLE),
        ], style=CONTENT_STYLE),
    ])

app.layout = serve_layout

# Callbacks
@app.callback(
    Output('param-graph', 'figure'),
    Input('param-dropdown', 'value')
)
def update_param_graph(param):
    fig = px.histogram(
        df, x=param, color='fault', barmode='overlay',
        title=f'Distribution of {param.title().replace("_", " ")} by Fault Status',
        color_discrete_map={0: '#4caf50', 1: '#e63946'}
    )
    fig.update_layout(
        plot_bgcolor='#f8f9fa',
        paper_bgcolor='#f8f9fa',
        font_color='#22223b',
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig

@app.callback(
    Output('filtered-table', 'data'),
    Input('fault-filter', 'value')
)
def update_table(fault_value):
    if fault_value == 'all':
        return df.to_dict('records')
    else:
        return df[df['fault'] == int(fault_value)].to_dict('records')

@app.callback(
    Output('prob-gauge', 'figure'),
    Output('input-zscores', 'figure'),
    *[Input({'type': 'feature-input', 'index': col}, 'value') for col in feature_cols]
)
def update_live_gauge(*values):
    # Build a valid numeric sample from inputs; fallback to medians when None/invalid
    try:
        numeric_values = []
        for i, col in enumerate(feature_cols):
            val = values[i] if i < len(values) else None
            if val is None or (isinstance(val, str) and val.strip() == ''):
                fallback = float(df[col].median()) if pd.api.types.is_numeric_dtype(df[col]) else 0.0
                numeric_values.append(fallback)
            else:
                numeric_values.append(float(val))
        sample = pd.DataFrame([numeric_values], columns=feature_cols)
    except Exception:
        return go.Figure(), go.Figure()
    try:
        xs = scaler.transform(sample)
        proba = float(model.predict_proba(xs)[0, 1])
        # Gauge figure for probability
        gauge_color = '#e63946' if proba >= 0.5 else '#4caf50'
        gauge_fig = go.Figure(go.Indicator(
            mode='gauge+number',
            value=proba * 100.0,
            number={'suffix': '%'},
            title={'text': 'Fault Probability'},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': gauge_color},
                'steps': [
                    {'range': [0, 50], 'color': '#e5ffe5'},
                    {'range': [50, 100], 'color': '#ffe5e5'}
                ]
            }
        ))
        gauge_fig.update_layout(plot_bgcolor='#f8f9fa', paper_bgcolor='#f8f9fa', font_color='#22223b', margin=dict(l=0, r=0, t=40, b=0))

        # Z-score bar chart for input vs dataset
        df_num = df[feature_cols].select_dtypes(include=[np.number])
        means = df_num.mean()
        stds = df_num.std(ddof=0).replace(0, np.nan)
        sample_num = sample[df_num.columns]
        zscores = ((sample_num.iloc[0] - means) / stds).fillna(0.0)
        z_abs_sorted = zscores.abs().sort_values(ascending=False)
        z_fig = px.bar(
            x=z_abs_sorted.values,
            y=[c.title().replace('_', ' ') for c in z_abs_sorted.index],
            orientation='h',
            title='Per-Feature Deviation (|z-score|)'
        )
        z_fig.update_layout(plot_bgcolor='#f8f9fa', paper_bgcolor='#f8f9fa', font_color='#22223b', margin=dict(l=0, r=0, t=40, b=0))

        return gauge_fig, z_fig
    except Exception:
        return go.Figure(), go.Figure()

@app.callback(
    Output('prediction-output', 'children'),
    Output('prediction-prob', 'children'),
    Input('predict-btn', 'n_clicks'),
    [State({'type': 'feature-input', 'index': col}, 'value') for col in feature_cols]
)
def run_inference(n_clicks, *values):
    if not n_clicks:
        return '', ''
    # Build sample
    numeric_values = []
    for i, col in enumerate(feature_cols):
        val = values[i] if i < len(values) else None
        if val is None or (isinstance(val, str) and val.strip() == ''):
            fallback = float(df[col].median()) if pd.api.types.is_numeric_dtype(df[col]) else 0.0
            numeric_values.append(fallback)
        else:
            numeric_values.append(float(val))
    sample = pd.DataFrame([numeric_values], columns=feature_cols)
    try:
        xs = scaler.transform(sample)
        pred = model.predict(xs)[0]
        proba = float(model.predict_proba(xs)[0, 1])
        if pred == 1:
            label_node = html.Span('Faulty: Yes', style={'color': '#e63946', 'fontWeight': 'bold'})
        else:
            label_node = html.Span('Faulty: No', style={'color': '#4caf50', 'fontWeight': 'bold'})
        return label_node, f'Fault Probability: {proba:.4f}'
    except Exception as e:
        return 'Prediction error', str(e)

if __name__ == '__main__':
    app.run(debug=True) 