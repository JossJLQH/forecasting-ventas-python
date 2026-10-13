import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing

print("1. Generando serie de tiempo sintética con tendencia y estacionalidad...")

# 1. Simulación de datos históricos de ventas/demanda diaria (365 días)
dates = pd.date_range(start='2025-01-01', periods=365, freq='D')
np.random.seed(42)

# Componentes: Tendencia ascendente + Estacionalidad semanal + Ruido aleatorio
trend = np.linspace(100, 250, 365)
seasonality = 30 * np.sin(2 * np.pi * dates.dayofweek / 7)
noise = np.random.normal(0, 10, 365)
sales = trend + seasonality + noise

df_history = pd.DataFrame({'Fecha': dates, 'Ventas': sales})

print("2. Entrenando modelo Holt-Winters y calculando métricas de error...")

# 2. Modelo de pronóstico Holt-Winters (Soporta tendencia y estacionalidad)
model = ExponentialSmoothing(
    df_history['Ventas'], 
    trend='add', 
    seasonal='add', 
    seasonal_periods=7
).fit()

# Cálculo de métricas de error (In-sample: comparación entre valores reales y ajustados)
fitted_values = model.fittedvalues
mae = np.mean(np.abs(df_history['Ventas'] - fitted_values))
rmse = np.sqrt(np.mean((df_history['Ventas'] - fitted_values) ** 2))

print(f"📊 Resultados de Evaluación:")
print(f"   • MAE (Error Absoluto Medio): {mae:.2f}")
print(f"   • RMSE (Raíz del Error Cuadrático Medio): {rmse:.2f}")

# Pronóstico a 30 días futuros
future_days = 30
forecast_values = model.forecast(future_days)
future_dates = pd.date_range(start=dates[-1] + pd.Timedelta(days=1), periods=future_days, freq='D')

df_forecast = pd.DataFrame({'Fecha': future_dates, 'Pronóstico': forecast_values})

print("3. Generando visualización interactiva con Plotly...")

# 3. Creación de la gráfica interactiva
fig = go.Figure()

# Histórico real
fig.add_trace(go.Scatter(
    x=df_history['Fecha'], 
    y=df_history['Ventas'],
    mode='lines',
    name='Histórico Real',
    line=dict(color='#38bdf8', width=2)
))

# Pronóstico futuro
fig.add_trace(go.Scatter(
    x=df_forecast['Fecha'], 
    y=df_forecast['Pronóstico'],
    mode='lines+markers',
    name='Pronóstico (30 días)',
    line=dict(color='#8b5cf6', width=2.5, dash='dash')
))

# 4. Agregar tarjeta de métricas dentro de la gráfica
metrics_text = f"<b>Métricas del Modelo:</b><br>• MAE: {mae:.2f}<br>• RMSE: {rmse:.2f}"
fig.add_annotation(
    xref="paper", yref="paper",
    x=0.03, y=0.90,
    text=metrics_text,
    showarrow=False,
    font=dict(size=12, color="#f8fafc", family="Inter, sans-serif"),
    align="left",
    bgcolor="rgba(30, 41, 59, 0.85)",
    bordercolor="rgba(56, 189, 248, 0.3)",
    borderwidth=1,
    borderpad=10
)

# Estilo Dark Mode moderno
fig.update_layout(
    title='<b>Pronóstico de Demanda y Ventas (Series de Tiempo)</b>',
    title_font_size=20,
    paper_bgcolor='#090d16',
    plot_bgcolor='#1e293b',
    font=dict(color='#f8fafc', family='Inter, sans-serif'),
    xaxis=dict(title='Fecha', gridcolor='rgba(255,255,255,0.05)'),
    yaxis=dict(title='Unidades / Ventas', gridcolor='rgba(255,255,255,0.05)'),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode='x unified'
)

# Guardar resultado en un archivo HTML interactivo autónomo
fig.write_html("pronostico_interactivo.html")
print("✅ ¡Pronóstico generado con éxito! Abre 'pronostico_interactivo.html' en tu navegador.")

# Mostrar ventana interactiva
fig.show()