import streamlit as st
import requests
import plotly.graph_objects as go
import numpy as np

st.title("Visualización de Predicciones")

try:
    # Obtener datos de la API
    response = requests.get("http://api:5555/data_for_plot")
    data = response.json()
    
    if data['success']:
        # Crear gráfica
        fig = go.Figure()
        
        # Añadir puntos de predicción
        fig.add_trace(go.Scatter(
            x=data['y_test'],
            y=data['y_pred'],
            mode='markers',
            marker=dict(color='blue', opacity=0.6),
            name='Predicciones'
        ))
        
        # Añadir línea de referencia
        min_val = data['min_value']
        max_val = data['max_value']
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Línea de Referencia'
        ))
        
        # Actualizar layout
        fig.update_layout(
            title="Predicciones vs Precio Real",
            xaxis_title="Precio Real",
            yaxis_title="Precio Predicho",
            showlegend=True
        )
        
        # Mostrar gráfica
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.error("Error al obtener datos de la API")
        
except requests.exceptions.ConnectionError:
    st.error("No se puede conectar a la API. Asegúrate de que esté corriendo.")
except Exception as e:
    st.error(f"Error: {str(e)}")