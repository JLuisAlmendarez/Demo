from flask import Flask, jsonify, request
import numpy as np
from tensorflow.keras.models import load_model
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

MODEL_PATH = 'src/price_prediction_20241205_023906.keras'
TEST_DATA_PATH = 'src/data.npy'
TEST_LABELS_PATH = 'src/data_y.npy'

try:
    # Cargar modelo y datos
    model = load_model(MODEL_PATH)
    X_test = np.load(TEST_DATA_PATH)
    y_test = np.load(TEST_LABELS_PATH)
    logger.info("Modelo y datos cargados exitosamente")
    
    # Hacer predicciones
    y_pred = model.predict(X_test)
    logger.info("Predicciones realizadas exitosamente")
    
    # Aplanar arrays
    y_test_flat = y_test.flatten()
    y_pred_flat = y_pred.flatten()
    
except Exception as e:
    logger.error(f"Error en inicialización: {str(e)}")
    raise

@app.route('/', methods=['GET'])
def health():
    return jsonify({
        'status': 'success',
        'model_loaded': True,
        'shapes': {
            'X_test': X_test.shape,
            'y_test': y_test.shape,
            'y_pred': y_pred.shape
        }
    })

@app.route('/data_for_plot', methods=['GET'])
def get_plot_data():
    """Endpoint para obtener los datos necesarios para la gráfica"""
    try:
        return jsonify({
            'success': True,
            'y_test': y_test_flat.tolist(),
            'y_pred': y_pred_flat.tolist(),
            'min_value': float(min(y_test_flat.min(), y_pred_flat.min())),
            'max_value': float(max(y_test_flat.max(), y_pred_flat.max()))
        })
        
    except Exception as e:
        logger.error(f"Error al enviar datos: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5555, debug=True)