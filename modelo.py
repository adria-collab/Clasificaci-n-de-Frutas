# modelo.py
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

def obtener_datos():
    """Retorna los datos de entrenamiento y los nombres de las clases."""
    X = np.array([
        [140, 8], [150, 7], [145, 9], [155, 8], [160, 7],   
        [210, 5], [220, 6], [200, 4], [230, 5], [215, 6]    
    ])
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  
    nombres_clases = ['Manzana', 'Naranja']
    return X, y, nombres_clases

def entrenar_y_predecir(kernel, peso_nuevo=175, dulzor_nuevo=6.5):
    """Entrena el modelo SVM con el kernel especificado y clasifica una nueva fruta."""
    X, y, nombres_clases = obtener_datos()
    
    # Normalización
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Entrenamiento del modelo
    modelo = SVC(kernel=kernel, C=1.0, gamma='scale')
    modelo.fit(X_scaled, y)
    
    # Clasificación de la nueva fruta
    nueva_fruta = np.array([[peso_nuevo, dulzor_nuevo]])
    nueva_fruta_scaled = scaler.transform(nueva_fruta)
    pred_nueva_idx = modelo.predict(nueva_fruta_scaled)[0]
    nombre_prediccion = nombres_clases[pred_nueva_idx]
    
    return {
        "modelo": modelo,
        "scaler": scaler,
        "X_scaled": X_scaled,
        "y": y,
        "nombres_clases": nombres_clases,
        "prediccion": nombre_prediccion,
        "nueva_fruta_scaled": nueva_fruta_scaled
    }