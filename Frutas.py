import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

def entrenar_y_predecir():
    # 1. Datos de entrenamiento
    X = np.array([
        [140, 8], [150, 7], [145, 9], [155, 8], [160, 7],
        [210, 5], [220, 6], [200, 4], [230, 5], [215, 6]
    ])
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    nombres_clases = ['Manzana', 'Naranja']

    # 2. Escalado
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Nueva fruta a clasificar
    nueva_fruta = np.array([[175, 6.5]])
    nueva_fruta_scaled = scaler.transform(nueva_fruta)

    # 4. Configuración de modelos (Kernels)
    kernels = ['linear', 'rbf']
    resultados = []

    for kernel in kernels:
        # Entrenamiento del modelo
        modelo = SVC(kernel=kernel, C=1.0, gamma='scale')
        modelo.fit(X_scaled, y)
        
        # Predicción de la nueva fruta
        pred_nueva = modelo.predict(nueva_fruta_scaled)[0]
        fruta_clasificada = nombres_clases[pred_nueva].upper()
        
        resultados.append({
            'kernel': kernel,
            'modelo': modelo,
            'prediccion': fruta_clasificada
        })

    return X_scaled, y, nueva_fruta_scaled, resultados