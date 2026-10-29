# app.py
import numpy as np
import matplotlib.pyplot as plt
from modelo import obtener_datos, entrenar_y_predecir

# Configuración inicial
X, y, nombres_clases = obtener_datos()
kernels = ['linear', 'rbf']
titulos = ['Kernel Lineal (Línea recta)', 'Kernel RBF (Frontera curva)']

# Crear la figura para la interfaz gráfica
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Malla para las regiones de decisión (necesita un scaler temporal o el de uno de los modelos)
from sklearn.preprocessing import StandardScaler
scaler_temp = StandardScaler().fit(X)
X_scaled_temp = scaler_temp.transform(X)

xx, yy = np.meshgrid(
    np.linspace(X_scaled_temp[:, 0].min() - 0.8, X_scaled_temp[:, 0].max() + 0.8, 300),
    np.linspace(X_scaled_temp[:, 1].min() - 0.8, X_scaled_temp[:, 1].max() + 0.8, 300)
)
grid = np.c_[xx.ravel(), yy.ravel()]

# Renderizar cada gráfico en la interfaz
for ax, kernel, titulo in zip(axes, kernels, titulos):
    # Llamada al backend (modelo.py)
    resultado = entrenar_y_predecir(kernel=kernel, peso_nuevo=175, dulzor_nuevo=6.5)
    
    modelo = resultado["modelo"]
    X_scaled = resultado["X_scaled"]
    nueva_fruta_scaled = resultado["nueva_fruta_scaled"]
    pred_nombre = resultado["prediccion"]
    
    # Frontera de decisión
    Z = modelo.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.25, cmap=plt.cm.viridis)
    
    # Líneas de margen
    Z_dec = modelo.decision_function(grid).reshape(xx.shape)
    ax.contour(xx, yy, Z_dec, colors='k', levels=[-1, 0, 1], alpha=0.7, linestyles=['--', '-', '--'])
    
    # Datos de entrenamiento
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap=plt.cm.viridis, edgecolors='k', s=90, label='Frutas')
    
    # Vectores de soporte
    sv = modelo.support_vectors_
    ax.scatter(sv[:, 0], sv[:, 1], s=250, facecolors='none', edgecolors='gold', linewidths=2.5, label='Vectores de Soporte')
    
    # Nueva fruta a clasificar
    ax.scatter(nueva_fruta_scaled[:, 0], nueva_fruta_scaled[:, 1], c='green', marker='*', s=350, edgecolors='black', label=f'Nueva: {pred_nombre}')
    
    # Estilos de la interfaz
    ax.set_title(titulo, fontsize=12, fontweight='bold')
    ax.set_xlabel('Peso (Normalizado)')
    ax.set_ylabel('Dulzor (Normalizado)')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.6)

    print(f"--- Resultado con {titulo} ---")
    print(f"La nueva fruta fue clasificada como: **{pred_nombre.upper()}**\n")

plt.suptitle('Clasificador SVM (Manzanas vs Naranjas) - Arquitectura Modular', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()