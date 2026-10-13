# 📈 Forecasting y Pronóstico de Ventas (Series Temporales)

Proyecto analítico desarrollado en **Python** para el modelado, pronóstico y evaluación de series temporales de demanda y ventas utilizando técnicas estadísticas avanzadas y visualizaciones interactivas en modo oscuro.

---

## 🚀 Características Principales
* **Simulación de Series de Tiempo:** Generación de un conjunto de datos históricos con componentes de tendencia ascendente, estacionalidad semanal y ruido aleatorio.
* **Modelo Predictivo:** Implementación de *Holt-Winters Exponential Smoothing* (a través de `statsmodels`) para capturar patrones estacionales y de tendencia de forma robusta.
* **Evaluación de Precisión:** Cálculo automático de métricas de error del modelo en el entorno de entrenamiento:
  * **MAE** (*Mean Absolute Error* / Error Absoluto Medio)
  * **RMSE** (*Root Mean Square Error* / Raíz del Error Cuadrático Medio)
* **Visualización Interactiva (Plotly):** Gráfica interactiva de alto impacto visual con diseño *Dark Mode*, comparación entre histórico y pronóstico a 30 días, y una tarjeta flotante de métricas incrustada.

---

## 🛠️ Tecnologías y Librerías Utilizadas
* **Python:** Lenguaje de programación principal.
* **Statsmodels:** Modelado estadístico y series temporales (`ExponentialSmoothing`).
* **Pandas & NumPy:** Manipulación de estructuras de datos temporales y cálculos numéricos.
* **Plotly:** Creación de gráficos interactivos avanzados.

---

## ⚙️ Instrucciones de Instalación y Ejecución

Sigue estos pasos para clonar y ejecutar este proyecto en tu entorno local:

1. **Clona este repositorio:**
   ```bash
   git clone [https://github.com/tu-usuario/forecasting-ventas-python.git](https://github.com/JossJLQH/forecasting-ventas-python.git)
   cd forecasting-ventas-python