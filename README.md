# Simulador de Apuestas con TrueSkill y Kelly

Este repositorio implementa un flujo completo para simular estrategias de apuesta deportiva utilizando:

- **TrueSkill Through Time (TTT)** para estimar la evolución de la habilidad de jugadoras
- **Criterio de Kelly** para definir el tamaño óptimo de las apuestas
- **Streamlit** para una visualización e interacción web amigable

Aplicado a partidos del circuito profesional femenino de tenis (WTA).

## Estimación de Habilidades – TrueSkill Through Time (TTT)

El script `TTT-Estimaciones.jl` genera predicciones de habilidad (`mu`, `sigma`) a partir de un CSV con partidos históricos.

### Entrada esperada:
- Columnas mínimas: `Date`, `Winner`, `Loser`, `B365W`, `B365L`
- Estas últimas dos pueden ser *odds* de cualquier casa de apuestas.

### Salida generada:
- `predicciones_wta.csv`, con columnas como:  
  `winner`, `loser`, `m_winner`, `s_winner`, `b_winner`, `m_loser`, `s_loser`, `b_loser`, `date`

> **Autoria del TrueSkill**:  
> Este módulo se basa en el repositorio oficial:  
> [TrueSkillThroughTime.jl](https://github.com/glandfried/TrueSkillThroughTime.jl)  
> Autor: Gustavo Landfried ([@glandfried](https://github.com/glandfried))

## Variantes del Simulador de Apuestas

Este proyecto aplica el criterio de **Kelly** sobre predicciones de habilidad generadas por TrueSkill Through Time (TTT). Se incluyen dos versiones del simulador:

### Simulador en Streamlit (`kelly-simulador/`)

- `kelly.py`: lógica de simulación (usada también en otras versiones)
- `kelly_streamlit.py`: interfaz visual para la web
- `requirements.txt`: dependencias necesarias

La versión en línea se encuentra desplegada en:  
https://kelly-causal-2025-v2.streamlit.app

➡️**Recomendado para uso general.** No requiere instalación local.

### Simulador en Tkinter (interfaz local de escritorio)

- `kelly.py`: misma lógica de simulación
- `kelly_gui.py`: interfaz gráfica construida con `tkinter`

➡️Alternativa liviana para experimentar localmente sin navegador.

### Datos de entrada

Ambos simuladores requieren un archivo `.csv` con predicciones generadas por `TTT-Estimaciones.jl`. Para replicar el análisis del trabajo final, se pueden utilizar directamente los archivos dentro de `inferencia/`.

