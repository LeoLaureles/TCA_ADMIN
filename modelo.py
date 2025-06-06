import streamlit as st
import pickle
import os
import pandas as pd
from snowflake.snowpark.context import get_active_session

# Obtener sesión activa de Snowflake
session = get_active_session()

# Función de limpieza
def limpiar_texto(texto):
    if not isinstance(texto, str):
        texto = str(texto)
    return " ".join(texto.strip().upper().split())

# Cargar modelo desde Stage
@st.cache_resource
def load_model():
    stage_path = r'@"RESERVACIONES"."PUBLIC"."MY_INTERNAL_STAGE"/final_model.pickle'
    local_folder = "tmp"
    local_file = os.path.join(local_folder, "final_model.pickle")

    os.makedirs(local_folder, exist_ok=True)
    session.file.get(stage_path, local_folder)

    with open(local_file, "rb") as f:
        return pickle.load(f)

# Leer tabla desde Snowflake
df_snowflake = session.table("RESERVACIONES.PUBLIC.ALMACENAMIENTO_DE_RESERVAS")
df = df_snowflake.to_pandas()

# Columnas que no deben usarse como entrada visual
columnas_ocultas = ['HIST_MENORES', 'HIST_TOTAL_HABITACIONES', 'HIST_ADULTOS', 'CANCELACION']

# df_visible queda sin usar para edición, pero editable_df lo dejamos oculto
editable_df = df.drop(columns=columnas_ocultas, errors='ignore')

st.title("TCA 'SISTEMA DE ALERTA/CANCELACIONES'")

# Solo mostrar el botón sin vista previa del DataFrame
if st.button("Predecir con modelo"):
    modelo = load_model()

    try:
        # Cargar pesos desde Snowflake
        pesos = {
            'peso_ciudad_agencia': session.table('RESERVACIONES.PUBLIC.PESO_CIUDAD_AGENCIA').to_pandas(),
            'peso_nombre_paquete': session.table('RESERVACIONES.PUBLIC.PESO_NOMBRE_PAQUETE').to_pandas(),
            'peso_nombre_canal': session.table('RESERVACIONES.PUBLIC.PESO_NOMBRE_CANAL').to_pandas(),
            'peso_nombre_estado': session.table('RESERVACIONES.PUBLIC.PESO_NOMBRE_ESTADO').to_pandas(),
            'peso_nombre_tipo_habitacion': session.table('RESERVACIONES.PUBLIC.PESO_NOMBRE_TIPO_HABITACION').to_pandas(),
            'peso_hist_menores': session.table('RESERVACIONES.PUBLIC.PESO_HIST_MENORES').to_pandas(),
            'peso_hist_total_habitaciones': session.table('RESERVACIONES.PUBLIC.PESO_HIST_TOTAL_HABITACIONES').to_pandas()
        }

        columnas_originales = [
            'CIUDAD_AGENCIA', 'NOMBRE_PAQUETE', 'NOMBRE_CANAL',
            'NOMBRE_ESTADO', 'NOMBRE_TIPO_HABITACION',
            'HIST_MENORES', 'HIST_TOTAL_HABITACIONES'
        ]

        columnas_modelo = [
            'ciudad_agencia', 'nombre_paquete', 'nombre_canal',
            'nombre_estado', 'nombre_tipo_habitacion',
            'hist_menores', 'hist_total_habitaciones'
        ]

        # Crear dataset de entrada uniendo columnas ocultas
        df_merge = pd.concat([editable_df, df[columnas_ocultas]], axis=1)
        df_transformado = df_merge[columnas_originales].copy()

        # Aplicar pesos por columna
        for col in columnas_originales:
            nombre_peso = f"peso_{col.lower()}"
            if nombre_peso in pesos:
                df_peso = pesos[nombre_peso]
                key_col = df_peso.columns[0]
                val_col = df_peso.columns[1]
                mapa = {limpiar_texto(k): v for k, v in zip(df_peso[key_col], df_peso[val_col])}
                df_transformado[col] = df_transformado[col].map(lambda x: mapa.get(limpiar_texto(x), None)).astype(float)

        # Renombrar y reordenar
        df_transformado.columns = columnas_modelo
        df_transformado = df_transformado[['nombre_paquete', 'ciudad_agencia', 'nombre_tipo_habitacion',
                                           'nombre_canal', 'nombre_estado', 'hist_menores', 'hist_total_habitaciones']]

        # Realizar predicción
        predicciones = modelo.predict(df_transformado)

        # Preparar DataFrame resultado (sin mostrar columna de predicción)
        resultado_final = editable_df.copy()
        resultado_final["Predicción"] = predicciones

        def resaltar_filas_rojas(row):
            if row.name in resultado_final.index and resultado_final.loc[row.name, 'Predicción'] == 1:
                return ['background-color: #ffcccc'] * len(row)
            else:
                return [''] * len(row)

        df_visible_sin_pred = resultado_final.drop(columns=["Predicción"])

        styled_df = df_visible_sin_pred.style \
            .apply(resaltar_filas_rojas, axis=1) \
            .set_table_styles([
                {'selector': 'thead th', 'props': [('background-color', '#2f2f2f'), ('color', 'white')]}
            ])

        st.success("¡Predicciones generadas!")
        st.dataframe(styled_df, use_container_width=True)
    except Exception as e:

        st.error(f"Ocurrió un error al predecir: {e}")
