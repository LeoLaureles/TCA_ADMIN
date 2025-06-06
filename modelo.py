import streamlit as st
import pandas as pd
import pickle
import os
import hashlib
from datetime import date, timedelta
from snowflake.snowpark.session import Session

# Usuarios autorizados con contraseñas hasheadas (SHA-256)
USUARIOS = {
    "admin": hashlib.sha256("1234".encode()).hexdigest(),
    "usuario": hashlib.sha256("abcd".encode()).hexdigest()
}

def verificar_login(usuario, contrasena):
    return usuario in USUARIOS and hashlib.sha256(contrasena.encode()).hexdigest() == USUARIOS[usuario]

# Mostrar login solo si no ha iniciado sesión
if "login" not in st.session_state:
    st.session_state["login"] = False

if not st.session_state["login"]:
    st.title("Inicio de sesión")
    usuario_input = st.text_input("Usuario", key="usuario_input")
    contrasena_input = st.text_input("Contraseña", type="password", key="contrasena_input")
    if st.button("Iniciar sesión"):
        if verificar_login(usuario_input, contrasena_input):
            st.session_state["login"] = True
            st.session_state["usuario"] = usuario_input
            st.success(f"Bienvenido, {usuario_input}")
            st.stop()
        else:
            st.error("Usuario o contraseña incorrectos")
    st.stop()

# Cerrar sesión
st.sidebar.button("Cerrar sesión", on_click=lambda: st.session_state.update({"login": False, "usuario": None}))

cnx = st.connection('snowflake')
session = cnx.session()

# Funciones de utilidad y modelo
@st.cache_resource
def load_model():
    stage_path = r'@"RESERVACIONES"."PUBLIC"."MY_INTERNAL_STAGE"/final_model.pickle'
    local_folder = "tmp"
    local_file = os.path.join(local_folder, "final_model.pickle")
    os.makedirs(local_folder, exist_ok=True)
    session.file.get(stage_path, local_folder)
    with open(local_file, "rb") as f:
        return pickle.load(f)

def limpiar_texto(texto):
    if not isinstance(texto, str):
        texto = str(texto)
    return " ".join(texto.strip().upper().split())

# Datos para predicción
df_snowflake = session.table("RESERVACIONES.PUBLIC.ALMACENAMIENTO_DE_RESERVAS")
df = df_snowflake.to_pandas()
columnas_ocultas = ['HIST_MENORES', 'HIST_TOTAL_HABITACIONES', 'HIST_ADULTOS', 'CANCELACION']
editable_df = df.drop(columns=columnas_ocultas, errors='ignore')

st.header("TCA SISTEMA DE ALERTAS HOTELES")

if st.button("Predecir con modelo"):
    modelo = load_model()
    try:
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

        df_merge = pd.concat([editable_df, df[columnas_ocultas]], axis=1)
        df_transformado = df_merge[columnas_originales].copy()

        for col in columnas_originales:
            nombre_peso = f"peso_{col.lower()}"
            if nombre_peso in pesos:
                df_peso = pesos[nombre_peso]
                key_col = df_peso.columns[0]
                val_col = df_peso.columns[1]
                mapa = {limpiar_texto(k): v for k, v in zip(df_peso[key_col], df_peso[val_col])}
                df_transformado[col] = df_transformado[col].map(lambda x: mapa.get(limpiar_texto(x), None)).astype(float)

        df_transformado.columns = columnas_modelo
        df_transformado = df_transformado[['nombre_paquete', 'ciudad_agencia', 'nombre_tipo_habitacion',
                                           'nombre_canal', 'nombre_estado', 'hist_menores', 'hist_total_habitaciones']]

        predicciones = modelo.predict(df_transformado)
        resultado_final = editable_df.copy()
        resultado_final["Prediccion"] = predicciones

        def resaltar_filas_rojas(row):
            if row.name in resultado_final.index and resultado_final.loc[row.name, 'Prediccion'] == 1:
                return ['background-color: #ffcccc'] * len(row)
            else:
                return [''] * len(row)

        df_visible_sin_pred = resultado_final.drop(columns=["Prediccion"])
        styled_df = df_visible_sin_pred.style \
            .apply(resaltar_filas_rojas, axis=1) \
            .set_table_styles([{'selector': 'thead th', 'props': [('background-color', '#2f2f2f'), ('color', 'white')]}])

        st.success("¡Predicciones generadas!")
        st.dataframe(styled_df, use_container_width=True)
    except Exception as e:
        st.error(f"Error al predecir: {e}")

# Continúa con el dashboard de visualización... (resto del código sigue aquí)
# <-- Aquí iría el resto del dashboard que ya tienes (reservas, fechas, gráficas, etc.)

    session = Session.builder.getOrCreate()
    df_reservas = session.table("RESERVACIONES.PUBLIC.RESERVACIONES_CLEANED").to_pandas()

    def convert_columns_to_datetime(df, columns, fmt=None):
        for col in columns:
            df[col] = pd.to_datetime(df[col], format=fmt, errors='coerce')
        return df

    df_reservas = convert_columns_to_datetime(df_reservas, ['DT_REGISTRO', 'DT_LLEGADA', 'DT_SALIDA'])

    def filter_by_status(df, id_estatus=None):
        grouped = df.groupby(['DT_REGISTRO', 'ID_ESTATUS_RESERVACIONES']).size().reset_index(name='count')
        return grouped if id_estatus is None else grouped[grouped.ID_ESTATUS_RESERVACIONES == id_estatus]

    def format_with_commas(number):
        return f"{number:,}"

    def aggregate_data(df, freq):
        pivoted = filter_by_status(df).pivot(index='DT_REGISTRO', columns='ID_ESTATUS_RESERVACIONES', values='count').reset_index()
        return pivoted.resample(freq, on='DT_REGISTRO').sum().reset_index()

    dictionary_reservas = {
        '1': "RESERVACION O (R)REGISTRO",
        '2': "CANCELADA",
        '3': "NO SHOW",
        '4': "RESERVACION EN TRANSICION",
        '5': "ROOMING LIST",
        '7': "PREREGISTRO",
        '8': "EN CASA (REGISTRO)",
        '9': "SALIDA"
    }

    col = st.columns(3)
    with col[0]:
        start_date = st.date_input("Fecha inicial", df_reservas['DT_REGISTRO'].min().date())
    with col[1]:
        end_date = st.date_input("Fecha final", df_reservas['DT_REGISTRO'].max().date())
    with col[2]:
        time_frame = st.selectbox("Periodo de tiempo", ("Diario", "Semanal", "Mensual", "Quarterly"))

    st.divider()
    mask = (df_reservas['DT_REGISTRO'].dt.date >= start_date) & (df_reservas['DT_REGISTRO'].dt.date <= end_date)
    df_filtered = df_reservas.loc[mask]

    if time_frame == 'Diario':
        df_display_agg = filter_by_status(df_filtered).pivot(index='DT_REGISTRO', columns='ID_ESTATUS_RESERVACIONES', values='count').reset_index()
    elif time_frame == 'Semanal':
        df_display_agg = aggregate_data(df_filtered, 'W')
    elif time_frame == 'Mensual':
        df_display_agg = aggregate_data(df_filtered, 'ME')
    elif time_frame == 'Quarterly':
        df_display_agg = aggregate_data(df_filtered, 'QE')

    df_display_agg = df_display_agg.fillna(0)
    df_display_agg['activas'] = df_display_agg[['1','4','5','7']].sum(axis=1)
    df_display_agg.rename(columns=dictionary_reservas, inplace=True)

    if len(df_display_agg) >= 2:
        reservs_growth = int(df_display_agg['activas'].iloc[-1] - df_display_agg['activas'].iloc[-2])
        cancel_growth = int(df_display_agg['CANCELADA'].iloc[-1] - df_display_agg['CANCELADA'].iloc[-2])
        noshow_growth = int(df_display_agg['NO SHOW'].iloc[-1] - df_display_agg['NO SHOW'].iloc[-2])
        checkout_growth = int(df_display_agg['SALIDA'].iloc[-1] - df_display_agg['SALIDA'].iloc[-2])
    else:
        reservs_growth = cancel_growth = noshow_growth = checkout_growth = 0

    cols = st.columns(3)
    with cols[0]:
        st.metric("Reservas Activas", format_with_commas(df_display_agg['activas'].sum()), format_with_commas(reservs_growth))
        st.bar_chart(df_display_agg, x="DT_REGISTRO", y='activas', color="#29B5E8", height=200)
    with cols[1]:
        st.metric("Cancelaciones", format_with_commas(df_display_agg['CANCELADA'].sum()), format_with_commas(cancel_growth))
        st.bar_chart(df_display_agg, x="DT_REGISTRO", y='CANCELADA', color="#FF9F36", height=200)
    with cols[2]:
        st.metric("No Show", format_with_commas(df_display_agg['NO SHOW'].sum()), format_with_commas(noshow_growth))
        st.bar_chart(df_display_agg, x="DT_REGISTRO", y='NO SHOW', color="#724fbd", height=200)

    with st.expander("Ver últimos 50 registros"):
        st.dataframe(df_filtered.sort_values(by='DT_REGISTRO', ascending=False).head(50))
