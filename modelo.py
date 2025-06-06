import pandas as pd
import streamlit as st
from datetime import timedelta
import streamlit as st
import pickle
import os
import pandas as pd

cnx = st.connection('snowflake')
session = cnx.session()

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

st.header("TCA SISTEMA DE ALERTAS HOTELES")


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

# If you're already inside Snowflake Notebooks, there's usually a pre-defined session:
session = Session.builder.getOrCreate()

# Now simply query your table
table = session.table("RESERVACIONES.PUBLIC.RESERVACIONES_CLEANED")

# If you want to convert it to pandas (for Streamlit later)
df = table.to_pandas()




def convert_columns_to_datetime(df, columns, fmt=None):
    """
    Converts multiple columns to datetime dtype.
    
    Parameters:
        df (pd.DataFrame): The dataframe.
        columns (list): List of column names to convert.
        fmt (str, optional): The input format (ex: '%Y%m%d'). If None, pandas will infer.
        
    Returns:
        pd.DataFrame: The dataframe with updated datetime columns.
    """
    for col in columns:
        df[col] = pd.to_datetime(df[col], format=fmt, errors='coerce')
    return df


df = convert_columns_to_datetime(df, ['DT_REGISTRO', 'DT_LLEGADA', 'DT_SALIDA'])

def filter_by_status (df, id_estatus = None):
    if id_estatus is None:
        ds = df.groupby(['DT_REGISTRO', 'ID_ESTATUS_RESERVACIONES']).size().reset_index(name='count')
    else :
        df_grouped = df.groupby(['DT_REGISTRO', 'ID_ESTATUS_RESERVACIONES']).size().reset_index(name='count')
        ds = df_grouped[df_grouped.ID_ESTATUS_RESERVACIONES == id_estatus]
    return ds


# Helper functions
def format_with_commas(number):
    return f"{number:,}"

def aggregate_data(df, freq):
    return filter_by_status(df,).pivot(index='DT_REGISTRO', columns='ID_ESTATUS_RESERVACIONES', values='count').reset_index().resample(freq, on='DT_REGISTRO').agg({
        '1': 'sum',  # reserva ----
        '2': 'sum',  # cancelacion ---
        '3': 'sum',  # no show ---
        '4': 'sum',  # en transicion
        '5': 'sum',  # rooming list ----
        # '6': 'sum',
        '7': 'sum',  # pre registro
        '8': 'sum',  # en casa (registro)
        '9': 'sum'   # salida ---
    }).reset_index()

def create_chart(y, color, height, chart_type):
    if chart_type=='Bar':
        st.bar_chart(df_display_agg, x="DT_REGISTRO", y=y, color=color, height=height)
    if chart_type=='Area':
        st.area_chart(df_display_agg, x="DT_REGISTRO", y=y, color=color, height=height)


dictionary_reservas = {
    # 0: "SIN DEFINIR",
    '1': "RESERVACION O (R)REGISTRO",
    '2': "CANCELADA",
    '3': "NO SHOW",
    '4': "RESERVACION EN TRANSICION",
    '5': "ROOMING LIST",
    # 6: "REGISTRO EN TRANSITO",
    '7': "PREREGISTRO",
    '8': "EN CASA (REGISTRO)",
    '9': "SALIDA"
}

    

# --------------------------------------------------------------------------------


# Input widgets
# Date range selection
col = st.columns(3)
with col[0]:
    start_date = st.date_input("Fecha inicial", df['DT_REGISTRO'].min().date(),df['DT_REGISTRO'].min().date(),df['DT_REGISTRO'].max().date())
with col[1]:
    end_date = st.date_input("Fecha final", df['DT_REGISTRO'].max().date(),df['DT_REGISTRO'].min().date(),df['DT_REGISTRO'].max().date())
# Time frame selection
with col[2]:
    time_frame = st.selectbox("Selecciona un periodo e tiempo",
        ("Diario", "Semanal", "Mensual", "Quarterly")
    )
# --------------------------------------------------------------------------------

st.divider()

# Filter data based on date range
mask = (df['DT_REGISTRO'].dt.date >= start_date) & (df['DT_REGISTRO'].dt.date <= end_date)
df_filtered = df.loc[mask]

df_display = df_filtered
df_group_status = filter_by_status(df,).pivot(index='DT_REGISTRO', columns='ID_ESTATUS_RESERVACIONES', values='count')

df_by_status = df_filtered.groupby(['DT_REGISTRO', 'ID_ESTATUS_RESERVACIONES']).size().reset_index(name='count')
cancels_df = df_by_status[df_by_status.ID_ESTATUS_RESERVACIONES == '2']

# Aggregate data based on selected time frame
if time_frame == 'Diario':
    df_display_agg = filter_by_status(df,).pivot(index='DT_REGISTRO', columns='ID_ESTATUS_RESERVACIONES', values='count').reset_index()
    # df_display_agg = df_group_status
elif time_frame == 'Semanal':
    df_display_agg = aggregate_data(df_filtered, 'W')
elif time_frame == 'Mensual':
    df_display_agg = aggregate_data(df_filtered, 'ME')
elif time_frame == 'Quarterly':
    df_display_agg = aggregate_data(df_filtered, 'QE')

df_display_agg = df_display_agg.fillna(0)
df_display_agg['activas'] = df_display_agg[['1','4','5','7']].sum(axis=1)
df_display_agg.rename(columns=dictionary_reservas, inplace = True)

# --------------------------------------------------------------------------------

# Compute metric growth based on selected time frame
if len(df_display_agg) >= 2:

    reservs_growth = int(df_display_agg['activas'].iloc[-1] - df_display_agg['activas'].iloc[-2])
    # rooming_growth = int(df_display_agg['5'].iloc[-1] - df_display_agg['5'].iloc[-2])
    cancel_growth = int(df_display_agg['CANCELADA'].iloc[-1] - df_display_agg['CANCELADA'].iloc[-2])
    noshow_growth = int(df_display_agg['NO SHOW'].iloc[-1] - df_display_agg['NO SHOW'].iloc[-2])
    checkout_growth = int(df_display_agg['SALIDA'].iloc[-1] - df_display_agg['SALIDA'].iloc[-2])


else:
    reservs_growth = cancel_growth = 0


# Create metrics columns
cols = st.columns(3)
with cols[0]:
    st.metric("Reservas Activas", 
              format_with_commas(df_display_agg['activas'].sum()),
              format_with_commas(reservs_growth)
             )
    st.bar_chart(df_display_agg, x="DT_REGISTRO", y='activas', color="#29B5E8", height=200)
with cols[1]:
    st.metric("Cancelations", 
              format_with_commas(df_display_agg['CANCELADA'].sum()), 
              format_with_commas(cancel_growth)
             )
    st.bar_chart(df_display_agg, x="DT_REGISTRO", y='CANCELADA', color="#FF9F36", height=200)
with cols[2]:
    st.metric("No Show", 
              format_with_commas(df_display_agg['NO SHOW'].sum()), 
              format_with_commas(noshow_growth)
             )
    st.bar_chart(df_display_agg, x="DT_REGISTRO", y='NO SHOW', color="#724fbd", height=200)

# cols2 = st.columns(3)
# with cols2[0]:
#     st.metric("No Show", 
#               format_with_commas(df_display_agg['3'].sum()), 
#               format_with_commas(noshow_growth)
#              )
#     st.bar_chart(df_display_agg, x="DT_REGISTRO", y='3', color="#724fbd", height=200)
# with cols2[1]:
#     st.metric("Salida (check out)", 
#               format_with_commas(df_display_agg['3'].sum()), 
#               format_with_commas(checkout_growth)
#              )
#     st.bar_chart(df_display_agg, x="DT_REGISTRO", y='9', color="#4fbd84", height=200)



# RESERVACION O (R)REGISTRO
# RESERVACION CANCELADA
# NO SHOW
# RESERVACION EN TRANSICION
# ROOMING LIST
# REGISTRO EN TRANSITO
# PREREGISTRO
# EN CASA (REGISTRO)
# SALIDA



# Display filtered DataFrame
with st.expander("Ver últimas 50 registros"):
    st.dataframe(df_display.sort_values(by = 'DT_REGISTRO', ascending = False).head(50))

