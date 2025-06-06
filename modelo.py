import streamlit as st
import pandas as pd
import pickle
import os
import hashlib
from datetime import date, timedelta
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.session import Session
import plotly.express as px


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
        # Primero hacemos el pivot
        df_pivot = filter_by_status(df,).pivot(index='DT_REGISTRO', columns='ID_ESTATUS_RESERVACIONES', values='count').reset_index().fillna(0)
        # Creamos automáticamente el diccionario de aggregation
        agg_dict = {col: 'sum' for col in df_pivot.columns if col != 'DT_REGISTRO'}
        
        # Aplicamos el resample y aggregation
        return df_pivot.resample(freq, on='DT_REGISTRO').agg(agg_dict).reset_index()
    
    
    def resample_reservations(df, freq):
        """
        Resamplea un dataframe de reservaciones en formato long por la frecuencia indicada.
        
        Parámetros:
        - df: DataFrame en formato long con columnas ['DT_REGISTRO', 'ID_ESTATUS_RESERVACIONES', 'counts']
        - freq: frecuencia de resampleo ('W', 'ME', 'QE', etc.)
    
        Retorna:
        - DataFrame resampleado.
        """
        # Aseguramos que DT_REGISTRO sea datetime
        df['DT_REGISTRO'] = pd.to_datetime(df['DT_REGISTRO'])
        
        # Agrupamos por estatus primero
        resampled = (
            df
            .set_index('DT_REGISTRO')
            .groupby('ID_ESTATUS_RESERVACIONES')
            .resample(freq)['counts']
            .sum()
            .reset_index()
            )
        
        return resampled
        
    
    def create_chart(y, color, height, chart_type):
        if chart_type=='Bar':
            st.bar_chart(df_display_agg, x="DT_REGISTRO", y=y, color=color, height=height)
        if chart_type=='Area':
            st.area_chart(df_display_agg, x="DT_REGISTRO", y=y, color=color, height=height)
    
    def get_col_activas(df):
        # Definimos las columnas relevantes
        if 'ACTIVAS' not in df.columns:
            df['ACTIVAS'] = 0
        return df
        
    def resumen_ingresos(df):
    
        df = df.copy()
        
        df['mes'] = df['DT_REGISTRO'].dt.month_name()
        df['año'] = df['DT_REGISTRO'].dt.year.astype(str)
    
        resumen = df.pivot_table(values='H_TFA_TOTAL', index = ['año','mes'], columns = 'estatus_costs', aggfunc = 'sum')
    
        return resumen
    
    dictionary_ids = {
        # 0: "SIN DEFINIR",
        '1': "ACTIVAS",
        '2': "CANCELADA",
        '3': "NO SHOW",
        '4': "ACTIVAS",
        '5': "ACTIVAS",
        # 6: "REGISTRO EN TRANSITO",
        '7': "ACTIVAS",
        '8': "EN CASA (REGISTRO)",
        '9': "SALIDA"
    }
    
    
    
        
    
    # --------------------------------------------------------------------------------
    
    
    # Input widgets
    # Date range selection
    col = st.columns(4)
    with col[0]:
        pais = st.selectbox("Filtrar por país:", ("México","EUA","Canada"))
        dict_paises = {
            "México":"157",
            "EUA":"232",
            "Canada":"38"
        }
        pais_select = dict_paises[pais]
    with col[1]:
        start_date = st.date_input("Fecha inicial", df['DT_REGISTRO'].min().date(),df['DT_REGISTRO'].min().date(),df['DT_REGISTRO'].max().date())
    with col[2]:
        end_date = st.date_input("Fecha final", df['DT_REGISTRO'].max().date(),df['DT_REGISTRO'].min().date(),df['DT_REGISTRO'].max().date())
    # Time frame selection
    with col[3]:
        time_frame = st.selectbox("Selecciona un periodo de tiempo",
            ("Diario", "Semanal", "Mensual", "Quarterly")
        )
    # --------------------------------------------------------------------------------
    
    # Filter data based on date range
    mask = (df['ID_PAIS_ORIGEN'] == pais_select) & (df['DT_REGISTRO'].dt.date >= start_date) & (df['DT_REGISTRO'].dt.date <= end_date)
    df_filtered = df.loc[mask]
    
    # --------------> ID_ESTATUS_RESERVACIONES EN UNA COLUMNA
    df_display = df_filtered.fillna(0) 
    
    df_display['ID_ESTATUS_RESERVACIONES'] = df_display.ID_ESTATUS_RESERVACIONES.map(dictionary_ids)
    
    # //////////////// GET COLUMN COUNTS
    df_group_by_status = df_display.groupby(['DT_REGISTRO', 'ID_ESTATUS_RESERVACIONES']).agg(
        counts=('ID_ESTATUS_RESERVACIONES', 'count')
    ).reset_index()
    
    
    def calcular_ganancias_costos(df):
        """
        Calcula ingresos y costos por estatus.
        """
        # Creamos estatus_costs
        df['estatus_costs'] = df['ID_ESTATUS_RESERVACIONES'].apply(
            lambda x: 'ingreso_reservas' if x in ['ACTIVAS', 'NO SHOW', 'SALIDA', 'EN CASA (REGISTRO)'] else
                      ('costo_cancelacion' if x == 'CANCELADA' else 'otros')
        )
    
        df['H_TFA_TOTAL'] = pd.to_numeric(df['H_TFA_TOTAL'], errors='coerce').fillna(0.0)
    
        return df[['DT_REGISTRO', 'ID_ESTATUS_RESERVACIONES', 'estatus_costs','H_TFA_TOTAL']]
    
    df_costs = calcular_ganancias_costos(df_display)
    
    
    # # --------------------------------------------------------------------------------
    
    
    
    # # # --------------> TOMA ID ESTATUS COMO COLUMNAS <--------------
    if time_frame == 'Diario':
        df_display_agg = df_group_by_status.pivot(index='DT_REGISTRO', columns='ID_ESTATUS_RESERVACIONES', values='counts').reset_index().fillna(0)
        # df_chart = df_group_by_status
        df_chart = df_costs
        
    elif time_frame == 'Semanal':
        df_display_agg = aggregate_data(df_display, 'W').fillna(0)
        # df_chart = resample_reservations(df_group_by_status, 'W')
        df_chart = df_costs.set_index('DT_REGISTRO').groupby('estatus_costs').resample('W')['H_TFA_TOTAL'].sum().round(2).reset_index()
    
    elif time_frame == 'Mensual':
        df_display_agg = aggregate_data(df_display, 'ME').fillna(0)
        # df_chart = resample_reservations(df_group_by_status, 'ME')
        df_chart = df_costs.set_index('DT_REGISTRO').groupby('estatus_costs').resample('ME')['H_TFA_TOTAL'].sum().round(2).reset_index()
    
    
    elif time_frame == 'Quarterly':
        df_display_agg = aggregate_data(df_display, 'QE').fillna(0)
        # df_chart = resample_reservations(df_group_by_status, 'QE')
        df_chart = df_costs.set_index('DT_REGISTRO').groupby('estatus_costs').resample('QE')['H_TFA_TOTAL'].sum().round(2).reset_index()
    
    
    df_display_agg = get_col_activas(df_display_agg) 
    
        
    # df_display_agg['ACTIVAS'] = get_col_activas(df_display_agg)
    # # df_display_agg.rename(columns=dictionary_reservas, inplace = True)
    
    
    # # --------------------------------------------------------------------------------
    
    
    
    # Compute metric growth based on selected time frame (RESTA AL ÚLTIMO REGISTRO EL PENÚLTIMO - CRECIMIENTO DE HOY VS AYER)
    if len(df_display_agg) >= 2:
    
        reservs_growth = int(df_display_agg['ACTIVAS'].iloc[-1] - df_display_agg['ACTIVAS'].iloc[-2])
        cancel_growth = int(df_display_agg['CANCELADA'].iloc[-1] - df_display_agg['CANCELADA'].iloc[-2])
        noshow_growth = int(df_display_agg['NO SHOW'].iloc[-1] - df_display_agg['NO SHOW'].iloc[-2])
        checkout_growth = int(df_display_agg['SALIDA'].iloc[-1] - df_display_agg['SALIDA'].iloc[-2])
    
    
    else:
        reservs_growth = cancel_growth = 0
    
    
    # # Create metrics columns
    with st.container(border=True):
        
    
    
        col1, col2 = st.columns([1, 3])
        with col1:
            selection = st.selectbox("SELECCIONA",("Histórico", "Último registro"), label_visibility="collapsed")
        with col2:
            print("")
    
        def get_metric(col, selec):
            if selec == "Histórico":
                metric = format_with_commas(df_display_agg[col].sum())
            elif selec == "Último registro":
                metric = format_with_commas(df_display_agg[col].iloc[-1])
    
            return metric
    
            
        cols = st.columns(3)
        with cols[0]:
            with st.container(border = True):
                st.metric("Total Reservas Activas", 
                          # format_with_commas(df_display_agg['ACTIVAS'].sum()),
                          get_metric('ACTIVAS',selection),
                          format_with_commas(reservs_growth)
                         )
                st.bar_chart(df_display_agg, x="DT_REGISTRO", y='ACTIVAS', color="#29B5E8", height=200)
        with cols[1]:
            with st.container(border = True):
                st.metric("Total Cancelations", 
                          # format_with_commas(df_display_agg['CANCELADA'].sum()), 
                          get_metric('CANCELADA',selection),
                          format_with_commas(cancel_growth)
                         )
                st.bar_chart(df_display_agg, x="DT_REGISTRO", y='CANCELADA', color="#FF9F36", height=200)
        with cols[2]:
            with st.container(border = True):
                st.metric("Total No Show", 
                          # format_with_commas(df_display_agg['NO SHOW'].sum()), 
                          get_metric('NO SHOW',selection),
                          format_with_commas(noshow_growth)
                         )
                st.bar_chart(df_display_agg, x="DT_REGISTRO", y='NO SHOW', color="#724fbd", height=200)
    
    
        st.divider()
    
    
    
        c1, c2 = st.columns([2.5, 1])
    
        with c1:
    
            df_chart = df_chart.copy()
        
            df_chart['H_TFA_TOTAL'] = pd.to_numeric(
                df_chart['H_TFA_TOTAL'].astype(str).str.replace(',', ''), 
                errors='coerce'
            ).fillna(0.0)
        
        
            # Aseguramos que DT_REGISTRO esté en datetime
            df_chart['DT_REGISTRO'] = pd.to_datetime(df_chart['DT_REGISTRO'])
            
            # Agregamos por fecha y estatus_costs (puede haber varias reservas por fecha)
            df_grouped = (
                df_chart.groupby(['DT_REGISTRO', 'estatus_costs'])['H_TFA_TOTAL']
                .sum()
                .reset_index()
            )
            
            # Graficamos
            fig = px.line(
                df_grouped,
                x='DT_REGISTRO',
                y='H_TFA_TOTAL',
                color='estatus_costs',
                title='Histórico de Reservas: Ingresos vs Costos de Cancelación',
                template='plotly_white'
            )
            
            fig.update_layout(
                xaxis_title="Fecha",
                yaxis_title="Monto",
                legend_title="Tipo de Movimiento",
                hovermode="x unified",
                height=600
            )
            
            fig.update_xaxes(tickangle=-45)
            
            st.plotly_chart(fig, use_container_width=True)
    
        with c2:
            df_cost_resumen = resumen_ingresos(df_costs)
        
            st.dataframe(df_cost_resumen, height= 600)
