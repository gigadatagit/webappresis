import streamlit as st
import json
import re
import os
import io
import math
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
import contextily as cx
from matplotlib.ticker import FuncFormatter
from datetime import datetime
from docxtpl import DocxTemplate, InlineImage
from docx.shared import Cm
from staticmap import StaticMap, CircleMarker
from typing import List, Union

def obtener_min_max_medidas(dataframes: List[pd.DataFrame]) -> Union[List[float], None]:
    """
    Recibe una lista de entre 1 y 4 DataFrames que contienen columnas con nombres como:
    'Medida 1 (Ω·m)', 'Medida 2 (Ω·m)', etc.
    
    Retorna una lista con el valor mínimo y el valor máximo encontrados en todas las columnas
    de todos los DataFrames.
    
    Si la lista está vacía, retorna None.
    """
    if not dataframes:
        return None

    # Inicializamos listas para acumular todos los valores de interés
    todos_los_valores = []

    for df in dataframes:
        # Filtrar columnas que contienen "Medida" en su nombre
        columnas_medidas = [col for col in df.columns if "Medida" in col and "Ω·m" in col]
        
        # Extraer los valores numéricos de esas columnas y agregarlos a la lista
        for col in columnas_medidas:
            valores_columna = pd.to_numeric(df[col], errors='coerce')  # Ignora errores no numéricos
            todos_los_valores.extend(valores_columna.dropna().tolist())

    if not todos_los_valores:
        return None  # No se encontraron valores válidos

    # Obtener mínimo y máximo global
    valor_min = min(todos_los_valores)
    valor_max = max(todos_los_valores)

    return [valor_min, valor_max]

def get_map_png_bytes(lon, lat, buffer_m=300, width_px=900, height_px=700, zoom=17):
    """
    Genera un PNG (bytes) de un mapa satelital con marcador en (lon, lat).
    - buffer_m: radio en metros alrededor del punto (controla "zoom").
    - zoom: nivel de teselas (18-19 suele ser bueno).
    """
    # Crear punto y reproyectar a Web Mercator
    gdf = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326").to_crs(epsg=3857)
    pt = gdf.geometry.iloc[0]
    
    # Calcular bounding box
    bbox = (pt.x - buffer_m, pt.y - buffer_m, pt.x + buffer_m, pt.y + buffer_m)

    # Crear figura
    fig, ax = plt.subplots(figsize=(width_px/100, height_px/100), dpi=100)
    ax.set_xlim(bbox[0], bbox[2])
    ax.set_ylim(bbox[1], bbox[3])

    # Añadir basemap (Esri World Imagery)
    cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery, crs="EPSG:3857", zoom=zoom)

    # Dibujar marcador
    gdf.plot(ax=ax, markersize=40, color="red")

    ax.set_axis_off()
    plt.tight_layout(pad=0)

    # Guardar a buffer en memoria
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def obtener_template_path(tipo_tramo: str, cantidad_tramos: int) -> str:
    """
    Retorna el path del template a usar basado en el tipo de tramo y la cantidad.
    Ejemplo: 'Trifásicos' y 3 → 'templateVLF3FS3TR.docx'
             'Monofásicos' y 10 → 'templateVLF1FS10TR.docx'
    """
    fases = "3FS" if tipo_tramo == "Trifásicos" else "1FS"
    nombre_template = f"templateVLF{fases}{cantidad_tramos}TR.docx"
    return os.path.join('templates', nombre_template)

def convertir_a_mayusculas(data):
    if isinstance(data, str):
        return data.upper()
    elif isinstance(data, dict):
        return {k: convertir_a_mayusculas(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convertir_a_mayusculas(v) for v in data]
    elif isinstance(data, tuple):
        return tuple(convertir_a_mayusculas(v) for v in data)
    else:
        return data  # cualquier otro tipo se deja igual
    
    
def plot_resistividad(df: pd.DataFrame, titulo_proyecto="PROYECTO SOCODA"):
    """
    Espera un DataFrame con columnas:
      - 'Distancia (m)'
      - 'Perfil 1 [Ωm]' ... 'Perfil 4 [Ωm]' (1..4 presentes)
      - 'Promedio [Ωm]'
      - 'BOX-COX [Ωm]'
    """
    # -------- detectar perfiles disponibles (1 a 4) --------
    patron = re.compile(r"^Perfil\s+(\d)\s*\[Ωm\]$", re.IGNORECASE)
    cols_perfiles = []
    for col in df.columns:
        m = patron.match(col)
        if m:
            cols_perfiles.append(col)
    # ordenar por número de perfil
    cols_perfiles.sort(key=lambda c: int(patron.match(c).group(1)))

    if not cols_perfiles:
        raise ValueError("No se encontraron columnas de 'Perfil n [Ωm]' en el DataFrame.")

    # -------- preparar figura --------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = df["Distancia (m)"]

    # formato de eje Y con coma decimal y separador de miles
    def es_fmt(y, _):
        # cambia '.' por ',' y agrega separador de miles
        txt = f"{y:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return txt
    ax.yaxis.set_major_formatter(FuncFormatter(es_fmt))

    # -------- trazar perfiles (1..4) --------
    # No fijo colores explícitos: matplotlib elegirá paleta automáticamente.
    marcadores = ["o", "o", "o", "o"]  # mismo estilo redondeado
    for i, col in enumerate(cols_perfiles):
        ax.plot(x, df[col], marker=marcadores[i % len(marcadores)], linewidth=2.5, label=col)

    # -------- trazar promedio y BOX-COX --------
    if "Promedio [Ωm]" in df.columns:
        ax.plot(x, df["Promedio [Ωm]"], linewidth=2.0, linestyle="-", label="RUTA PROMEDIO [Ωm]")
    if "BOX-COX [Ωm]" in df.columns:
        ax.plot(x, df["BOX-COX [Ωm]"], linewidth=2.0, linestyle="-", label="BOX-COX  [Ωm]")

    # -------- estética --------
    ax.set_title(f"RESISTIVIDAD DE TERRENO - {titulo_proyecto}", fontsize=18, weight="bold", pad=12)
    ax.set_xlabel("Separación entre electrodos [m]", fontsize=12)
    ax.set_ylabel("Resistividad [Ωm]", fontsize=12)
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xlim(min(x), max(x))

    # opcional: acercar el eje Y al rango de los datos con un pequeño margen
    y_all = pd.concat([df[c] for c in cols_perfiles + [c for c in ["Promedio [Ωm]", "BOX-COX [Ωm]"] if c in df]], axis=0)
    y_min, y_max = float(y_all.min()), float(y_all.max())
    margen = 0.08 * (y_max - y_min if y_max > y_min else max(1.0, y_max))
    ax.set_ylim(max(0, y_min - margen), y_max + margen)

    # leyenda debajo, en varias columnas si hay muchas series
    n_series = len(cols_perfiles) + ("Promedio [Ωm]" in df.columns) + ("BOX-COX [Ωm]" in df.columns)
    ncols = 3 if n_series >= 5 else 2
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=ncols, frameon=False)

    plt.tight_layout()
    plt.show()


def plot_resistividad_to_buffer(df: pd.DataFrame, titulo_proyecto="PROYECTO SOCODA") -> io.BytesIO:
    patron = re.compile(r"^Perfil\s+(\d)\s*\[Ωm\]$", re.IGNORECASE)
    cols_perfiles = sorted([c for c in df.columns if patron.match(c)],
                           key=lambda c: int(patron.match(c).group(1)))
    if not cols_perfiles:
        raise ValueError("No hay columnas 'Perfil n [Ωm]' en el DataFrame.")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = df["Distancia (m)"]

    # Formato con separador latino
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda y, _: f"{y:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
    ))

    # Dibujar perfiles con anotaciones
    for col in cols_perfiles:
        ax.plot(x, df[col], marker="o", linewidth=2.0, label=col)
        for xi, yi in zip(x, df[col]):
            ax.annotate(f"{yi:.0f}", (xi, yi), textcoords="offset points",
                        xytext=(0, 5), ha="center", fontsize=8, color="black")

    # RUTA PROMEDIO con línea más gruesa
    if "Promedio [Ωm]" in df.columns:
        ax.plot(x, df["Promedio [Ωm]"], linewidth=3.5, label="RUTA PROMEDIO [Ωm]")

    # BOX-COX con línea más gruesa
    if "BOX-COX [Ωm]" in df.columns:
        ax.plot(x, df["BOX-COX [Ωm]"], linewidth=3.5, label="BOX-COX [Ωm]")

    # Configuración de ejes y título
    ax.set_title(f"RESISTIVIDAD DE TERRENO - {titulo_proyecto}", fontsize=18, loc='center')
    ax.set_xlabel("Separación entre electrodos [m]")
    ax.set_ylabel("Resistividad [Ωm]")
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xlim((min(x)-1), (max(x)+1))

    # Ajuste dinámico del eje Y
    series = cols_perfiles + [c for c in ["Promedio [Ωm]", "BOX-COX [Ωm]"] if c in df]
    y_all = pd.concat([df[c] for c in series])
    y_min, y_max = float(y_all.min()), float(y_all.max())
    m = 0.08 * (y_max - y_min if y_max > y_min else max(1.0, y_max))
    ax.set_ylim(0, y_max + m)

    # Leyenda
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
              ncol=3 if len(series) >= 5 else 2, frameon=False)
    plt.tight_layout()

    # Guardar en buffer
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf



# Configuración de plantilla de Word (una sola instancia)
#template_path = os.path.join('templates', 'templateVLF3FS3TR.docx')

# Diccionario de labels para preguntas de verificación
preguntas_verificacion = {
    'frmVerfCabPreg1': 'El rótulo del cable en su chaqueta es legible y congruente con lo instalado en sitio',
    'frmVerfCabPreg2': 'Limpieza de cada una de las terminales',
    'frmVerfCabPreg3': 'Marcación correcta de los cables en ambos extremos',
    'frmVerfCabPreg4': 'Verificación de continuidad del cable de acuerdo a las marcaciones',
    'frmVerfCabPreg5': 'Verificación del tendido y conexionado del cable XLPE',
    'frmVerfCabPreg6': 'Distancias de seguridad entre cables apropiadas para hacer la prueba VLF'
}

#if 'doc' not in st.session_state:
#    st.session_state.doc = DocxTemplate(template_path)

# Inicialización de estado
if 'step' not in st.session_state:
    st.session_state.step = 1
    st.session_state.data = {}

st.title("Formulario Resistividad - Word Automatizado")

# Funciones de navegación
def next_step():
    missing = [k for k, v in st.session_state.data.items() if v is None or v == ""]
    if missing:
        st.error("Por favor completa todos los campos antes de continuar.")
    else:
        st.session_state.step += 1
        st.rerun()

def prev_step():
    if st.session_state.step > 1:
        st.session_state.step -= 1
        st.rerun()

# Paso 1: Información General
if st.session_state.step == 1:
    st.header("Paso 1: Información General")
    st.session_state.data['nombreProyecto'] = st.text_input("Nombre del Proyecto", key='nombreProyecto')
    st.session_state.data['nombreCiudadoMunicipio'] = st.text_input("Ciudad o Municipio", key='ciudad')
    st.session_state.data['nombreDepartamento'] = st.text_input("Departamento", key='departamento')
    st.session_state.data['tipoCoordenada'] = st.selectbox(f"Tipo de Imagen para las Coordenadas", ["Urbano", "Rural"], key=f'tipo_coordenada')
    st.session_state.data['nombreCompleto'] = st.text_input("Nombre Completo", key='nombre')
    st.session_state.data['nroConteoTarjeta'] = st.text_input("Número de CONTE o Tarjeta Profesional", key='conte_tarjeta')
    st.session_state.data['nombreCargo'] = st.text_input("Nombre del Cargo", key='cargo')
    st.session_state.data['fechaCreacionSinFormato'] = st.date_input("Fecha de Creación", key='fecha_creacion', value=datetime.now())
    st.session_state.data['fechaCreacion'] = st.session_state.data['fechaCreacionSinFormato'].strftime("%Y-%m-%d")
    st.session_state.data['direccionProyecto'] = st.text_input("Dirección", key='direccion')
    st.session_state.data['cantidadPerfiles'] = st.selectbox("Cantidad de Perfiles", [1, 2, 3, 4], key='cantidad_perfiles')
    st.session_state.data['latitud'] = st.number_input("Latitud", key='latitud', format="%.6f")
    st.session_state.data['longitud'] = st.number_input("Longitud", key='longitud', format="%.6f")
    #Agregar el campo de selección de Rural o Urbano para la generación de la imagen 

    cols = st.columns([1,1])
    if cols[1].button("Siguiente"):
        next_step()

# Paso 2: Datos Técnicos
elif st.session_state.step == 2:
    st.header("Paso 2: Datos Técnicos de los Perfiles")
    #st.session_state.data['tensionPrueba'] = st.selectbox("Tensión de Prueba", ["Aceptación", "Mantenimiento"], key='tension')
    #st.session_state.data['valTensionPrueba'] = 21 if st.session_state.data['tensionPrueba'] == "Aceptación" else 16
    #tipo = st.selectbox("Tipo de Tramos", ["Trifásicos", "Monofásicos"], key='tipo_tramos')
    #st.session_state.data['tipoTramos'] = tipo
    #max_tramos = 10 if tipo == "Trifásicos" else 20
    
    cantidad_perfiles = int(st.session_state.data['cantidadPerfiles'])
    
    #MEDIDAS = [1, 2, 3]
    #DISTANCIAS = [1, 2, 4, 6, 8, 10]

    #for p in range(1, cantidad_perfiles + 1):
    #    for m in MEDIDAS:
    #        for d in DISTANCIAS:
    #            nombre_llave = f"valPerf{p}Med{m}Dis{d}"
    #            widget_key  = f"val_P{p}_M{m}_D{d}"
    #            valor_inicial = float(st.session_state.data.get(nombre_llave, 0.0))
    #            st.session_state.data[nombre_llave] = st.number_input(
    #                f"P{p} · Med {m} · Dist {d} [Ω·m]",
    #                key=widget_key,
    #                min_value=0.0,
    #                value=valor_inicial,
    #                format="%.2f"
    #            )
                
                
    MEDIDAS = [1, 2, 3]
    DISTANCIAS = [1, 2, 4, 6, 8, 10]

    for p in range(1, cantidad_perfiles + 1):
        # Campos numéricos dinámicos
        for m in MEDIDAS:
            
            st.markdown("---")  # Línea divisoria entre medidas
            
            for d in DISTANCIAS:
                nombre_llave = f"valPerf{p}Med{m}Dis{d}"
                widget_key  = f"val_P{p}_M{m}_D{d}"
                valor_inicial = float(st.session_state.data.get(nombre_llave, 0.0))
                st.session_state.data[nombre_llave] = st.number_input(
                    f"Perfil #{p} · Medida #{m} · Distancia {d}m [Ωm]",
                    key=widget_key,
                    min_value=0.0,
                    value=valor_inicial,
                    format="%.2f"
                )



        # Campo de comentarios por perfil (luego de los number_input)
        comentarios_key = f"comentariosPerf{p}"
        st.session_state.data[comentarios_key] = st.text_area(
            f"Comentarios para el Perfil {p}",
            key=f"comentarios_{p}",
            value=st.session_state.data.get(comentarios_key, "")
        )
        
        st.markdown("---") # Línea divisoria entre perfiles

    cols = st.columns([1,1,1])
    if cols[0].button("Anterior"):
        prev_step()
    if cols[1].button("Siguiente"):
        #next_step()
        # Crear ruta del template dinámicamente
        #tipo_tramos = st.session_state.data.get('tipoTramos')
        #cantidad_tramos = st.session_state.data.get('cantidadTramos')
        #template_path = obtener_template_path(tipo_tramos, cantidad_tramos)
        
        nroPerfiles = int(st.session_state.data['cantidadPerfiles'])
    
        if nroPerfiles == 1:
        
            template_path = 'templates/templateRES1PR.docx'
            
        elif nroPerfiles == 2:
            
            template_path = 'templates/templateRES2PR.docx'
            
        elif nroPerfiles == 3:
            
            template_path = 'templates/templateRES3PR.docx'
            
        elif nroPerfiles == 4:
            
            template_path = 'templates/templateRES4PR_V2.docx'
        
        # Cargar la plantilla en el estado de sesión
        try:
            st.session_state.doc = DocxTemplate(template_path)
            next_step()
        except FileNotFoundError:
            st.error(f"No se encontró la plantilla: {template_path}")

# Paso 3: Formulario de Verificación
elif st.session_state.step == 3:
    st.header("Paso 3: Subida de Imágenes de Pruebas de Perfiles")
    
    datos_Sin_Mayuscula = st.session_state.data.copy()
    datos = convertir_a_mayusculas(datos_Sin_Mayuscula)
    
    nroPerfiles = int(st.session_state.data['cantidadPerfiles'])
    
    if nroPerfiles == 1:
    
        df_Valores_Perfil1 = pd.DataFrame({
            'Distancia (m)': [1, 2, 4, 6, 8, 10],
            'Medida 1 (Ω·m)': [st.session_state.data.get(f'valPerf1Med1Dis{d}', 0.0) for d in [1, 2, 4, 6, 8, 10]],
            'Medida 2 (Ω·m)': [st.session_state.data.get(f'valPerf1Med2Dis{d}', 0.0) for d in [1, 2, 4, 6, 8, 10]],
            'Medida 3 (Ω·m)': [st.session_state.data.get(f'valPerf1Med3Dis{d}', 0.0) for d in [1, 2, 4, 6, 8, 10]],
        })
        
        df_Valores_Promedio_Perfil1 = pd.DataFrame({
            'Distancia (m)': [1, 2, 4, 6, 8, 10],
            'Valor Promedio (Ω·m)': [
                round(
                    (
                        st.session_state.data.get(f'valPerf1Med1Dis{d}', 0.0) +
                        st.session_state.data.get(f'valPerf1Med2Dis{d}', 0.0) +
                        st.session_state.data.get(f'valPerf1Med3Dis{d}', 0.0)
                    ) / 3, 2
                ) for d in [1, 2, 4, 6, 8, 10]
            ]
        })
        
        st.markdown("### Valores Ingresados y Promedios - Perfil #1")
        
        df_Medidas_Con_Promedio_Perfil1 = df_Valores_Perfil1.copy()
        df_Medidas_Con_Promedio_Perfil1['Promedio [Ωm]'] = df_Valores_Promedio_Perfil1['Valor Promedio (Ω·m)']
        
        st.dataframe(df_Medidas_Con_Promedio_Perfil1)
        
        # Obtener valores min y max de todos los perfiles
        valor_Minimo_Perfiles = obtener_min_max_medidas([df_Valores_Perfil1])[0]
        valor_Maximo_Perfiles = obtener_min_max_medidas([df_Valores_Perfil1])[1]
        
        st.session_state.data['valMinPerfiles'] = valor_Minimo_Perfiles
        st.session_state.data['valMaxPerfiles'] = valor_Maximo_Perfiles
        
        
        st.session_state.data['valPerf1Dis1Prom'] = df_Valores_Promedio_Perfil1.loc[df_Valores_Promedio_Perfil1['Distancia (m)'] == 1, 'Valor Promedio (Ω·m)'].values[0]
        st.session_state.data['valPerf1Dis2Prom'] = df_Valores_Promedio_Perfil1.loc[df_Valores_Promedio_Perfil1['Distancia (m)'] == 2, 'Valor Promedio (Ω·m)'].values[0]
        st.session_state.data['valPerf1Dis4Prom'] = df_Valores_Promedio_Perfil1.loc[df_Valores_Promedio_Perfil1['Distancia (m)'] == 4, 'Valor Promedio (Ω·m)'].values[0]
        st.session_state.data['valPerf1Dis6Prom'] = df_Valores_Promedio_Perfil1.loc[df_Valores_Promedio_Perfil1['Distancia (m)'] == 6, 'Valor Promedio (Ω·m)'].values[0]
        st.session_state.data['valPerf1Dis8Prom'] = df_Valores_Promedio_Perfil1.loc[df_Valores_Promedio_Perfil1['Distancia (m)'] == 8, 'Valor Promedio (Ω·m)'].values[0]
        st.session_state.data['valPerf1Dis10Prom'] = df_Valores_Promedio_Perfil1.loc[df_Valores_Promedio_Perfil1['Distancia (m)'] == 10, 'Valor Promedio (Ω·m)'].values[0]
        
    elif nroPerfiles == 2:
        
        df_Valores_Perfil1 = pd.DataFrame({
            'Distancia (m)': [1, 2, 4, 6, 8, 10],
            'Medida 1 (Ω·m)': [st.session_state.data.get(f'valPerf1Med1Dis{d}', 0.0) for d in [1, 2, 4, 6, 8, 10]],
            'Medida 2 (Ω·m)': [st.session_state.data.get(f'valPerf1Med2Dis{d}', 0.0) for d in [1, 2, 4, 6, 8, 10]],
            'Medida 3 (Ω·m)': [st.session_state.data.get(f'valPerf1Med3Dis{d}', 0.0) for d in [1, 2, 4, 6, 8, 10]],
        })
        
        df_Valores_Promedio_Perfil1 = pd.DataFrame({
            'Distancia (m)': [1, 2, 4, 6, 8, 10],
            'Valor Promedio (Ω·m)': [
                round(
                    (
                        st.session_state.data.get(f'valPerf1Med1Dis{d}', 0.0) +
                        st.session_state.data.get(f'valPerf1Med2Dis{d}', 0.0) +
                        st.session_state.data.get(f'valPerf1Med3Dis{d}', 0.0)
                    ) / 3, 2
                ) for d in [1, 2, 4, 6, 8, 10]
            ]
        })
        
        st.session_state.data['valPerf1Dis1Prom'] = df_Valores_Promedio_Perfil1.loc[df_Valores_Promedio_Perfil1['Distancia (m)'] == 1, 'Valor Promedio (Ω·m)'].values[0]
        st.session_state.data['valPerf1Dis2Prom'] = df_Valores_Promedio_Perfil1.loc[df_Valores_Promedio_Perfil1['Distancia (m)'] == 2, 'Valor Promedio (Ω·m)'].values[0]
        st.session_state.data['valPerf1Dis4Prom'] = df_Valores_Promedio_Perfil1.loc[df_Valores_Promedio_Perfil1['Distancia (m)'] == 4, 'Valor Promedio (Ω·m)'].values[0]
        st.session_state.data['valPerf1Dis6Prom'] = df_Valores_Promedio_Perfil1.loc[df_Valores_Promedio_Perfil1['Distancia (m)'] == 6, 'Valor Promedio (Ω·m)'].values[0]
        st.session_state.data['valPerf1Dis8Prom'] = df_Valores_Promedio_Perfil1.loc[df_Valores_Promedio_Perfil1['Distancia (m)'] == 8, 'Valor Promedio (Ω·m)'].values[0]
        st.session_state.data['valPerf1Dis10Prom'] = df_Valores_Promedio_Perfil1.loc[df_Valores_Promedio_Perfil1['Distancia (m)'] == 10, 'Valor Promedio (Ω·m)'].values[0]
        
        df_Valores_Perfil2 = pd.DataFrame({
            'Distancia (m)': [1, 2, 4, 6, 8, 10],
            'Medida 1 (Ω·m)': [st.session_state.data.get(f'valPerf2Med1Dis{d}', 0.0) for d in [1, 2, 4, 6, 8, 10]],
            'Medida 2 (Ω·m)': [st.session_state.data.get(f'valPerf2Med2Dis{d}', 0.0) for d in [1, 2, 4, 6, 8, 10]],
            'Medida 3 (Ω·m)': [st.session_state.data.get(f'valPerf2Med3Dis{d}', 0.0) for d in [1, 2, 4, 6, 8, 10]],
        })
        
        df_Valores_Promedio_Perfil2 = pd.DataFrame({
            'Distancia (m)': [1, 2, 4, 6, 8, 10],
            'Valor Promedio (Ω·m)': [
                round(
                    (
                        st.session_state.data.get(f'valPerf2Med1Dis{d}', 0.0) +
                        st.session_state.data.get(f'valPerf2Med2Dis{d}', 0.0) +
                        st.session_state.data.get(f'valPerf2Med3Dis{d}', 0.0)
                    ) / 3, 2
                ) for d in [1, 2, 4, 6, 8, 10]
            ]
        })
        
        st.markdown("### Valores Ingresados y Promedios - Perfil #1")
        
        df_Medidas_Con_Promedio_Perfil1 = df_Valores_Perfil1.copy()
        df_Medidas_Con_Promedio_Perfil1['Promedio [Ωm]'] = df_Valores_Promedio_Perfil1['Valor Promedio (Ω·m)']
        
        st.dataframe(df_Medidas_Con_Promedio_Perfil1)
        
        st.markdown("### Valores Ingresados y Promedios - Perfil #2")
        
        df_Medidas_Con_Promedio_Perfil2 = df_Valores_Perfil2.copy()
        df_Medidas_Con_Promedio_Perfil2['Promedio [Ωm]'] = df_Valores_Promedio_Perfil2['Valor Promedio (Ω·m)']
        
        st.dataframe(df_Medidas_Con_Promedio_Perfil2)
        
        # Obtener valores min y max de todos los perfiles
        valor_Minimo_Perfiles = obtener_min_max_medidas([df_Valores_Perfil1, df_Valores_Perfil2])[0]
        valor_Maximo_Perfiles = obtener_min_max_medidas([df_Valores_Perfil1, df_Valores_Perfil2])[1]
        
        st.session_state.data['valMinPerfiles'] = valor_Minimo_Perfiles
        st.session_state.data['valMaxPerfiles'] = valor_Maximo_Perfiles
        
        st.session_state.data['valPerf2Dis1Prom'] = df_Valores_Promedio_Perfil2.loc[df_Valores_Promedio_Perfil2['Distancia (m)'] == 1, 'Valor Promedio (Ω·m)'].values[0]
        st.session_state.data['valPerf2Dis2Prom'] = df_Valores_Promedio_Perfil2.loc[df_Valores_Promedio_Perfil2['Distancia (m)'] == 2, 'Valor Promedio (Ω·m)'].values[0]
        st.session_state.data['valPerf2Dis4Prom'] = df_Valores_Promedio_Perfil2.loc[df_Valores_Promedio_Perfil2['Distancia (m)'] == 4, 'Valor Promedio (Ω·m)'].values[0]
        st.session_state.data['valPerf2Dis6Prom'] = df_Valores_Promedio_Perfil2.loc[df_Valores_Promedio_Perfil2['Distancia (m)'] == 6, 'Valor Promedio (Ω·m)'].values[0]
        st.session_state.data['valPerf2Dis8Prom'] = df_Valores_Promedio_Perfil2.loc[df_Valores_Promedio_Perfil2['Distancia (m)'] == 8, 'Valor Promedio (Ω·m)'].values[0]
        st.session_state.data['valPerf2Dis10Prom'] = df_Valores_Promedio_Perfil2.loc[df_Valores_Promedio_Perfil2['Distancia (m)'] == 10, 'Valor Promedio (Ω·m)'].values[0]
        
        st.markdown('### Listado de Valores de Promedios por cada Perfil y el Promedio General')
        
        df_Listado_Valores_Promedios_y_Promedio = pd.DataFrame({
            'Distancia [m]': [1, 2, 4, 6, 8, 10],
            'Valor Promedio Perfil 1 [Ωm]': df_Valores_Promedio_Perfil1['Valor Promedio (Ω·m)'],
            'Valor Promedio Perfil 2 [Ωm]': df_Valores_Promedio_Perfil2['Valor Promedio (Ω·m)'],
            'Valor Promedio General [Ωm]': [
                round(
                    (
                        df_Valores_Promedio_Perfil1.loc[df_Valores_Promedio_Perfil1['Distancia (m)'] == d, 'Valor Promedio (Ω·m)'].values[0] +
                        df_Valores_Promedio_Perfil2.loc[df_Valores_Promedio_Perfil2['Distancia (m)'] == d, 'Valor Promedio (Ω·m)'].values[0]
                    ) / 4, 4
                ) for d in [1, 2, 4, 6, 8, 10]
            ]
        })
        
        st.session_state.data['valPromGenDis1'] = df_Listado_Valores_Promedios_y_Promedio.loc[df_Listado_Valores_Promedios_y_Promedio['Distancia [m]'] == 1, 'Valor Promedio General [Ωm]'].values[0]
        st.session_state.data['valPromGenDis2'] = df_Listado_Valores_Promedios_y_Promedio.loc[df_Listado_Valores_Promedios_y_Promedio['Distancia [m]'] == 2, 'Valor Promedio General [Ωm]'].values[0]
        st.session_state.data['valPromGenDis4'] = df_Listado_Valores_Promedios_y_Promedio.loc[df_Listado_Valores_Promedios_y_Promedio['Distancia [m]'] == 4, 'Valor Promedio General [Ωm]'].values[0]
        st.session_state.data['valPromGenDis6'] = df_Listado_Valores_Promedios_y_Promedio.loc[df_Listado_Valores_Promedios_y_Promedio['Distancia [m]'] == 6, 'Valor Promedio General [Ωm]'].values[0]
        st.session_state.data['valPromGenDis8'] = df_Listado_Valores_Promedios_y_Promedio.loc[df_Listado_Valores_Promedios_y_Promedio['Distancia [m]'] == 8, 'Valor Promedio General [Ωm]'].values[0]
        st.session_state.data['valPromGenDis10'] = df_Listado_Valores_Promedios_y_Promedio.loc[df_Listado_Valores_Promedios_y_Promedio['Distancia [m]'] == 10, 'Valor Promedio General [Ωm]'].values[0]
        
    elif nroPerfiles == 3:
    
        df_Valores_Perfil1 = pd.DataFrame({
            'Distancia (m)': [1, 2, 4, 6, 8, 10],
            'Medida 1 (Ω·m)': [st.session_state.data.get(f'valPerf1Med1Dis{d}', 0.0) for d in [1, 2, 4, 6, 8, 10]],
            'Medida 2 (Ω·m)': [st.session_state.data.get(f'valPerf1Med2Dis{d}', 0.0) for d in [1, 2, 4, 6, 8, 10]],
            'Medida 3 (Ω·m)': [st.session_state.data.get(f'valPerf1Med3Dis{d}', 0.0) for d in [1, 2, 4, 6, 8, 10]],
        })
        
        df_Valores_Promedio_Perfil1 = pd.DataFrame({
            'Distancia (m)': [1, 2, 4, 6, 8, 10],
            'Valor Promedio (Ω·m)': [
                round(
                    (
                        st.session_state.data.get(f'valPerf1Med1Dis{d}', 0.0) +
                        st.session_state.data.get(f'valPerf1Med2Dis{d}', 0.0) +
                        st.session_state.data.get(f'valPerf1Med3Dis{d}', 0.0)
                    ) / 3, 2
                ) for d in [1, 2, 4, 6, 8, 10]
            ]
        })
        
        st.session_state.data['valPerf1Dis1Prom'] = df_Valores_Promedio_Perfil1.loc[df_Valores_Promedio_Perfil1['Distancia (m)'] == 1, 'Valor Promedio (Ω·m)'].values[0]
        st.session_state.data['valPerf1Dis2Prom'] = df_Valores_Promedio_Perfil1.loc[df_Valores_Promedio_Perfil1['Distancia (m)'] == 2, 'Valor Promedio (Ω·m)'].values[0]
        st.session_state.data['valPerf1Dis4Prom'] = df_Valores_Promedio_Perfil1.loc[df_Valores_Promedio_Perfil1['Distancia (m)'] == 4, 'Valor Promedio (Ω·m)'].values[0]
        st.session_state.data['valPerf1Dis6Prom'] = df_Valores_Promedio_Perfil1.loc[df_Valores_Promedio_Perfil1['Distancia (m)'] == 6, 'Valor Promedio (Ω·m)'].values[0]
        st.session_state.data['valPerf1Dis8Prom'] = df_Valores_Promedio_Perfil1.loc[df_Valores_Promedio_Perfil1['Distancia (m)'] == 8, 'Valor Promedio (Ω·m)'].values[0]
        st.session_state.data['valPerf1Dis10Prom'] = df_Valores_Promedio_Perfil1.loc[df_Valores_Promedio_Perfil1['Distancia (m)'] == 10, 'Valor Promedio (Ω·m)'].values[0]
        
        df_Valores_Perfil2 = pd.DataFrame({
            'Distancia (m)': [1, 2, 4, 6, 8, 10],
            'Medida 1 (Ω·m)': [st.session_state.data.get(f'valPerf2Med1Dis{d}', 0.0) for d in [1, 2, 4, 6, 8, 10]],
            'Medida 2 (Ω·m)': [st.session_state.data.get(f'valPerf2Med2Dis{d}', 0.0) for d in [1, 2, 4, 6, 8, 10]],
            'Medida 3 (Ω·m)': [st.session_state.data.get(f'valPerf2Med3Dis{d}', 0.0) for d in [1, 2, 4, 6, 8, 10]],
        })
        
        df_Valores_Promedio_Perfil2 = pd.DataFrame({
            'Distancia (m)': [1, 2, 4, 6, 8, 10],
            'Valor Promedio (Ω·m)': [
                round(
                    (
                        st.session_state.data.get(f'valPerf2Med1Dis{d}', 0.0) +
                        st.session_state.data.get(f'valPerf2Med2Dis{d}', 0.0) +
                        st.session_state.data.get(f'valPerf2Med3Dis{d}', 0.0)
                    ) / 3, 2
                ) for d in [1, 2, 4, 6, 8, 10]
            ]
        })
        
        st.session_state.data['valPerf2Dis1Prom'] = df_Valores_Promedio_Perfil2.loc[df_Valores_Promedio_Perfil2['Distancia (m)'] == 1, 'Valor Promedio (Ω·m)'].values[0]
        st.session_state.data['valPerf2Dis2Prom'] = df_Valores_Promedio_Perfil2.loc[df_Valores_Promedio_Perfil2['Distancia (m)'] == 2, 'Valor Promedio (Ω·m)'].values[0]
        st.session_state.data['valPerf2Dis4Prom'] = df_Valores_Promedio_Perfil2.loc[df_Valores_Promedio_Perfil2['Distancia (m)'] == 4, 'Valor Promedio (Ω·m)'].values[0]
        st.session_state.data['valPerf2Dis6Prom'] = df_Valores_Promedio_Perfil2.loc[df_Valores_Promedio_Perfil2['Distancia (m)'] == 6, 'Valor Promedio (Ω·m)'].values[0]
        st.session_state.data['valPerf2Dis8Prom'] = df_Valores_Promedio_Perfil2.loc[df_Valores_Promedio_Perfil2['Distancia (m)'] == 8, 'Valor Promedio (Ω·m)'].values[0]
        st.session_state.data['valPerf2Dis10Prom'] = df_Valores_Promedio_Perfil2.loc[df_Valores_Promedio_Perfil2['Distancia (m)'] == 10, 'Valor Promedio (Ω·m)'].values[0]
        
        df_Valores_Perfil3 = pd.DataFrame({
            'Distancia (m)': [1, 2, 4, 6, 8, 10],
            'Medida 1 (Ω·m)': [st.session_state.data.get(f'valPerf3Med1Dis{d}', 0.0) for d in [1, 2, 4, 6, 8, 10]],
            'Medida 2 (Ω·m)': [st.session_state.data.get(f'valPerf3Med2Dis{d}', 0.0) for d in [1, 2, 4, 6, 8, 10]],
            'Medida 3 (Ω·m)': [st.session_state.data.get(f'valPerf3Med3Dis{d}', 0.0) for d in [1, 2, 4, 6, 8, 10]],
        })
        
        df_Valores_Promedio_Perfil3 = pd.DataFrame({
            'Distancia (m)': [1, 2, 4, 6, 8, 10],
            'Valor Promedio (Ω·m)': [
                round(
                    (
                        st.session_state.data.get(f'valPerf3Med1Dis{d}', 0.0) +
                        st.session_state.data.get(f'valPerf3Med2Dis{d}', 0.0) +
                        st.session_state.data.get(f'valPerf3Med3Dis{d}', 0.0)
                    ) / 3, 2
                ) for d in [1, 2, 4, 6, 8, 10]
            ]
        })
        
        st.markdown("### Valores Ingresados y Promedios - Perfil #1")
        
        df_Medidas_Con_Promedio_Perfil1 = df_Valores_Perfil1.copy()
        df_Medidas_Con_Promedio_Perfil1['Promedio [Ωm]'] = df_Valores_Promedio_Perfil1['Valor Promedio (Ω·m)']
        
        st.dataframe(df_Medidas_Con_Promedio_Perfil1)
        
        st.markdown("### Valores Ingresados y Promedios - Perfil #2")
        
        df_Medidas_Con_Promedio_Perfil2 = df_Valores_Perfil2.copy()
        df_Medidas_Con_Promedio_Perfil2['Promedio [Ωm]'] = df_Valores_Promedio_Perfil2['Valor Promedio (Ω·m)']
        
        st.dataframe(df_Medidas_Con_Promedio_Perfil2)
        
        st.markdown("### Valores Ingresados y Promedios - Perfil #3")
        
        df_Medidas_Con_Promedio_Perfil3 = df_Valores_Perfil3.copy()
        df_Medidas_Con_Promedio_Perfil3['Promedio [Ωm]'] = df_Valores_Promedio_Perfil3['Valor Promedio (Ω·m)']
        
        st.dataframe(df_Medidas_Con_Promedio_Perfil3)
        
        # Obtener valores min y max de todos los perfiles
        valor_Minimo_Perfiles = obtener_min_max_medidas([df_Valores_Perfil1, df_Valores_Perfil2, df_Valores_Perfil3])[0]
        valor_Maximo_Perfiles = obtener_min_max_medidas([df_Valores_Perfil1, df_Valores_Perfil2, df_Valores_Perfil3])[1]
        
        st.session_state.data['valMinPerfiles'] = valor_Minimo_Perfiles
        st.session_state.data['valMaxPerfiles'] = valor_Maximo_Perfiles
        
        st.session_state.data['valPerf3Dis1Prom'] = df_Valores_Promedio_Perfil3.loc[df_Valores_Promedio_Perfil3['Distancia (m)'] == 1, 'Valor Promedio (Ω·m)'].values[0]
        st.session_state.data['valPerf3Dis2Prom'] = df_Valores_Promedio_Perfil3.loc[df_Valores_Promedio_Perfil3['Distancia (m)'] == 2, 'Valor Promedio (Ω·m)'].values[0]
        st.session_state.data['valPerf3Dis4Prom'] = df_Valores_Promedio_Perfil3.loc[df_Valores_Promedio_Perfil3['Distancia (m)'] == 4, 'Valor Promedio (Ω·m)'].values[0]
        st.session_state.data['valPerf3Dis6Prom'] = df_Valores_Promedio_Perfil3.loc[df_Valores_Promedio_Perfil3['Distancia (m)'] == 6, 'Valor Promedio (Ω·m)'].values[0]
        st.session_state.data['valPerf3Dis8Prom'] = df_Valores_Promedio_Perfil3.loc[df_Valores_Promedio_Perfil3['Distancia (m)'] == 8, 'Valor Promedio (Ω·m)'].values[0]
        st.session_state.data['valPerf3Dis10Prom'] = df_Valores_Promedio_Perfil3.loc[df_Valores_Promedio_Perfil3['Distancia (m)'] == 10, 'Valor Promedio (Ω·m)'].values[0]
        
        st.markdown('### Listado de Valores de Promedios por cada Perfil y el Promedio General')
        
        df_Listado_Valores_Promedios_y_Promedio = pd.DataFrame({
            'Distancia [m]': [1, 2, 4, 6, 8, 10],
            'Valor Promedio Perfil 1 [Ωm]': df_Valores_Promedio_Perfil1['Valor Promedio (Ω·m)'],
            'Valor Promedio Perfil 2 [Ωm]': df_Valores_Promedio_Perfil2['Valor Promedio (Ω·m)'],
            'Valor Promedio Perfil 3 [Ωm]': df_Valores_Promedio_Perfil3['Valor Promedio (Ω·m)'],
            'Valor Promedio General [Ωm]': [
                round(
                    (
                        df_Valores_Promedio_Perfil1.loc[df_Valores_Promedio_Perfil1['Distancia (m)'] == d, 'Valor Promedio (Ω·m)'].values[0] +
                        df_Valores_Promedio_Perfil2.loc[df_Valores_Promedio_Perfil2['Distancia (m)'] == d, 'Valor Promedio (Ω·m)'].values[0] +
                        df_Valores_Promedio_Perfil3.loc[df_Valores_Promedio_Perfil3['Distancia (m)'] == d, 'Valor Promedio (Ω·m)'].values[0]
                    ) / 4, 4
                ) for d in [1, 2, 4, 6, 8, 10]
            ]
        })
        
        st.session_state.data['valPromGenDis1'] = df_Listado_Valores_Promedios_y_Promedio.loc[df_Listado_Valores_Promedios_y_Promedio['Distancia [m]'] == 1, 'Valor Promedio General [Ωm]'].values[0]
        st.session_state.data['valPromGenDis2'] = df_Listado_Valores_Promedios_y_Promedio.loc[df_Listado_Valores_Promedios_y_Promedio['Distancia [m]'] == 2, 'Valor Promedio General [Ωm]'].values[0]
        st.session_state.data['valPromGenDis4'] = df_Listado_Valores_Promedios_y_Promedio.loc[df_Listado_Valores_Promedios_y_Promedio['Distancia [m]'] == 4, 'Valor Promedio General [Ωm]'].values[0]
        st.session_state.data['valPromGenDis6'] = df_Listado_Valores_Promedios_y_Promedio.loc[df_Listado_Valores_Promedios_y_Promedio['Distancia [m]'] == 6, 'Valor Promedio General [Ωm]'].values[0]
        st.session_state.data['valPromGenDis8'] = df_Listado_Valores_Promedios_y_Promedio.loc[df_Listado_Valores_Promedios_y_Promedio['Distancia [m]'] == 8, 'Valor Promedio General [Ωm]'].values[0]
        st.session_state.data['valPromGenDis10'] = df_Listado_Valores_Promedios_y_Promedio.loc[df_Listado_Valores_Promedios_y_Promedio['Distancia [m]'] == 10, 'Valor Promedio General [Ωm]'].values[0]
        
        st.dataframe(df_Listado_Valores_Promedios_y_Promedio)
        
    elif nroPerfiles == 4:
        
        df_Valores_Perfil1 = pd.DataFrame({
            'Distancia (m)': [1, 2, 4, 6, 8, 10],
            'Medida 1 (Ω·m)': [st.session_state.data.get(f'valPerf1Med1Dis{d}', 0.0) for d in [1, 2, 4, 6, 8, 10]],
            'Medida 2 (Ω·m)': [st.session_state.data.get(f'valPerf1Med2Dis{d}', 0.0) for d in [1, 2, 4, 6, 8, 10]],
            'Medida 3 (Ω·m)': [st.session_state.data.get(f'valPerf1Med3Dis{d}', 0.0) for d in [1, 2, 4, 6, 8, 10]],
        })
        
        df_Valores_Promedio_Perfil1 = pd.DataFrame({
            'Distancia (m)': [1, 2, 4, 6, 8, 10],
            'Valor Promedio (Ω·m)': [
                round(
                    (
                        st.session_state.data.get(f'valPerf1Med1Dis{d}', 0.0) +
                        st.session_state.data.get(f'valPerf1Med2Dis{d}', 0.0) +
                        st.session_state.data.get(f'valPerf1Med3Dis{d}', 0.0)
                    ) / 3, 4
                ) for d in [1, 2, 4, 6, 8, 10]
            ]
        })
        
        st.session_state.data['valPerf1Dis1Prom'] = df_Valores_Promedio_Perfil1.loc[df_Valores_Promedio_Perfil1['Distancia (m)'] == 1, 'Valor Promedio (Ω·m)'].values[0]
        st.session_state.data['valPerf1Dis2Prom'] = df_Valores_Promedio_Perfil1.loc[df_Valores_Promedio_Perfil1['Distancia (m)'] == 2, 'Valor Promedio (Ω·m)'].values[0]
        st.session_state.data['valPerf1Dis4Prom'] = df_Valores_Promedio_Perfil1.loc[df_Valores_Promedio_Perfil1['Distancia (m)'] == 4, 'Valor Promedio (Ω·m)'].values[0]
        st.session_state.data['valPerf1Dis6Prom'] = df_Valores_Promedio_Perfil1.loc[df_Valores_Promedio_Perfil1['Distancia (m)'] == 6, 'Valor Promedio (Ω·m)'].values[0]
        st.session_state.data['valPerf1Dis8Prom'] = df_Valores_Promedio_Perfil1.loc[df_Valores_Promedio_Perfil1['Distancia (m)'] == 8, 'Valor Promedio (Ω·m)'].values[0]
        st.session_state.data['valPerf1Dis10Prom'] = df_Valores_Promedio_Perfil1.loc[df_Valores_Promedio_Perfil1['Distancia (m)'] == 10, 'Valor Promedio (Ω·m)'].values[0]
        
        df_Valores_Perfil2 = pd.DataFrame({
            'Distancia (m)': [1, 2, 4, 6, 8, 10],
            'Medida 1 (Ω·m)': [st.session_state.data.get(f'valPerf2Med1Dis{d}', 0.0) for d in [1, 2, 4, 6, 8, 10]],
            'Medida 2 (Ω·m)': [st.session_state.data.get(f'valPerf2Med2Dis{d}', 0.0) for d in [1, 2, 4, 6, 8, 10]],
            'Medida 3 (Ω·m)': [st.session_state.data.get(f'valPerf2Med3Dis{d}', 0.0) for d in [1, 2, 4, 6, 8, 10]],
        })
        
        df_Valores_Promedio_Perfil2 = pd.DataFrame({
            'Distancia (m)': [1, 2, 4, 6, 8, 10],
            'Valor Promedio (Ω·m)': [
                round(
                    (
                        st.session_state.data.get(f'valPerf2Med1Dis{d}', 0.0) +
                        st.session_state.data.get(f'valPerf2Med2Dis{d}', 0.0) +
                        st.session_state.data.get(f'valPerf2Med3Dis{d}', 0.0)
                    ) / 3, 4
                ) for d in [1, 2, 4, 6, 8, 10]
            ]
        })
        
        st.session_state.data['valPerf2Dis1Prom'] = df_Valores_Promedio_Perfil2.loc[df_Valores_Promedio_Perfil2['Distancia (m)'] == 1, 'Valor Promedio (Ω·m)'].values[0]
        st.session_state.data['valPerf2Dis2Prom'] = df_Valores_Promedio_Perfil2.loc[df_Valores_Promedio_Perfil2['Distancia (m)'] == 2, 'Valor Promedio (Ω·m)'].values[0]
        st.session_state.data['valPerf2Dis4Prom'] = df_Valores_Promedio_Perfil2.loc[df_Valores_Promedio_Perfil2['Distancia (m)'] == 4, 'Valor Promedio (Ω·m)'].values[0]
        st.session_state.data['valPerf2Dis6Prom'] = df_Valores_Promedio_Perfil2.loc[df_Valores_Promedio_Perfil2['Distancia (m)'] == 6, 'Valor Promedio (Ω·m)'].values[0]
        st.session_state.data['valPerf2Dis8Prom'] = df_Valores_Promedio_Perfil2.loc[df_Valores_Promedio_Perfil2['Distancia (m)'] == 8, 'Valor Promedio (Ω·m)'].values[0]
        st.session_state.data['valPerf2Dis10Prom'] = df_Valores_Promedio_Perfil2.loc[df_Valores_Promedio_Perfil2['Distancia (m)'] == 10, 'Valor Promedio (Ω·m)'].values[0]
        
        df_Valores_Perfil3 = pd.DataFrame({
            'Distancia (m)': [1, 2, 4, 6, 8, 10],
            'Medida 1 (Ω·m)': [st.session_state.data.get(f'valPerf3Med1Dis{d}', 0.0) for d in [1, 2, 4, 6, 8, 10]],
            'Medida 2 (Ω·m)': [st.session_state.data.get(f'valPerf3Med2Dis{d}', 0.0) for d in [1, 2, 4, 6, 8, 10]],
            'Medida 3 (Ω·m)': [st.session_state.data.get(f'valPerf3Med3Dis{d}', 0.0) for d in [1, 2, 4, 6, 8, 10]],
        })
        
        df_Valores_Promedio_Perfil3 = pd.DataFrame({
            'Distancia (m)': [1, 2, 4, 6, 8, 10],
            'Valor Promedio (Ω·m)': [
                round(
                    (
                        st.session_state.data.get(f'valPerf3Med1Dis{d}', 0.0) +
                        st.session_state.data.get(f'valPerf3Med2Dis{d}', 0.0) +
                        st.session_state.data.get(f'valPerf3Med3Dis{d}', 0.0)
                    ) / 3, 4
                ) for d in [1, 2, 4, 6, 8, 10]
            ]
        })
        
        st.session_state.data['valPerf3Dis1Prom'] = df_Valores_Promedio_Perfil3.loc[df_Valores_Promedio_Perfil3['Distancia (m)'] == 1, 'Valor Promedio (Ω·m)'].values[0]
        st.session_state.data['valPerf3Dis2Prom'] = df_Valores_Promedio_Perfil3.loc[df_Valores_Promedio_Perfil3['Distancia (m)'] == 2, 'Valor Promedio (Ω·m)'].values[0]
        st.session_state.data['valPerf3Dis4Prom'] = df_Valores_Promedio_Perfil3.loc[df_Valores_Promedio_Perfil3['Distancia (m)'] == 4, 'Valor Promedio (Ω·m)'].values[0]
        st.session_state.data['valPerf3Dis6Prom'] = df_Valores_Promedio_Perfil3.loc[df_Valores_Promedio_Perfil3['Distancia (m)'] == 6, 'Valor Promedio (Ω·m)'].values[0]
        st.session_state.data['valPerf3Dis8Prom'] = df_Valores_Promedio_Perfil3.loc[df_Valores_Promedio_Perfil3['Distancia (m)'] == 8, 'Valor Promedio (Ω·m)'].values[0]
        st.session_state.data['valPerf3Dis10Prom'] = df_Valores_Promedio_Perfil3.loc[df_Valores_Promedio_Perfil3['Distancia (m)'] == 10, 'Valor Promedio (Ω·m)'].values[0]
        
        df_Valores_Perfil4 = pd.DataFrame({
            'Distancia (m)': [1, 2, 4, 6, 8, 10],
            'Medida 1 (Ω·m)': [st.session_state.data.get(f'valPerf4Med1Dis{d}', 0.0) for d in [1, 2, 4, 6, 8, 10]],
            'Medida 2 (Ω·m)': [st.session_state.data.get(f'valPerf4Med2Dis{d}', 0.0) for d in [1, 2, 4, 6, 8, 10]],
            'Medida 3 (Ω·m)': [st.session_state.data.get(f'valPerf4Med3Dis{d}', 0.0) for d in [1, 2, 4, 6, 8, 10]],
        })
        
        df_Valores_Promedio_Perfil4 = pd.DataFrame({
            'Distancia (m)': [1, 2, 4, 6, 8, 10],
            'Valor Promedio (Ω·m)': [
                round(
                    (
                        st.session_state.data.get(f'valPerf4Med1Dis{d}', 0.0) +
                        st.session_state.data.get(f'valPerf4Med2Dis{d}', 0.0) +
                        st.session_state.data.get(f'valPerf4Med3Dis{d}', 0.0)
                    ) / 3, 4
                ) for d in [1, 2, 4, 6, 8, 10]
            ]
        })
        
        st.markdown("### Valores Ingresados y Promedios - Perfil #1")
        
        df_Medidas_Con_Promedio_Perfil1 = df_Valores_Perfil1.copy()
        df_Medidas_Con_Promedio_Perfil1['Promedio [Ωm]'] = df_Valores_Promedio_Perfil1['Valor Promedio (Ω·m)']
        
        st.dataframe(df_Medidas_Con_Promedio_Perfil1)
        
        st.markdown("### Valores Ingresados y Promedios - Perfil #2")
        
        df_Medidas_Con_Promedio_Perfil2 = df_Valores_Perfil2.copy()
        df_Medidas_Con_Promedio_Perfil2['Promedio [Ωm]'] = df_Valores_Promedio_Perfil2['Valor Promedio (Ω·m)']
        
        st.dataframe(df_Medidas_Con_Promedio_Perfil2)
        
        st.markdown("### Valores Ingresados y Promedios - Perfil #3")
        
        df_Medidas_Con_Promedio_Perfil3 = df_Valores_Perfil3.copy()
        df_Medidas_Con_Promedio_Perfil3['Promedio [Ωm]'] = df_Valores_Promedio_Perfil3['Valor Promedio (Ω·m)']
        
        st.dataframe(df_Medidas_Con_Promedio_Perfil3)
        
        st.markdown("### Valores Ingresados y Promedios - Perfil #4")
        
        df_Medidas_Con_Promedio_Perfil4 = df_Valores_Perfil4.copy()
        df_Medidas_Con_Promedio_Perfil4['Promedio [Ωm]'] = df_Valores_Promedio_Perfil4['Valor Promedio (Ω·m)']
        
        st.dataframe(df_Medidas_Con_Promedio_Perfil4)
        
        # Obtener valores min y max de todos los perfiles
        valor_Minimo_Perfiles = obtener_min_max_medidas([df_Valores_Perfil1, df_Valores_Perfil2, df_Valores_Perfil3, df_Valores_Perfil4])[0]
        valor_Maximo_Perfiles = obtener_min_max_medidas([df_Valores_Perfil1, df_Valores_Perfil2, df_Valores_Perfil3, df_Valores_Perfil4])[1]
        
        st.session_state.data['valMinPerfiles'] = valor_Minimo_Perfiles
        st.session_state.data['valMaxPerfiles'] = valor_Maximo_Perfiles
        
        st.session_state.data['valPerf4Dis1Prom'] = df_Valores_Promedio_Perfil4.loc[df_Valores_Promedio_Perfil4['Distancia (m)'] == 1, 'Valor Promedio (Ω·m)'].values[0]
        st.session_state.data['valPerf4Dis2Prom'] = df_Valores_Promedio_Perfil4.loc[df_Valores_Promedio_Perfil4['Distancia (m)'] == 2, 'Valor Promedio (Ω·m)'].values[0]
        st.session_state.data['valPerf4Dis4Prom'] = df_Valores_Promedio_Perfil4.loc[df_Valores_Promedio_Perfil4['Distancia (m)'] == 4, 'Valor Promedio (Ω·m)'].values[0]
        st.session_state.data['valPerf4Dis6Prom'] = df_Valores_Promedio_Perfil4.loc[df_Valores_Promedio_Perfil4['Distancia (m)'] == 6, 'Valor Promedio (Ω·m)'].values[0]
        st.session_state.data['valPerf4Dis8Prom'] = df_Valores_Promedio_Perfil4.loc[df_Valores_Promedio_Perfil4['Distancia (m)'] == 8, 'Valor Promedio (Ω·m)'].values[0]
        st.session_state.data['valPerf4Dis10Prom'] = df_Valores_Promedio_Perfil4.loc[df_Valores_Promedio_Perfil4['Distancia (m)'] == 10, 'Valor Promedio (Ω·m)'].values[0]
        
        st.markdown('### Listado de Valores de Promedios por cada Perfil y el Promedio General')
        
        df_Listado_Valores_Promedios_y_Promedio = pd.DataFrame({
            'Distancia [m]': [1, 2, 4, 6, 8, 10],
            'Valor Promedio Perfil 1 [Ωm]': df_Valores_Promedio_Perfil1['Valor Promedio (Ω·m)'],
            'Valor Promedio Perfil 2 [Ωm]': df_Valores_Promedio_Perfil2['Valor Promedio (Ω·m)'],
            'Valor Promedio Perfil 3 [Ωm]': df_Valores_Promedio_Perfil3['Valor Promedio (Ω·m)'],
            'Valor Promedio Perfil 4 [Ωm]': df_Valores_Promedio_Perfil4['Valor Promedio (Ω·m)'],
            'Valor Promedio General [Ωm]': [
                round(
                    (
                        df_Valores_Promedio_Perfil1.loc[df_Valores_Promedio_Perfil1['Distancia (m)'] == d, 'Valor Promedio (Ω·m)'].values[0] +
                        df_Valores_Promedio_Perfil2.loc[df_Valores_Promedio_Perfil2['Distancia (m)'] == d, 'Valor Promedio (Ω·m)'].values[0] +
                        df_Valores_Promedio_Perfil3.loc[df_Valores_Promedio_Perfil3['Distancia (m)'] == d, 'Valor Promedio (Ω·m)'].values[0] +
                        df_Valores_Promedio_Perfil4.loc[df_Valores_Promedio_Perfil4['Distancia (m)'] == d, 'Valor Promedio (Ω·m)'].values[0]
                    ) / 4, 4
                ) for d in [1, 2, 4, 6, 8, 10]
            ]
        })
        
        st.session_state.data['valPromGenDis1'] = df_Listado_Valores_Promedios_y_Promedio.loc[df_Listado_Valores_Promedios_y_Promedio['Distancia [m]'] == 1, 'Valor Promedio General [Ωm]'].values[0]
        st.session_state.data['valPromGenDis2'] = df_Listado_Valores_Promedios_y_Promedio.loc[df_Listado_Valores_Promedios_y_Promedio['Distancia [m]'] == 2, 'Valor Promedio General [Ωm]'].values[0]
        st.session_state.data['valPromGenDis4'] = df_Listado_Valores_Promedios_y_Promedio.loc[df_Listado_Valores_Promedios_y_Promedio['Distancia [m]'] == 4, 'Valor Promedio General [Ωm]'].values[0]
        st.session_state.data['valPromGenDis6'] = df_Listado_Valores_Promedios_y_Promedio.loc[df_Listado_Valores_Promedios_y_Promedio['Distancia [m]'] == 6, 'Valor Promedio General [Ωm]'].values[0]
        st.session_state.data['valPromGenDis8'] = df_Listado_Valores_Promedios_y_Promedio.loc[df_Listado_Valores_Promedios_y_Promedio['Distancia [m]'] == 8, 'Valor Promedio General [Ωm]'].values[0]
        st.session_state.data['valPromGenDis10'] = df_Listado_Valores_Promedios_y_Promedio.loc[df_Listado_Valores_Promedios_y_Promedio['Distancia [m]'] == 10, 'Valor Promedio General [Ωm]'].values[0]
        
        st.dataframe(df_Listado_Valores_Promedios_y_Promedio)
        
    if nroPerfiles == 1:
        
        listado_Valores_Promedio_Perfil1 = df_Valores_Promedio_Perfil1['Valor Promedio (Ω·m)'].tolist()
        
        listado_Valores_LogaritmoNatural = [round(math.log(v), 4) if v > 0 else 0 for v in listado_Valores_Promedio_Perfil1]
        
        st.session_state.data['lnPerf1Dis1Prom'] = listado_Valores_LogaritmoNatural[0]
        st.session_state.data['lnPerf1Dis2Prom'] = listado_Valores_LogaritmoNatural[1]
        st.session_state.data['lnPerf1Dis4Prom'] = listado_Valores_LogaritmoNatural[2]
        st.session_state.data['lnPerf1Dis6Prom'] = listado_Valores_LogaritmoNatural[3]
        st.session_state.data['lnPerf1Dis8Prom'] = listado_Valores_LogaritmoNatural[4]
        st.session_state.data['lnPerf1Dis10Prom'] = listado_Valores_LogaritmoNatural[5]
        
        st.markdown('### Listado de Valores de Logaritmo Natural')
        
        st.dataframe(listado_Valores_LogaritmoNatural)
        
    elif nroPerfiles == 2:
        
        listado_Valores_Promedio_Perfil1 = df_Valores_Promedio_Perfil1['Valor Promedio (Ω·m)'].tolist()
        listado_Valores_Promedio_Perfil2 = df_Valores_Promedio_Perfil2['Valor Promedio (Ω·m)'].tolist()
        
        listado_Valores_A_Realizar_Logaritmo = listado_Valores_Promedio_Perfil1 + listado_Valores_Promedio_Perfil2
        
        listado_Valores_LogaritmoNatural = [round(math.log(v), 4) if v > 0 else 0 for v in listado_Valores_A_Realizar_Logaritmo]
        
        st.session_state.data['lnPerf1Dis1Prom'] = listado_Valores_LogaritmoNatural[0]
        st.session_state.data['lnPerf1Dis2Prom'] = listado_Valores_LogaritmoNatural[1]
        st.session_state.data['lnPerf1Dis4Prom'] = listado_Valores_LogaritmoNatural[2]
        st.session_state.data['lnPerf1Dis6Prom'] = listado_Valores_LogaritmoNatural[3]
        st.session_state.data['lnPerf1Dis8Prom'] = listado_Valores_LogaritmoNatural[4]
        st.session_state.data['lnPerf1Dis10Prom'] = listado_Valores_LogaritmoNatural[5]
        
        st.session_state.data['lnPerf2Dis1Prom'] = listado_Valores_LogaritmoNatural[6]
        st.session_state.data['lnPerf2Dis2Prom'] = listado_Valores_LogaritmoNatural[7]
        st.session_state.data['lnPerf2Dis4Prom'] = listado_Valores_LogaritmoNatural[8]
        st.session_state.data['lnPerf2Dis6Prom'] = listado_Valores_LogaritmoNatural[9]
        st.session_state.data['lnPerf2Dis8Prom'] = listado_Valores_LogaritmoNatural[10]
        st.session_state.data['lnPerf2Dis10Prom'] = listado_Valores_LogaritmoNatural[11]
        
        st.markdown('### Listado de Valores de Logaritmo Natural')
        
        st.dataframe(listado_Valores_LogaritmoNatural)
        
    elif nroPerfiles == 3:
        
        listado_Valores_Promedio_Perfil1 = df_Valores_Promedio_Perfil1['Valor Promedio (Ω·m)'].tolist()
        listado_Valores_Promedio_Perfil2 = df_Valores_Promedio_Perfil2['Valor Promedio (Ω·m)'].tolist()
        listado_Valores_Promedio_Perfil3 = df_Valores_Promedio_Perfil3['Valor Promedio (Ω·m)'].tolist()
        
        listado_Valores_A_Realizar_Logaritmo = listado_Valores_Promedio_Perfil1 + listado_Valores_Promedio_Perfil2 + listado_Valores_Promedio_Perfil3
        
        listado_Valores_LogaritmoNatural = [round(math.log(v), 4) if v > 0 else 0 for v in listado_Valores_A_Realizar_Logaritmo]
        
        st.session_state.data['lnPerf1Dis1Prom'] = listado_Valores_LogaritmoNatural[0]
        st.session_state.data['lnPerf1Dis2Prom'] = listado_Valores_LogaritmoNatural[1]
        st.session_state.data['lnPerf1Dis4Prom'] = listado_Valores_LogaritmoNatural[2]
        st.session_state.data['lnPerf1Dis6Prom'] = listado_Valores_LogaritmoNatural[3]
        st.session_state.data['lnPerf1Dis8Prom'] = listado_Valores_LogaritmoNatural[4]
        st.session_state.data['lnPerf1Dis10Prom'] = listado_Valores_LogaritmoNatural[5]
        
        st.session_state.data['lnPerf2Dis1Prom'] = listado_Valores_LogaritmoNatural[6]
        st.session_state.data['lnPerf2Dis2Prom'] = listado_Valores_LogaritmoNatural[7]
        st.session_state.data['lnPerf2Dis4Prom'] = listado_Valores_LogaritmoNatural[8]
        st.session_state.data['lnPerf2Dis6Prom'] = listado_Valores_LogaritmoNatural[9]
        st.session_state.data['lnPerf2Dis8Prom'] = listado_Valores_LogaritmoNatural[10]
        st.session_state.data['lnPerf2Dis10Prom'] = listado_Valores_LogaritmoNatural[11]
        
        st.session_state.data['lnPerf3Dis1Prom'] = listado_Valores_LogaritmoNatural[12]
        st.session_state.data['lnPerf3Dis2Prom'] = listado_Valores_LogaritmoNatural[13]
        st.session_state.data['lnPerf3Dis4Prom'] = listado_Valores_LogaritmoNatural[14]
        st.session_state.data['lnPerf3Dis6Prom'] = listado_Valores_LogaritmoNatural[15]
        st.session_state.data['lnPerf3Dis8Prom'] = listado_Valores_LogaritmoNatural[16]
        st.session_state.data['lnPerf3Dis10Prom'] = listado_Valores_LogaritmoNatural[17]
        
        st.markdown('### Listado de Valores de Logaritmo Natural')
        
        st.dataframe(listado_Valores_LogaritmoNatural)
        
    elif nroPerfiles == 4:
        
        listado_Valores_Promedio_Perfil1 = df_Valores_Promedio_Perfil1['Valor Promedio (Ω·m)'].tolist()
        listado_Valores_Promedio_Perfil2 = df_Valores_Promedio_Perfil2['Valor Promedio (Ω·m)'].tolist()
        listado_Valores_Promedio_Perfil3 = df_Valores_Promedio_Perfil3['Valor Promedio (Ω·m)'].tolist()
        listado_Valores_Promedio_Perfil4 = df_Valores_Promedio_Perfil4['Valor Promedio (Ω·m)'].tolist()
        
        listado_Valores_A_Realizar_Logaritmo = listado_Valores_Promedio_Perfil1 + listado_Valores_Promedio_Perfil2 + listado_Valores_Promedio_Perfil3 + listado_Valores_Promedio_Perfil4
        
        listado_Valores_LogaritmoNatural = [round(math.log(v), 6) if v > 0 else 0 for v in listado_Valores_A_Realizar_Logaritmo]
        
        st.session_state.data['lnPerf1Dis1Prom'] = listado_Valores_LogaritmoNatural[0]
        st.session_state.data['lnPerf1Dis2Prom'] = listado_Valores_LogaritmoNatural[1]
        st.session_state.data['lnPerf1Dis4Prom'] = listado_Valores_LogaritmoNatural[2]
        st.session_state.data['lnPerf1Dis6Prom'] = listado_Valores_LogaritmoNatural[3]
        st.session_state.data['lnPerf1Dis8Prom'] = listado_Valores_LogaritmoNatural[4]
        st.session_state.data['lnPerf1Dis10Prom'] = listado_Valores_LogaritmoNatural[5]
        
        st.session_state.data['lnPerf2Dis1Prom'] = listado_Valores_LogaritmoNatural[6]
        st.session_state.data['lnPerf2Dis2Prom'] = listado_Valores_LogaritmoNatural[7]
        st.session_state.data['lnPerf2Dis4Prom'] = listado_Valores_LogaritmoNatural[8]
        st.session_state.data['lnPerf2Dis6Prom'] = listado_Valores_LogaritmoNatural[9]
        st.session_state.data['lnPerf2Dis8Prom'] = listado_Valores_LogaritmoNatural[10]
        st.session_state.data['lnPerf2Dis10Prom'] = listado_Valores_LogaritmoNatural[11]
        
        st.session_state.data['lnPerf3Dis1Prom'] = listado_Valores_LogaritmoNatural[12]
        st.session_state.data['lnPerf3Dis2Prom'] = listado_Valores_LogaritmoNatural[13]
        st.session_state.data['lnPerf3Dis4Prom'] = listado_Valores_LogaritmoNatural[14]
        st.session_state.data['lnPerf3Dis6Prom'] = listado_Valores_LogaritmoNatural[15]
        st.session_state.data['lnPerf3Dis8Prom'] = listado_Valores_LogaritmoNatural[16]
        st.session_state.data['lnPerf3Dis10Prom'] = listado_Valores_LogaritmoNatural[17]
        
        st.session_state.data['lnPerf4Dis1Prom'] = listado_Valores_LogaritmoNatural[18]
        st.session_state.data['lnPerf4Dis2Prom'] = listado_Valores_LogaritmoNatural[19]
        st.session_state.data['lnPerf4Dis4Prom'] = listado_Valores_LogaritmoNatural[20]
        st.session_state.data['lnPerf4Dis6Prom'] = listado_Valores_LogaritmoNatural[21]
        st.session_state.data['lnPerf4Dis8Prom'] = listado_Valores_LogaritmoNatural[22]
        st.session_state.data['lnPerf4Dis10Prom'] = listado_Valores_LogaritmoNatural[23]
        
        st.markdown('### Listado de Valores de Logaritmo Natural')
        
        st.dataframe(listado_Valores_LogaritmoNatural)
        
    #st.markdown("### Valores Promedio por cada Perfil")
    
    #st.dataframe(df_Valores_Promedio_Perfil1)
    #st.dataframe(df_Valores_Promedio_Perfil2)
    #st.dataframe(df_Valores_Promedio_Perfil3)
    #st.dataframe(df_Valores_Promedio_Perfil4)
    
    val_Suma_Listado_Promedios = sum(listado_Valores_A_Realizar_Logaritmo)
    val_Suma_Listado_Logaritmos = sum(listado_Valores_LogaritmoNatural)
        
    
    val_Promedio_Listado_Promedios = round(val_Suma_Listado_Promedios / len(listado_Valores_A_Realizar_Logaritmo), 2) if len(listado_Valores_A_Realizar_Logaritmo) > 0 else 0
    val_Promedio_Listado_Logaritmos = round(val_Suma_Listado_Logaritmos / len(listado_Valores_LogaritmoNatural), 2) if len(listado_Valores_LogaritmoNatural) > 0 else 0
    
    
    if nroPerfiles == 1:
        
        st.markdown('### Listado de Valores Elevados al Cuadrado respecto al Logaritmo')
    
        listado_Valores_Elevados = [(valor - val_Promedio_Listado_Logaritmos) ** 2 for valor in listado_Valores_LogaritmoNatural]
        
        st.session_state.data['cuadradoPerf1Dis1'] = listado_Valores_Elevados[0]
        st.session_state.data['cuadradoPerf1Dis2'] = listado_Valores_Elevados[1]
        st.session_state.data['cuadradoPerf1Dis4'] = listado_Valores_Elevados[2]
        st.session_state.data['cuadradoPerf1Dis6'] = listado_Valores_Elevados[3]
        st.session_state.data['cuadradoPerf1Dis8'] = listado_Valores_Elevados[4]
        st.session_state.data['cuadradoPerf1Dis10'] = listado_Valores_Elevados[5]
        
        st.dataframe(listado_Valores_Elevados)
        
    elif nroPerfiles == 2:
        
        st.markdown('### Listado de Valores Elevados al Cuadrado respecto al Logaritmo')
    
        listado_Valores_Elevados = [(valor - val_Promedio_Listado_Logaritmos) ** 2 for valor in listado_Valores_LogaritmoNatural]
        
        st.session_state.data['cuadradoPerf1Dis1'] = listado_Valores_Elevados[0]
        st.session_state.data['cuadradoPerf1Dis2'] = listado_Valores_Elevados[1]
        st.session_state.data['cuadradoPerf1Dis4'] = listado_Valores_Elevados[2]
        st.session_state.data['cuadradoPerf1Dis6'] = listado_Valores_Elevados[3]
        st.session_state.data['cuadradoPerf1Dis8'] = listado_Valores_Elevados[4]
        st.session_state.data['cuadradoPerf1Dis10'] = listado_Valores_Elevados[5]
        
        st.session_state.data['cuadradoPerf2Dis1'] = listado_Valores_Elevados[6]
        st.session_state.data['cuadradoPerf2Dis2'] = listado_Valores_Elevados[7]
        st.session_state.data['cuadradoPerf2Dis4'] = listado_Valores_Elevados[8]
        st.session_state.data['cuadradoPerf2Dis6'] = listado_Valores_Elevados[9]
        st.session_state.data['cuadradoPerf2Dis8'] = listado_Valores_Elevados[10]
        st.session_state.data['cuadradoPerf2Dis10'] = listado_Valores_Elevados[11]
        
        st.dataframe(listado_Valores_Elevados)
        
    elif nroPerfiles == 3:
        
        st.markdown('### Listado de Valores Elevados al Cuadrado respecto al Logaritmo')
    
        listado_Valores_Elevados = [(valor - val_Promedio_Listado_Logaritmos) ** 2 for valor in listado_Valores_LogaritmoNatural]
        
        st.session_state.data['cuadradoPerf1Dis1'] = listado_Valores_Elevados[0]
        st.session_state.data['cuadradoPerf1Dis2'] = listado_Valores_Elevados[1]
        st.session_state.data['cuadradoPerf1Dis4'] = listado_Valores_Elevados[2]
        st.session_state.data['cuadradoPerf1Dis6'] = listado_Valores_Elevados[3]
        st.session_state.data['cuadradoPerf1Dis8'] = listado_Valores_Elevados[4]
        st.session_state.data['cuadradoPerf1Dis10'] = listado_Valores_Elevados[5]
        
        st.session_state.data['cuadradoPerf2Dis1'] = listado_Valores_Elevados[6]
        st.session_state.data['cuadradoPerf2Dis2'] = listado_Valores_Elevados[7]
        st.session_state.data['cuadradoPerf2Dis4'] = listado_Valores_Elevados[8]
        st.session_state.data['cuadradoPerf2Dis6'] = listado_Valores_Elevados[9]
        st.session_state.data['cuadradoPerf2Dis8'] = listado_Valores_Elevados[10]
        st.session_state.data['cuadradoPerf2Dis10'] = listado_Valores_Elevados[11]
        
        st.session_state.data['cuadradoPerf3Dis1'] = listado_Valores_Elevados[12]
        st.session_state.data['cuadradoPerf3Dis2'] = listado_Valores_Elevados[13]
        st.session_state.data['cuadradoPerf3Dis4'] = listado_Valores_Elevados[14]
        st.session_state.data['cuadradoPerf3Dis6'] = listado_Valores_Elevados[15]
        st.session_state.data['cuadradoPerf3Dis8'] = listado_Valores_Elevados[16]
        st.session_state.data['cuadradoPerf3Dis10'] = listado_Valores_Elevados[17]
        
        st.dataframe(listado_Valores_Elevados)
        
    elif nroPerfiles == 4:
        
        st.markdown('### Listado de Valores Elevados al Cuadrado respecto al Logaritmo')
    
        listado_Valores_Elevados = [(valor - val_Promedio_Listado_Logaritmos) ** 2 for valor in listado_Valores_LogaritmoNatural]
        
        st.session_state.data['cuadradoPerf1Dis1'] = listado_Valores_Elevados[0]
        st.session_state.data['cuadradoPerf1Dis2'] = listado_Valores_Elevados[1]
        st.session_state.data['cuadradoPerf1Dis4'] = listado_Valores_Elevados[2]
        st.session_state.data['cuadradoPerf1Dis6'] = listado_Valores_Elevados[3]
        st.session_state.data['cuadradoPerf1Dis8'] = listado_Valores_Elevados[4]
        st.session_state.data['cuadradoPerf1Dis10'] = listado_Valores_Elevados[5]
        
        st.session_state.data['cuadradoPerf2Dis1'] = listado_Valores_Elevados[6]
        st.session_state.data['cuadradoPerf2Dis2'] = listado_Valores_Elevados[7]
        st.session_state.data['cuadradoPerf2Dis4'] = listado_Valores_Elevados[8]
        st.session_state.data['cuadradoPerf2Dis6'] = listado_Valores_Elevados[9]
        st.session_state.data['cuadradoPerf2Dis8'] = listado_Valores_Elevados[10]
        st.session_state.data['cuadradoPerf2Dis10'] = listado_Valores_Elevados[11]
        
        st.session_state.data['cuadradoPerf3Dis1'] = listado_Valores_Elevados[12]
        st.session_state.data['cuadradoPerf3Dis2'] = listado_Valores_Elevados[13]
        st.session_state.data['cuadradoPerf3Dis4'] = listado_Valores_Elevados[14]
        st.session_state.data['cuadradoPerf3Dis6'] = listado_Valores_Elevados[15]
        st.session_state.data['cuadradoPerf3Dis8'] = listado_Valores_Elevados[16]
        st.session_state.data['cuadradoPerf3Dis10'] = listado_Valores_Elevados[17]
        
        st.session_state.data['cuadradoPerf4Dis1'] = listado_Valores_Elevados[18]
        st.session_state.data['cuadradoPerf4Dis2'] = listado_Valores_Elevados[19]
        st.session_state.data['cuadradoPerf4Dis4'] = listado_Valores_Elevados[20]
        st.session_state.data['cuadradoPerf4Dis6'] = listado_Valores_Elevados[21]
        st.session_state.data['cuadradoPerf4Dis8'] = listado_Valores_Elevados[22]
        st.session_state.data['cuadradoPerf4Dis10'] = listado_Valores_Elevados[23]
        
        st.dataframe(listado_Valores_Elevados)
    
    
    val_Suma_Listado_Elevados = sum(listado_Valores_Elevados)
    val_Promedio_Listado_Elevados = round(sum(listado_Valores_Elevados) / len(listado_Valores_Elevados), 2) if len(listado_Valores_Elevados) > 0 else 0
    
    
    listado_A_Mostrar = [val_Suma_Listado_Promedios, val_Suma_Listado_Logaritmos, val_Suma_Listado_Elevados, val_Promedio_Listado_Promedios, val_Promedio_Listado_Logaritmos, val_Promedio_Listado_Elevados]
    
    st.session_state.data['valSumaListadoPromedios'] = listado_A_Mostrar[0]
    st.session_state.data['valSumaListadoLogaritmos'] = listado_A_Mostrar[1]
    st.session_state.data['valSumaListadoElevados'] = listado_A_Mostrar[2]
    
    st.session_state.data['valPromedioListadoPromedios'] = listado_A_Mostrar[3]
    st.session_state.data['valPromedioListadoLogaritmos'] = listado_A_Mostrar[4]
    st.session_state.data['valPromedioListadoElevados'] = listado_A_Mostrar[5]
    
    st.markdown('### Listado de Valores de Sumatorias y Promedios')
    
    st.dataframe(listado_A_Mostrar)
    
    
    
    #media = sum(listado_Valores_A_Realizar_Logaritmo) / len(listado_Valores_A_Realizar_Logaritmo)
    #varianza = sum((x - media) ** 2 for x in listado_Valores_A_Realizar_Logaritmo) / len(listado_Valores_A_Realizar_Logaritmo)  # poblacional
    #desviacion_Estandar = varianza ** 0.5
    
    desviacion_Estandar = np.std(listado_Valores_A_Realizar_Logaritmo)
    
    st.session_state.data['valDesvEstandar'] = desviacion_Estandar
    
    st.markdown('### Valor de Desviación Estándar de los Promedios de los Perfiles')
    
    st.dataframe([desviacion_Estandar])
    
    st.markdown('### Valor de la Raíz Cuadrada del Promedio de los Valores Elevados al Cuadrado respecto al Logaritmo')
    
    val_Raiz_Promedio_Elevados = round(val_Promedio_Listado_Elevados ** 0.5, 2) if val_Promedio_Listado_Elevados >= 0 else 0
    
    st.session_state.data['valRaizPromedioElevados'] = val_Raiz_Promedio_Elevados
    
    st.dataframe([val_Raiz_Promedio_Elevados])
    
    st.markdown('### Valor del Anti logaritmo basado en el Promedio del Logaritmo Natural y la Raíz Cuadrada del Promedio de los Valores Elevados al Cuadrado respecto al Logaritmo')
    
    val_AntiLogaritmo = (val_Raiz_Promedio_Elevados * 0.524411 + val_Promedio_Listado_Logaritmos)
    
    st.session_state.data['valAntiLogaritmo'] = val_AntiLogaritmo
    
    st.dataframe([val_AntiLogaritmo])
    
    st.markdown('### Valor de Resistividad con el 70%')
    
    val_Resistividad = round(math.exp(val_AntiLogaritmo), 2) if val_AntiLogaritmo else 0
    
    st.session_state.data['valResistividad'] = val_Resistividad
    
    st.dataframe([val_Resistividad])
    
    st.markdown('### Resultado de la tabla Box-Cox')
    
    df_Tabla_BoxCox = pd.DataFrame({
        'Distancia de Separación entre Electrodos (m)': [1, 2, 4, 6, 8, 10],
        'BOX-COX (Ohmios-m)': [val_Resistividad, val_Resistividad, val_Resistividad, val_Resistividad, val_Resistividad, val_Resistividad]
    })
    
    st.dataframe(df_Tabla_BoxCox)
    
    
    st.markdown('### Tabla de Medidas de Resistividad por Perfil [Ωm], Promedio [Ωm] y Box-Cox [Ωm]')
    
    if nroPerfiles == 1:
        
    
        df_Tabla_Medidas_Resistividad = pd.DataFrame({
            'Distancia (m)': [1, 2, 4, 6, 8, 10],
            'Perfil 1 [Ωm]': df_Valores_Promedio_Perfil1['Valor Promedio (Ω·m)'].tolist(),
            'Promedio [Ωm]': val_Promedio_Listado_Promedios if nroPerfiles else ['N/A']*6,
            'BOX-COX [Ωm]': [val_Resistividad]*6
        })
        
        st.dataframe(df_Tabla_Medidas_Resistividad)
        
        st.markdown('### Gráfica de Resistividad de Terreno')
        
        var_Titulo = f'PROYECTO {str(st.session_state.data["nombreProyecto"]).upper()}'
        
        # Generar el buffer PNG
        buf = plot_resistividad_to_buffer(df_Tabla_Medidas_Resistividad, titulo_proyecto=f'{var_Titulo}')

        # Opción 1: pasar bytes directamente a Streamlit
        png_bytes = buf.getvalue()
        st.image(png_bytes, use_container_width=True)
        
    elif nroPerfiles == 2:
        
        df_Tabla_Medidas_Resistividad = pd.DataFrame({
            'Distancia (m)': [1, 2, 4, 6, 8, 10],
            'Perfil 1 [Ωm]': df_Valores_Promedio_Perfil1['Valor Promedio (Ω·m)'].tolist(),
            'Perfil 2 [Ωm]': df_Valores_Promedio_Perfil2['Valor Promedio (Ω·m)'].tolist(),
            'Promedio [Ωm]': val_Promedio_Listado_Promedios if nroPerfiles else ['N/A']*6,
            'BOX-COX [Ωm]': [val_Resistividad]*6
        })
        
        st.dataframe(df_Tabla_Medidas_Resistividad)
        
        st.markdown('### Gráfica de Resistividad de Terreno')
        
        var_Titulo = f'PROYECTO {str(st.session_state.data["nombreProyecto"]).upper()}'
        
        # Generar el buffer PNG
        buf = plot_resistividad_to_buffer(df_Tabla_Medidas_Resistividad, titulo_proyecto=f'{var_Titulo}')
        
        datos['imgCurvaResistividad'] = InlineImage(st.session_state.doc, buf, Cm(18))

        # Opción 1: pasar bytes directamente a Streamlit
        png_bytes = buf.getvalue()
        st.image(png_bytes, use_container_width=True)
        
    elif nroPerfiles == 3:
        
        df_Tabla_Medidas_Resistividad = pd.DataFrame({
            'Distancia (m)': [1, 2, 4, 6, 8, 10],
            'Perfil 1 [Ωm]': df_Valores_Promedio_Perfil1['Valor Promedio (Ω·m)'].tolist(),
            'Perfil 2 [Ωm]': df_Valores_Promedio_Perfil2['Valor Promedio (Ω·m)'].tolist(),
            'Perfil 3 [Ωm]': df_Valores_Promedio_Perfil3['Valor Promedio (Ω·m)'].tolist(),
            'Promedio [Ωm]': val_Promedio_Listado_Promedios if nroPerfiles else ['N/A']*6,
            'BOX-COX [Ωm]': [val_Resistividad]*6
        })
        
        st.dataframe(df_Tabla_Medidas_Resistividad)
        
        st.markdown('### Gráfica de Resistividad de Terreno')
        
        var_Titulo = f'PROYECTO {str(st.session_state.data["nombreProyecto"]).upper()}'
        
        # Generar el buffer PNG
        buf = plot_resistividad_to_buffer(df_Tabla_Medidas_Resistividad, titulo_proyecto=f'{var_Titulo}')
        
        datos['imgCurvaResistividad'] = InlineImage(st.session_state.doc, buf, Cm(18))

        # Opción 1: pasar bytes directamente a Streamlit
        png_bytes = buf.getvalue()
        st.image(png_bytes, use_container_width=True)
        
    elif nroPerfiles == 4:
        
        df_Tabla_Medidas_Resistividad = pd.DataFrame({
            'Distancia (m)': [1, 2, 4, 6, 8, 10],
            'Perfil 1 [Ωm]': df_Valores_Promedio_Perfil1['Valor Promedio (Ω·m)'].tolist(),
            'Perfil 2 [Ωm]': df_Valores_Promedio_Perfil2['Valor Promedio (Ω·m)'].tolist(),
            'Perfil 3 [Ωm]': df_Valores_Promedio_Perfil3['Valor Promedio (Ω·m)'].tolist(),
            'Perfil 4 [Ωm]': df_Valores_Promedio_Perfil4['Valor Promedio (Ω·m)'].tolist(),
            'Promedio [Ωm]': val_Promedio_Listado_Promedios if nroPerfiles else ['N/A']*6,
            'BOX-COX [Ωm]': [val_Resistividad]*6
        })
        
        st.dataframe(df_Tabla_Medidas_Resistividad)
        
        st.markdown('### Gráfica de Resistividad de Terreno')
        
        var_Titulo = f'PROYECTO {str(st.session_state.data["nombreProyecto"]).upper()}'
        
        # Generar el buffer PNG
        buf = plot_resistividad_to_buffer(df_Tabla_Medidas_Resistividad, titulo_proyecto=f'{var_Titulo}')
        
        datos['imgCurvaResistividad'] = InlineImage(st.session_state.doc, buf, Cm(18))

        # Opción 1: pasar bytes directamente a Streamlit
        png_bytes = buf.getvalue()
        st.image(png_bytes, use_container_width=True)
    
    
    for nroPerfil in range(1, nroPerfiles + 1):
        
        st.markdown("---")  # Línea divisoria entre perfiles
        
        for nroPrueba in range(1, 7):
            data_key = f"imgPerf{nroPerfil}Prueba{nroPrueba}"

            uploaded = st.file_uploader(
                f"Imagen de Perfil #{nroPerfil} - Prueba #{nroPrueba}",
                type=["png", "jpg", "jpeg"],
                key=data_key
            )

            if uploaded:
                buf = io.BytesIO(uploaded.read())
                buf.seek(0)
                datos[data_key] = InlineImage(st.session_state.doc, buf, Cm(8), Cm(6))
            else:
                datos[data_key] = None
                
                
    st.markdown("---")
    
    key_Img_Campo1 = f"imgCampo1"
    uploaded_1 = st.file_uploader(f"Imagen de Prueba en Campo", type=['png','jpg','jpeg'], key=key_Img_Campo1)
    if uploaded_1:
        buf = io.BytesIO(uploaded_1.read())
        buf.seek(0)
        datos[key_Img_Campo1] = InlineImage(st.session_state.doc, buf, Cm(14))
    else:
        datos[key_Img_Campo1] = None
        
    st.markdown("---")
        
    key_Img_Campo2 = f"imgCampo2"
    uploaded_2 = st.file_uploader(f"Imagen de Prueba en Campo", type=['png','jpg','jpeg'], key=key_Img_Campo2)
    if uploaded_2:
        buf = io.BytesIO(uploaded_2.read())
        buf.seek(0)
        datos[key_Img_Campo2] = InlineImage(st.session_state.doc, buf, Cm(14))
    else:
        datos[key_Img_Campo2] = None
        
        
    if st.session_state.data['tipoCoordenada'] == "Urbano":
        
        if st.session_state.data['latitud'] and st.session_state.data['longitud']:
            try:
                lat = float(str(datos['latitud']).replace(',', '.'))
                lon = float(str(datos['longitud']).replace(',', '.'))
                mapa = StaticMap(600, 400)
                mapa.add_marker(CircleMarker((lon, lat), 'red', 12))
                img_map = mapa.render()
                buf_map = io.BytesIO()
                img_map.save(buf_map, format='PNG')
                buf_map.seek(0)
                datos['imgMapsProyecto'] = InlineImage(st.session_state.doc, buf_map, Cm(18))
            except Exception as e:
                st.error(f"Coordenadas inválidas para el mapa. {e}")
        else:
            st.error("Faltan coordenadas para el mapa.")
                
    else:
            
        if st.session_state.data['latitud'] and st.session_state.data['longitud']:
            try:
                lat = float(str(st.session_state.data['latitud']).replace(',', '.'))
                
                lon = float(str(st.session_state.data['longitud']).replace(',', '.'))
                
                st.warning(f"Prueba de coordenada en modo rural (latitud): {lat}")
                st.warning(f"Prueba de coordenada en modo rural (longitud): {lon}")
                    
                png_bytes = get_map_png_bytes(lon, lat, buffer_m=300, zoom=17)
                    
                buf_map = io.BytesIO(png_bytes)
                buf_map.seek(0)
                datos['imgMapsProyecto'] = InlineImage(st.session_state.doc, buf_map, Cm(18))
            except Exception as e:
                st.error(f"Coordenadas inválidas para el mapa. {e}")
        else:
            st.error("Faltan coordenadas para el mapa.")

    if st.button("Generar Word"):
        doc = st.session_state.doc
        # Añadir fecha al contexto
        ahora = datetime.now()
        meses = ["ENERO","FEBRERO","MARZO","ABRIL","MAYO","JUNIO","JULIO","AGOSTO","SEPTIEMBRE","OCTUBRE","NOVIEMBRE","DICIEMBRE"]
        datos['dia'] = ahora.day
        datos['mes'] = meses[ahora.month-1]
        datos['anio'] = ahora.year

        doc.render(datos)
        output = io.BytesIO()
        doc.save(output)
        output.seek(0)
        st.download_button(
            "Descargar Reporte Word",
            data=output,
            file_name="reporteProtocoloResistividad.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
                
    
