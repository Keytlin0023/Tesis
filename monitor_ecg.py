import sys
import os
import csv
import threading
import time
from datetime import datetime
from collections import deque

import numpy as np
import serial
import pyqtgraph as pg

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QFrame,
    QPushButton,
    QSpinBox,
    QMessageBox,
    QLineEdit,
    QFileDialog,
    QProgressBar,
    QGraphicsDropShadowEffect,
)


PUERTO = "COM3"
BAUDIOS = 460800

FRECUENCIA_ECG = 500
FRECUENCIA_GSR = 50
VENTANA_SEGUNDOS = 5
MAX_MUESTRAS_ECG = FRECUENCIA_ECG * VENTANA_SEGUNDOS
CENTRO_ECG = 2048.0


ESTABILIZACION_PREDETERMINADA_S = 60
FRACCION_ADAPTACION = 0.50


EXTENSION_BASAL_GSR_S = 0
MAX_EXTENSION_BASAL_GSR_S = 0


VENTANA_EVALUACION_GSR_CORTA_S = 10.0
VENTANA_EVALUACION_GSR_LARGA_S = 20.0
VENTANA_CALCULO_BASAL_GSR_S = 30.0
MIN_SEGUNDOS_BASAL_GSR = 20.0


MAX_PENDIENTE_GSR_CORTA_PCT_MIN = 1.5
MAX_PENDIENTE_GSR_LARGA_PCT_MIN = 2.0

TIMEOUT_GSR_S = 2.0
TIMEOUT_PPG_S = 2.0
TIEMPO_ESTABILIZACION_PPG_S = 5.0


RR_MIN = 0.38
RR_MAX = 2.00
PERIODO_REFRACTARIO = 0.30
MUESTRAS_REFRACTARIO = int(PERIODO_REFRACTARIO * FRECUENCIA_ECG)
TOLERANCIA_RR = 0.35
TIMEOUT_LATIDO = 2.5
MUESTRAS_TIMEOUT = int(TIMEOUT_LATIDO * FRECUENCIA_ECG)
FACTOR_UMBRAL = 4.0
UMBRAL_MINIMO = 40.0
RR_MINIMOS_FC = 3
RR_MINIMOS_HRV = 20
MAX_RR_HISTORIAL = 120


datos_ecg = deque(maxlen=MAX_MUESTRAS_ECG)
picos_r = deque(maxlen=150)

temperatura_actual = np.nan
estado_leads_actual = 0
ultimo_ecg_raw = np.nan
ultimo_ecg_filtrado = np.nan
ultimo_numero_ecg = 0


red_actual = 0
ir_actual = 0
dedo_ppg_actual = 0
spo2_actual = np.nan
fc_ppg_actual = np.nan
ppg_valido_sensor = 0
ppg_recibido = False
ultimo_ppg_rx = 0.0
ppg_dedo_desde = None


gsr_raw_actual = 0
gsr_filtrado_actual = np.nan
gsr_recibido = False
ultimo_gsr_rx = 0.0
ultimo_gsr_timestamp_s = None

gsr_muestras_basal = deque(maxlen=FRECUENCIA_GSR * 240)
gsr_basal_provisional = np.nan
gsr_basal = np.nan
gsr_delta = np.nan
gsr_delta_pct = np.nan
gsr_pendiente_pct_min = np.nan
gsr_pendiente_corta_pct_min = np.nan
gsr_basal_estable = False
gsr_extension_usada_s = 0


estado_serial = "Esperando ESP32..."
ejecutando = True
bloqueo = threading.Lock()


ultimo_timestamp_raw_us = None
offset_timestamp_us = 0
ultimo_timestamp_esp32_us = None


ultimo_contador_tipo = {"E": None, "P": None, "G": None, "T": None}
perdidas_tipo = {"E": 0, "P": 0, "G": 0, "T": 0}


modo_prueba = "idle"

inicio_estabilizacion = None
inicio_registro = None

duracion_estabilizacion_nominal = ESTABILIZACION_PREDETERMINADA_S
duracion_estabilizacion_efectiva = ESTABILIZACION_PREDETERMINADA_S
duracion_prueba_segundos = 60

timestamp_inicio_sesion_esp32_us = None
timestamp_inicio_prueba_esp32_us = None


CARPETA_REGISTROS = os.path.abspath("registros_ecg")
os.makedirs(CARPETA_REGISTROS, exist_ok=True)

nombre_base_archivo = ""
ruta_csv_actual = ""
archivo_csv = None
escritor_csv = None
contador_flush = 0

COLUMNAS_CSV = [
    "Nombre_registro",
    "FechaHora_PC",
    "Fase",
    "Tipo_muestra",
    "Numero_muestra",
    "Timestamp_ESP32_us",
    "Tiempo_sesion_s",
    "Tiempo_prueba_s",
    "Perdidas_desde_anterior",

    "ECG_RAW_ADC",
    "ECG_FILTRADO_ADC",
    "Electrodos_conectados",
    "Calidad_ECG",
    "Pico_R",
    "RR_actual_ms",
    "RR_estable_ms",
    "FC_ECG_BPM",

    "RMSSD_fase_ms",
    "SDNN_fase_ms",
    "RR_validos_fase",
    "RMSSD_basal_ms",
    "SDNN_basal_ms",
    "RR_validos_basal",
    "RMSSD_prueba_ms",
    "SDNN_prueba_ms",
    "RR_validos_prueba",

    "PPG_RED_RAW",
    "PPG_IR_RAW",
    "Dedo_PPG",
    "Estado_PPG",
    "SpO2_pct",
    "FC_PPG_BPM",
    "PPG_valido",

    "GSR_RAW_ADC",
    "GSR_FILTRADO",
    "GSR_BASAL",
    "GSR_DELTA_ADC",
    "GSR_DELTA_PCT",
    "Estado_GSR",
    "GSR_pendiente_basal_20s_pct_min",
    "GSR_pendiente_basal_10s_pct_min",
    "GSR_basal_estable",

    "Temperatura_C",
]


def tiempo_adaptacion_s():
    return float(duracion_estabilizacion_nominal) * FRACCION_ADAPTACION


def fase_actual(ahora=None):
    if modo_prueba == "idle":
        return "IDLE"

    if modo_prueba == "recording":
        return "PRUEBA"

    if inicio_estabilizacion is None:
        return "ADAPTACION"

    if ahora is None:
        ahora = time.monotonic()

    transcurrido = max(0.0, ahora - inicio_estabilizacion)

    if transcurrido < tiempo_adaptacion_s():
        return "ADAPTACION"

    return "BASAL"


def fase_por_timestamp(timestamp_us):
    if modo_prueba == "idle":
        return "IDLE"

    if timestamp_inicio_sesion_esp32_us is None:
        return "ADAPTACION" if modo_prueba == "stabilizing" else "PRUEBA"

    tiempo_sesion = max(
        0.0,
        (timestamp_us - timestamp_inicio_sesion_esp32_us) / 1e6,
    )

    if (
        timestamp_inicio_prueba_esp32_us is not None
        and timestamp_us >= timestamp_inicio_prueba_esp32_us
    ):
        return "PRUEBA"

    if tiempo_sesion < tiempo_adaptacion_s():
        return "ADAPTACION"

    return "BASAL"


class DetectorQRS:
    def __init__(self):
        self.historial_amplitud = deque(maxlen=FRECUENCIA_ECG * 2)
        self.intervalos_rr = deque(maxlen=MAX_RR_HISTORIAL)
        self.intervalos_rr_basal = deque(maxlen=600)
        self.intervalos_rr_prueba = deque(maxlen=1200)

        self.ultimo_pico = None
        self.fase_ultimo_pico = None

        self.magnitud_anterior2 = 0.0
        self.magnitud_anterior1 = 0.0
        self.ecg_anterior2 = CENTRO_ECG
        self.ecg_anterior1 = CENTRO_ECG
        self.indice_anterior2 = 0
        self.indice_anterior1 = 0

        self.umbral = UMBRAL_MINIMO
        self.contador_umbral = 0

        self.bpm = np.nan
        self.rr_actual_ms = np.nan
        self.rr_estable_ms = np.nan

        self.rmssd = np.nan
        self.sdnn = np.nan
        self.rmssd_basal = np.nan
        self.sdnn_basal = np.nan
        self.rmssd_prueba = np.nan
        self.sdnn_prueba = np.nan

        self.artefacto_hasta = -1
        self.rr_rechazados = 0
        self.rr_validos_totales = 0

    def reiniciar_completo(self):
        self.__init__()

    def perder_referencia(self):
        self.ultimo_pico = None
        self.fase_ultimo_pico = None
        self.bpm = np.nan
        self.rr_actual_ms = np.nan
        self.rr_estable_ms = np.nan

    def marcar_artefacto(self, indice_actual):
        self.artefacto_hasta = max(
            self.artefacto_hasta,
            indice_actual + int(1.5 * FRECUENCIA_ECG),
        )

    def hay_artefacto(self, indice_actual):
        return indice_actual <= self.artefacto_hasta

    def actualizar_umbral(self):
        self.contador_umbral += 1
        if self.contador_umbral < 25:
            return
        self.contador_umbral = 0

        if len(self.historial_amplitud) < FRECUENCIA_ECG // 2:
            self.umbral = UMBRAL_MINIMO
            return

        amplitudes = np.asarray(self.historial_amplitud, dtype=float)
        mediana = np.median(amplitudes)
        mad = np.median(np.abs(amplitudes - mediana))
        sigma = 1.4826 * mad
        self.umbral = max(UMBRAL_MINIMO, mediana + FACTOR_UMBRAL * sigma)

    @staticmethod
    def _metricas_hrv(rr_deque):
        if len(rr_deque) < RR_MINIMOS_HRV:
            return np.nan, np.nan

        rr = np.asarray(rr_deque, dtype=float)
        diferencias = np.diff(rr)
        rmssd = np.sqrt(np.mean(diferencias ** 2)) * 1000.0
        sdnn = np.std(rr, ddof=1) * 1000.0
        return float(rmssd), float(sdnn)

    def actualizar_metricas(self):
        if self.intervalos_rr:
            rr = np.asarray(self.intervalos_rr, dtype=float)
            self.rr_actual_ms = rr[-1] * 1000.0

            if len(rr) >= RR_MINIMOS_FC:
                rr_estable = np.median(rr[-5:])
                self.rr_estable_ms = rr_estable * 1000.0
                if rr_estable > 0:
                    self.bpm = 60.0 / rr_estable

            self.rmssd, self.sdnn = self._metricas_hrv(self.intervalos_rr)

        self.rmssd_basal, self.sdnn_basal = self._metricas_hrv(
            self.intervalos_rr_basal
        )
        self.rmssd_prueba, self.sdnn_prueba = self._metricas_hrv(
            self.intervalos_rr_prueba
        )

    def metricas_fase(self, fase):
        if fase == "BASAL":
            return self.rmssd_basal, self.sdnn_basal, len(self.intervalos_rr_basal)
        if fase == "PRUEBA":
            return self.rmssd_prueba, self.sdnn_prueba, len(self.intervalos_rr_prueba)
        return np.nan, np.nan, 0

    def comprobar_timeout(self, indice_actual):
        if self.ultimo_pico is None:
            return

        if indice_actual - self.ultimo_pico > MUESTRAS_TIMEOUT:
            self.marcar_artefacto(indice_actual)
            self.perder_referencia()

    def _aceptar_pico(self, candidato_indice, candidato_ecg, fase, rr=None):
        if rr is not None:
            self.intervalos_rr.append(rr)
            self.rr_validos_totales += 1


            if self.fase_ultimo_pico == fase:
                if fase == "BASAL":
                    self.intervalos_rr_basal.append(rr)
                elif fase == "PRUEBA":
                    self.intervalos_rr_prueba.append(rr)

            self.actualizar_metricas()

        self.ultimo_pico = candidato_indice
        self.fase_ultimo_pico = fase
        return True, candidato_indice, candidato_ecg

    def procesar(self, valor_ecg, indice_muestra, fase):
        self.comprobar_timeout(indice_muestra)

        ecg_centrado = float(valor_ecg) - CENTRO_ECG
        magnitud = abs(ecg_centrado)
        self.historial_amplitud.append(magnitud)
        self.actualizar_umbral()

        detectado = False
        indice_pico = None
        valor_pico = None

        es_maximo = (
            self.magnitud_anterior1 > self.magnitud_anterior2
            and self.magnitud_anterior1 >= magnitud
            and self.magnitud_anterior1 > self.umbral
        )

        if es_maximo:
            candidato_indice = self.indice_anterior1
            candidato_ecg = self.ecg_anterior1

            if self.ultimo_pico is None:
                detectado, indice_pico, valor_pico = self._aceptar_pico(
                    candidato_indice, candidato_ecg, fase
                )
            else:
                diferencia = candidato_indice - self.ultimo_pico

                if diferencia >= MUESTRAS_REFRACTARIO:
                    rr = diferencia / FRECUENCIA_ECG

                    if rr < RR_MIN:
                        self.rr_rechazados += 1
                        self.marcar_artefacto(candidato_indice)

                    elif rr > RR_MAX:
                        self.rr_rechazados += 1
                        self.marcar_artefacto(candidato_indice)
                        detectado, indice_pico, valor_pico = self._aceptar_pico(
                            candidato_indice, candidato_ecg, fase
                        )

                    else:
                        coherente = True
                        mediana_rr = None

                        if len(self.intervalos_rr) >= 4:
                            recientes = np.asarray(list(self.intervalos_rr)[-7:], dtype=float)
                            mediana_rr = np.median(recientes)
                            limite_inferior = mediana_rr * (1.0 - TOLERANCIA_RR)
                            limite_superior = mediana_rr * (1.0 + TOLERANCIA_RR)
                            coherente = limite_inferior <= rr <= limite_superior

                        if coherente:
                            detectado, indice_pico, valor_pico = self._aceptar_pico(
                                candidato_indice, candidato_ecg, fase, rr=rr
                            )
                        else:
                            self.rr_rechazados += 1
                            self.marcar_artefacto(candidato_indice)

                            if mediana_rr is not None and rr > mediana_rr * 1.55:
                                detectado, indice_pico, valor_pico = self._aceptar_pico(
                                    candidato_indice, candidato_ecg, fase
                                )

        self.magnitud_anterior2 = self.magnitud_anterior1
        self.magnitud_anterior1 = magnitud
        self.ecg_anterior2 = self.ecg_anterior1
        self.ecg_anterior1 = valor_ecg
        self.indice_anterior2 = self.indice_anterior1
        self.indice_anterior1 = indice_muestra

        return detectado, indice_pico, valor_pico


detector_qrs = DetectorQRS()


def reiniciar_ppg_python():
    global spo2_actual, fc_ppg_actual, ppg_valido_sensor
    global ppg_dedo_desde, ppg_recibido, ultimo_ppg_rx

    spo2_actual = np.nan
    fc_ppg_actual = np.nan
    ppg_valido_sensor = 0


    ppg_recibido = False
    ultimo_ppg_rx = 0.0
    ppg_dedo_desde = None


def ppg_listo_ahora(ahora=None):
    if ahora is None:
        ahora = time.monotonic()

    if not ppg_recibido:
        return False
    if ahora - ultimo_ppg_rx > TIMEOUT_PPG_S:
        return False
    if dedo_ppg_actual != 1:
        return False
    if ppg_valido_sensor != 1:
        return False
    if not np.isfinite(spo2_actual) or not np.isfinite(fc_ppg_actual):
        return False
    if ppg_dedo_desde is None:
        return False
    if ahora - ppg_dedo_desde < TIEMPO_ESTABILIZACION_PPG_S:
        return False
    return True


def estado_ppg_ahora(ahora=None):
    if ahora is None:
        ahora = time.monotonic()


    if not ppg_recibido or ahora - ultimo_ppg_rx > TIMEOUT_PPG_S:
        return "SIN FLUJO"
    if dedo_ppg_actual != 1:
        return "SIN DEDO"
    if not ppg_listo_ahora(ahora):
        return "CALCULANDO"
    return "OK"


def reiniciar_gsr_python():
    global gsr_raw_actual, gsr_filtrado_actual
    global gsr_recibido, ultimo_gsr_rx, ultimo_gsr_timestamp_s
    global gsr_basal_provisional, gsr_basal
    global gsr_delta, gsr_delta_pct
    global gsr_pendiente_pct_min, gsr_pendiente_corta_pct_min
    global gsr_basal_estable
    global gsr_extension_usada_s

    gsr_raw_actual = 0
    gsr_filtrado_actual = np.nan
    gsr_recibido = False
    ultimo_gsr_rx = 0.0
    ultimo_gsr_timestamp_s = None
    gsr_muestras_basal.clear()
    gsr_basal_provisional = np.nan
    gsr_basal = np.nan
    gsr_delta = np.nan
    gsr_delta_pct = np.nan
    gsr_pendiente_pct_min = np.nan
    gsr_pendiente_corta_pct_min = np.nan
    gsr_basal_estable = False
    gsr_extension_usada_s = 0


def _muestras_gsr_ventana(segundos, ahora_sensor_s=None):
    if not gsr_muestras_basal:
        return []

    if ahora_sensor_s is None:
        ahora_sensor_s = gsr_muestras_basal[-1][0]

    limite = ahora_sensor_s - segundos
    return [
        (t, v)
        for t, v in gsr_muestras_basal
        if t >= limite and np.isfinite(v)
    ]


def actualizar_basal_provisional_gsr(ahora_sensor_s=None):
    global gsr_basal_provisional

    muestras = _muestras_gsr_ventana(
        VENTANA_CALCULO_BASAL_GSR_S,
        ahora_sensor_s,
    )

    if len(muestras) >= 5:
        valores = np.asarray([v for _, v in muestras], dtype=float)
        gsr_basal_provisional = float(np.median(valores))
    else:
        gsr_basal_provisional = np.nan


def _pendiente_gsr_pct_min(segundos, ahora_sensor_s=None):
    muestras = _muestras_gsr_ventana(segundos, ahora_sensor_s)

    if len(muestras) < 5:
        return np.nan, len(muestras)

    tiempos = np.asarray([t for t, _ in muestras], dtype=float)
    valores = np.asarray([v for _, v in muestras], dtype=float)
    tiempos = tiempos - tiempos[0]

    if tiempos[-1] <= 0:
        return np.nan, len(muestras)

    referencia = float(np.median(valores))
    if referencia == 0:
        return np.nan, len(muestras)

    pendiente_adc_s = float(np.polyfit(tiempos, valores, 1)[0])
    pendiente_pct_min = pendiente_adc_s / referencia * 60.0 * 100.0
    return float(pendiente_pct_min), len(muestras)


def evaluar_estabilidad_gsr(ahora_sensor_s=None):
    global gsr_pendiente_pct_min, gsr_pendiente_corta_pct_min

    pendiente_corta, n_corta = _pendiente_gsr_pct_min(
        VENTANA_EVALUACION_GSR_CORTA_S,
        ahora_sensor_s,
    )
    pendiente_larga, n_larga = _pendiente_gsr_pct_min(
        VENTANA_EVALUACION_GSR_LARGA_S,
        ahora_sensor_s,
    )

    gsr_pendiente_corta_pct_min = pendiente_corta
    gsr_pendiente_pct_min = pendiente_larga

    minimo_largo = int(FRECUENCIA_GSR * MIN_SEGUNDOS_BASAL_GSR)
    if n_larga < minimo_largo:
        return False, pendiente_larga, n_larga

    if not np.isfinite(pendiente_corta) or not np.isfinite(pendiente_larga):
        return False, pendiente_larga, n_larga

    estable = (
        abs(pendiente_corta) <= MAX_PENDIENTE_GSR_CORTA_PCT_MIN
        and
        abs(pendiente_larga) <= MAX_PENDIENTE_GSR_LARGA_PCT_MIN
    )

    return bool(estable), float(pendiente_larga), n_larga


def congelar_basal_gsr(forzar=False):
    global gsr_basal, gsr_basal_estable


    ahora_sensor_s = (
        gsr_muestras_basal[-1][0]
        if gsr_muestras_basal
        else None
    )

    actualizar_basal_provisional_gsr(ahora_sensor_s)
    estable, _, _ = evaluar_estabilidad_gsr(ahora_sensor_s)

    muestras = _muestras_gsr_ventana(
        VENTANA_CALCULO_BASAL_GSR_S,
        ahora_sensor_s,
    )
    suficientes = len(muestras) >= int(
        FRECUENCIA_GSR * MIN_SEGUNDOS_BASAL_GSR
    )

    if suficientes and np.isfinite(gsr_basal_provisional) and (estable or forzar):
        gsr_basal = float(gsr_basal_provisional)
        gsr_basal_estable = bool(estable)
    else:
        gsr_basal = np.nan
        gsr_basal_estable = False

    actualizar_derivados_gsr()
    return np.isfinite(gsr_basal)


def basal_gsr_para_mostrar():
    if np.isfinite(gsr_basal):
        return gsr_basal
    return gsr_basal_provisional


def actualizar_derivados_gsr():
    global gsr_delta, gsr_delta_pct

    if (
        np.isfinite(gsr_basal)
        and gsr_basal != 0
        and np.isfinite(gsr_filtrado_actual)
    ):
        gsr_delta = gsr_basal - gsr_filtrado_actual
        gsr_delta_pct = gsr_delta / gsr_basal * 100.0
    else:
        gsr_delta = np.nan
        gsr_delta_pct = np.nan


def estado_gsr_ahora(ahora=None):
    if ahora is None:
        ahora = time.monotonic()

    if not gsr_recibido or ahora - ultimo_gsr_rx > TIMEOUT_GSR_S:
        return "SIN DATOS"

    if gsr_raw_actual <= 5 or gsr_raw_actual >= 4090:
        return "SATURADO"

    fase = fase_actual(ahora)
    if fase == "ADAPTACION":
        return "ADAPTANDO"
    if fase == "BASAL":
        estable, _, n = evaluar_estabilidad_gsr()
        if n < int(FRECUENCIA_GSR * MIN_SEGUNDOS_BASAL_GSR):
            return "CALCULANDO BASAL"
        return "BASAL ESTABLE" if estable else "BASAL DERIVANDO"

    if not np.isfinite(gsr_basal):
        return "SIN BASAL"

    if not gsr_basal_estable:
        return "BASAL INESTABLE"

    return "OK"


def evaluar_calidad(ecg, estado_leads, indice_actual):
    if not estado_leads:
        return "SEÑAL BAJA"

    if len(ecg) < FRECUENCIA_ECG:
        return "INICIANDO"

    valores = np.asarray(
        [valor for _, valor in ecg[-FRECUENCIA_ECG * 2:]],
        dtype=float,
    )

    desviacion = np.std(valores)
    if desviacion < 8:
        return "SEÑAL BAJA"

    saturacion = np.mean((valores <= 5) | (valores >= 4090))
    if saturacion > 0.02:
        return "SEÑAL BAJA"

    if detector_qrs.hay_artefacto(indice_actual):
        return "INESTABLE"

    if desviacion > 700:
        return "INESTABLE"

    return "BUENA"


def convertir_float(texto):
    texto = texto.strip()
    if texto.lower() == "nan":
        return np.nan
    return float(texto)


def desenvolver_micros(timestamp_raw):
    global ultimo_timestamp_raw_us, offset_timestamp_us, ultimo_timestamp_esp32_us

    timestamp_raw = int(timestamp_raw) & 0xFFFFFFFF

    if ultimo_timestamp_raw_us is not None:
        if timestamp_raw < ultimo_timestamp_raw_us:
            diferencia = ultimo_timestamp_raw_us - timestamp_raw
            if diferencia > 0x80000000:
                offset_timestamp_us += 0x100000000

    ultimo_timestamp_raw_us = timestamp_raw
    resultado = offset_timestamp_us + timestamp_raw
    ultimo_timestamp_esp32_us = resultado
    return resultado


def actualizar_perdidas(tipo, numero):
    anterior = ultimo_contador_tipo[tipo]
    perdidas = 0

    if anterior is not None:
        if numero > anterior:
            perdidas = max(0, numero - anterior - 1)
        elif anterior > 0xFFFFFF00 and numero < 0x00000100:
            perdidas = max(0, (0xFFFFFFFF - anterior) + numero)

    ultimo_contador_tipo[tipo] = numero
    perdidas_tipo[tipo] += perdidas
    return perdidas


def reiniciar_control_sesion():
    global timestamp_inicio_sesion_esp32_us, timestamp_inicio_prueba_esp32_us

    timestamp_inicio_sesion_esp32_us = None
    timestamp_inicio_prueba_esp32_us = None

    for clave in ultimo_contador_tipo:
        ultimo_contador_tipo[clave] = None
        perdidas_tipo[clave] = 0


def limpiar_nombre_archivo(nombre):
    nombre = nombre.strip()
    if nombre.lower().endswith(".csv"):
        nombre = nombre[:-4]

    for caracter in '<>:"/\\|?*':
        nombre = nombre.replace(caracter, "_")

    return nombre.strip()


def generar_ruta_csv():
    nombre = limpiar_nombre_archivo(nombre_base_archivo)
    fecha = datetime.now().strftime("%Y-%m-%d")
    ruta = os.path.join(CARPETA_REGISTROS, f"{nombre} {fecha}.csv")

    if not os.path.exists(ruta):
        return ruta

    numero = 2
    while True:
        ruta = os.path.join(CARPETA_REGISTROS, f"{nombre} {fecha} ({numero}).csv")
        if not os.path.exists(ruta):
            return ruta
        numero += 1


def abrir_registro_csv():
    global archivo_csv, escritor_csv, ruta_csv_actual, contador_flush

    ruta_csv_actual = generar_ruta_csv()
    archivo_csv = open(ruta_csv_actual, "w", newline="", encoding="utf-8")
    escritor_csv = csv.DictWriter(archivo_csv, fieldnames=COLUMNAS_CSV)
    escritor_csv.writeheader()
    archivo_csv.flush()
    contador_flush = 0


def cerrar_registro_csv():
    global archivo_csv, escritor_csv

    if archivo_csv is not None:
        try:
            archivo_csv.flush()
            archivo_csv.close()
        except Exception:
            pass

    archivo_csv = None
    escritor_csv = None


def registrar_evento(tipo, numero, timestamp_us, perdidas=0, **datos):
    global contador_flush, timestamp_inicio_sesion_esp32_us, timestamp_inicio_prueba_esp32_us

    if escritor_csv is None or modo_prueba == "idle":
        return

    if timestamp_inicio_sesion_esp32_us is None:
        timestamp_inicio_sesion_esp32_us = timestamp_us

    fase = fase_por_timestamp(timestamp_us)
    tiempo_sesion = max(0.0, (timestamp_us - timestamp_inicio_sesion_esp32_us) / 1e6)

    if fase == "PRUEBA":
        if timestamp_inicio_prueba_esp32_us is None:
            timestamp_inicio_prueba_esp32_us = timestamp_us
        tiempo_prueba = max(0.0, (timestamp_us - timestamp_inicio_prueba_esp32_us) / 1e6)
        tiempo_prueba_txt = f"{tiempo_prueba:.6f}"
    else:
        tiempo_prueba_txt = ""

    fila = {columna: "" for columna in COLUMNAS_CSV}
    fila.update(
        {
            "Nombre_registro": nombre_base_archivo,
            "FechaHora_PC": datetime.now().isoformat(timespec="milliseconds"),
            "Fase": fase,
            "Tipo_muestra": tipo,
            "Numero_muestra": numero,
            "Timestamp_ESP32_us": timestamp_us,
            "Tiempo_sesion_s": f"{tiempo_sesion:.6f}",
            "Tiempo_prueba_s": tiempo_prueba_txt,
            "Perdidas_desde_anterior": perdidas,
        }
    )

    fila.update(datos)
    escritor_csv.writerow(fila)

    contador_flush += 1
    if contador_flush >= 1000:
        archivo_csv.flush()
        contador_flush = 0


def leer_serial():
    global ejecutando, estado_serial
    global temperatura_actual, estado_leads_actual
    global ultimo_ecg_raw, ultimo_ecg_filtrado, ultimo_numero_ecg
    global red_actual, ir_actual, dedo_ppg_actual
    global spo2_actual, fc_ppg_actual, ppg_valido_sensor
    global ppg_recibido, ultimo_ppg_rx, ppg_dedo_desde
    global gsr_raw_actual, gsr_filtrado_actual
    global gsr_recibido, ultimo_gsr_rx, ultimo_gsr_timestamp_s

    while ejecutando:
        try:
            estado_serial = f"Conectando a {PUERTO}..."
            puerto = serial.Serial(PUERTO, BAUDIOS, timeout=0.1)
            time.sleep(2)
            puerto.reset_input_buffer()
            estado_serial = f"Conectado: {PUERTO} @ {BAUDIOS} baud"

            ultimo_estado_leads = None

            while ejecutando:
                linea = puerto.readline()
                if not linea:
                    continue

                try:
                    texto = linea.decode("utf-8", errors="ignore").strip()
                    if not texto:
                        continue

                    partes = texto.split(",")
                    tipo = partes[0]


                    if tipo == "E" and len(partes) == 6:
                        numero = int(float(partes[1]))
                        timestamp_us = desenvolver_micros(int(float(partes[2])))
                        ecg_raw = float(partes[3])
                        ecg_filtrado = float(partes[4])
                        leads = int(float(partes[5]))

                        if not (0 <= ecg_raw <= 4095 and 0 <= ecg_filtrado <= 4095):
                            continue

                        ahora = time.monotonic()
                        fase = fase_por_timestamp(timestamp_us)

                        with bloqueo:
                            perdidas = actualizar_perdidas("E", numero) if modo_prueba != "idle" else 0

                            if (
                                ultimo_estado_leads is not None
                                and leads != ultimo_estado_leads
                                and leads == 0
                            ):
                                detector_qrs.perder_referencia()

                            ultimo_estado_leads = leads
                            estado_leads_actual = leads
                            ultimo_ecg_raw = ecg_raw
                            ultimo_ecg_filtrado = ecg_filtrado
                            ultimo_numero_ecg = numero

                            datos_ecg.append((numero, ecg_filtrado))

                            pico_r_detectado = 0
                            if leads == 1:
                                detectado, indice_pico, valor_pico = detector_qrs.procesar(
                                    ecg_filtrado,
                                    numero,
                                    fase,
                                )
                                if detectado:
                                    pico_r_detectado = 1
                                    picos_r.append((indice_pico, valor_pico))

                            calidad = evaluar_calidad(list(datos_ecg), leads, numero)
                            rmssd_fase, sdnn_fase, rr_fase = detector_qrs.metricas_fase(fase)

                            registrar_evento(
                                "ECG",
                                numero,
                                timestamp_us,
                                perdidas,
                                ECG_RAW_ADC=f"{ecg_raw:.0f}",
                                ECG_FILTRADO_ADC=f"{ecg_filtrado:.0f}",
                                Electrodos_conectados=leads,
                                Calidad_ECG=calidad,
                                Pico_R=pico_r_detectado,
                                RR_actual_ms=(
                                    f"{detector_qrs.rr_actual_ms:.1f}"
                                    if np.isfinite(detector_qrs.rr_actual_ms)
                                    else ""
                                ),
                                RR_estable_ms=(
                                    f"{detector_qrs.rr_estable_ms:.1f}"
                                    if np.isfinite(detector_qrs.rr_estable_ms)
                                    else ""
                                ),
                                FC_ECG_BPM=(
                                    f"{detector_qrs.bpm:.1f}"
                                    if np.isfinite(detector_qrs.bpm)
                                    else ""
                                ),
                                RMSSD_fase_ms=(
                                    f"{rmssd_fase:.2f}" if np.isfinite(rmssd_fase) else ""
                                ),
                                SDNN_fase_ms=(
                                    f"{sdnn_fase:.2f}" if np.isfinite(sdnn_fase) else ""
                                ),
                                RR_validos_fase=rr_fase,
                                RMSSD_basal_ms=(
                                    f"{detector_qrs.rmssd_basal:.2f}"
                                    if np.isfinite(detector_qrs.rmssd_basal)
                                    else ""
                                ),
                                SDNN_basal_ms=(
                                    f"{detector_qrs.sdnn_basal:.2f}"
                                    if np.isfinite(detector_qrs.sdnn_basal)
                                    else ""
                                ),
                                RR_validos_basal=len(detector_qrs.intervalos_rr_basal),
                                RMSSD_prueba_ms=(
                                    f"{detector_qrs.rmssd_prueba:.2f}"
                                    if np.isfinite(detector_qrs.rmssd_prueba)
                                    else ""
                                ),
                                SDNN_prueba_ms=(
                                    f"{detector_qrs.sdnn_prueba:.2f}"
                                    if np.isfinite(detector_qrs.sdnn_prueba)
                                    else ""
                                ),
                                RR_validos_prueba=len(detector_qrs.intervalos_rr_prueba),
                            )


                    elif tipo == "P" and len(partes) == 9:
                        numero = int(float(partes[1]))
                        timestamp_us = desenvolver_micros(int(float(partes[2])))
                        nuevo_red = int(float(partes[3]))
                        nuevo_ir = int(float(partes[4]))
                        nuevo_dedo = int(float(partes[5]))
                        nuevo_spo2 = convertir_float(partes[6])
                        nueva_fc_ppg = convertir_float(partes[7])
                        nuevo_valido = int(float(partes[8]))
                        ahora = time.monotonic()

                        with bloqueo:
                            perdidas = actualizar_perdidas("P", numero) if modo_prueba != "idle" else 0
                            dedo_anterior = dedo_ppg_actual

                            red_actual = nuevo_red
                            ir_actual = nuevo_ir
                            dedo_ppg_actual = nuevo_dedo
                            spo2_actual = nuevo_spo2
                            fc_ppg_actual = nueva_fc_ppg
                            ppg_valido_sensor = nuevo_valido
                            ppg_recibido = True
                            ultimo_ppg_rx = ahora


                            if nuevo_dedo == 1:
                                if ppg_dedo_desde is None:
                                    ppg_dedo_desde = ahora
                            else:
                                ppg_dedo_desde = None
                                spo2_actual = np.nan
                                fc_ppg_actual = np.nan
                                ppg_valido_sensor = 0

                            estado_ppg = estado_ppg_ahora(ahora)
                            ppg_lista = ppg_listo_ahora(ahora)

                            registrar_evento(
                                "PPG",
                                numero,
                                timestamp_us,
                                perdidas,
                                PPG_RED_RAW=nuevo_red,
                                PPG_IR_RAW=nuevo_ir,
                                Dedo_PPG=nuevo_dedo,
                                Estado_PPG=estado_ppg,
                                SpO2_pct=(
                                    f"{nuevo_spo2:.1f}"
                                    if ppg_lista and np.isfinite(nuevo_spo2)
                                    else ""
                                ),
                                FC_PPG_BPM=(
                                    f"{nueva_fc_ppg:.1f}"
                                    if ppg_lista and np.isfinite(nueva_fc_ppg)
                                    else ""
                                ),
                                PPG_valido=1 if ppg_lista else 0,
                            )


                    elif tipo == "G" and len(partes) == 5:
                        numero = int(float(partes[1]))
                        timestamp_us = desenvolver_micros(int(float(partes[2])))
                        nuevo_raw = int(float(partes[3]))
                        nuevo_filtrado = convertir_float(partes[4])
                        ahora = time.monotonic()

                        with bloqueo:
                            perdidas = actualizar_perdidas("G", numero) if modo_prueba != "idle" else 0

                            gsr_raw_actual = nuevo_raw
                            gsr_filtrado_actual = nuevo_filtrado
                            gsr_recibido = True
                            ultimo_gsr_rx = ahora
                            ultimo_gsr_timestamp_s = timestamp_us / 1_000_000.0

                            fase = fase_por_timestamp(timestamp_us)
                            if (
                                fase == "BASAL"
                                and np.isfinite(nuevo_filtrado)
                                and 5 < nuevo_raw < 4090
                            ):


                                gsr_muestras_basal.append(
                                    (ultimo_gsr_timestamp_s, nuevo_filtrado)
                                )
                                actualizar_basal_provisional_gsr(
                                    ultimo_gsr_timestamp_s
                                )
                                evaluar_estabilidad_gsr(
                                    ultimo_gsr_timestamp_s
                                )

                            actualizar_derivados_gsr()
                            estado_gsr = estado_gsr_ahora(ahora)

                            registrar_evento(
                                "GSR",
                                numero,
                                timestamp_us,
                                perdidas,
                                GSR_RAW_ADC=nuevo_raw,
                                GSR_FILTRADO=(
                                    f"{nuevo_filtrado:.2f}"
                                    if np.isfinite(nuevo_filtrado)
                                    else ""
                                ),
                                GSR_BASAL=(
                                    f"{gsr_basal:.2f}"
                                    if np.isfinite(gsr_basal)
                                    else ""
                                ),
                                GSR_DELTA_ADC=(
                                    f"{gsr_delta:.2f}" if np.isfinite(gsr_delta) else ""
                                ),
                                GSR_DELTA_PCT=(
                                    f"{gsr_delta_pct:.3f}"
                                    if np.isfinite(gsr_delta_pct)
                                    else ""
                                ),
                                Estado_GSR=estado_gsr,
                                GSR_pendiente_basal_20s_pct_min=(
                                    f"{gsr_pendiente_pct_min:.3f}"
                                    if np.isfinite(gsr_pendiente_pct_min)
                                    else ""
                                ),
                                GSR_pendiente_basal_10s_pct_min=(
                                    f"{gsr_pendiente_corta_pct_min:.3f}"
                                    if np.isfinite(gsr_pendiente_corta_pct_min)
                                    else ""
                                ),
                                GSR_basal_estable=(
                                    1 if gsr_basal_estable else 0
                                ) if np.isfinite(gsr_basal) else "",
                            )


                    elif tipo == "T" and len(partes) == 4:
                        numero = int(float(partes[1]))
                        timestamp_us = desenvolver_micros(int(float(partes[2])))
                        nueva_temp = convertir_float(partes[3])

                        with bloqueo:
                            perdidas = actualizar_perdidas("T", numero) if modo_prueba != "idle" else 0
                            if np.isfinite(nueva_temp):
                                temperatura_actual = nueva_temp

                            registrar_evento(
                                "TEMP",
                                numero,
                                timestamp_us,
                                perdidas,
                                Temperatura_C=(
                                    f"{nueva_temp:.2f}" if np.isfinite(nueva_temp) else ""
                                ),
                            )

                except (ValueError, IndexError):
                    pass

            puerto.close()

        except serial.SerialException as error:
            estado_serial = f"No se pudo abrir {PUERTO}: {error}"
            time.sleep(1)


COLOR_FONDO = "#F4F7FB"
COLOR_PANEL = "#FFFFFF"
COLOR_TEXTO = "#0F172A"
COLOR_SECUNDARIO = "#64748B"
COLOR_BORDE = "#DCE3EC"
AZUL = "#2563EB"
AZUL_OSCURO = "#163A70"
CIAN = "#0891B2"
VERDE = "#059669"
AMBAR = "#D97706"
ROJO = "#DC2626"
VIOLETA = "#7C3AED"
ESMERALDA = "#0F766E"
FONDO_GRAFICA = "#08111F"


def sombra(widget, blur=14, dy=2, alpha=24):
    efecto = QGraphicsDropShadowEffect(widget)
    efecto.setBlurRadius(blur)
    efecto.setOffset(0, dy)
    efecto.setColor(QColor(15, 23, 42, alpha))
    widget.setGraphicsEffect(efecto)


class TarjetaParametro(QFrame):
    def __init__(self, titulo, unidad="", acento=AZUL):
        super().__init__()
        self.acento = acento
        self.setObjectName("tarjeta")
        self.setMinimumHeight(86)
        self.setMaximumHeight(94)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(13, 8, 13, 7)
        layout.setSpacing(1)

        fila_titulo = QHBoxLayout()
        barra = QFrame()
        barra.setFixedSize(4, 17)
        barra.setStyleSheet(f"background: {acento}; border-radius: 2px;")
        fila_titulo.addWidget(barra)

        titulo_label = QLabel(titulo)
        titulo_label.setObjectName("tituloTarjeta")
        fila_titulo.addWidget(titulo_label)
        fila_titulo.addStretch()
        layout.addLayout(fila_titulo)

        fila_valor = QHBoxLayout()
        self.valor = QLabel("--")
        self.valor.setObjectName("valorTarjeta")
        self.valor.setStyleSheet(f"color: {acento};")
        fila_valor.addWidget(self.valor)

        self.unidad_label = QLabel(unidad)
        self.unidad_label.setObjectName("unidadTarjeta")
        self.unidad_label.setAlignment(Qt.AlignBottom)
        fila_valor.addWidget(self.unidad_label)
        fila_valor.addStretch()
        layout.addLayout(fila_valor)

        sombra(self, blur=12, dy=2, alpha=18)

    def cambiar_valor(self, valor, estado="normal"):
        colores = {
            "normal": self.acento,
            "ok": VERDE,
            "warning": AMBAR,
            "error": ROJO,
            "muted": COLOR_SECUNDARIO,
        }
        self.valor.setText(str(valor))
        self.valor.setStyleSheet(f"color: {colores.get(estado, self.acento)};")


class MonitorFisiologico(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Monitor fisiológico - ESP32")
        self.resize(1500, 940)
        self.setMinimumSize(1180, 760)

        central = QWidget()
        central.setObjectName("central")
        self.setCentralWidget(central)

        self.setStyleSheet(
            f"""
            QWidget#central {{ background: {COLOR_FONDO}; }}
            QLabel {{ color: {COLOR_TEXTO}; font-family: "Segoe UI"; }}
            QFrame#panel, QFrame#tarjeta {{
                background: {COLOR_PANEL};
                border: 1px solid {COLOR_BORDE};
                border-radius: 12px;
            }}
            QLabel#tituloPrincipal {{ color: white; font-size: 25px; font-weight: 700; }}
            QLabel#subtituloPrincipal {{ color: #CBD5E1; font-size: 11px; }}
            QLabel#tituloSeccion {{ color: {COLOR_TEXTO}; font-size: 12px; font-weight: 700; }}
            QLabel#tituloTarjeta {{ color: {COLOR_SECUNDARIO}; font-size: 10px; font-weight: 700; }}
            QLabel#valorTarjeta {{ font-size: 22px; font-weight: 700; }}
            QLabel#unidadTarjeta {{ color: {COLOR_SECUNDARIO}; font-size: 9px; padding-bottom: 4px; }}
            QLineEdit, QSpinBox {{
                background: #F8FAFC;
                color: {COLOR_TEXTO};
                border: 1px solid #CBD5E1;
                border-radius: 7px;
                min-height: 30px;
                padding: 0 8px;
                font-size: 10px;
            }}
            QLineEdit:focus, QSpinBox:focus {{ border: 1px solid {AZUL}; background: white; }}
            QPushButton {{
                min-height: 31px;
                border-radius: 7px;
                padding: 0 12px;
                font-size: 10px;
                font-weight: 700;
            }}
            QPushButton#primario {{ background: {AZUL}; color: white; border: none; }}
            QPushButton#primario:hover {{ background: #1D4ED8; }}
            QPushButton#secundario {{ background: #F8FAFC; color: {AZUL_OSCURO}; border: 1px solid #CBD5E1; }}
            QPushButton#secundario:hover {{ background: #EFF6FF; border: 1px solid #93C5FD; }}
            QPushButton#peligro {{ background: #FEF2F2; color: {ROJO}; border: 1px solid #FECACA; }}
            QPushButton#peligro:hover {{ background: #FEE2E2; }}
            QPushButton:disabled {{ background: #F1F5F9; color: #94A3B8; border: 1px solid #E2E8F0; }}
            QProgressBar {{ border: none; background: #E2E8F0; border-radius: 4px; height: 7px; }}
            QProgressBar::chunk {{ background: {AZUL}; border-radius: 4px; }}
            """
        )

        layout_principal = QVBoxLayout(central)
        layout_principal.setContentsMargins(14, 12, 14, 10)
        layout_principal.setSpacing(8)


        encabezado = QFrame()
        encabezado.setStyleSheet(
            f"background: {AZUL_OSCURO}; border: none; border-radius: 14px;"
        )
        encabezado.setFixedHeight(76)
        sombra(encabezado, 18, 3, 30)

        layout_encabezado = QHBoxLayout(encabezado)
        layout_encabezado.setContentsMargins(20, 11, 20, 11)

        bloque_titulo = QVBoxLayout()
        bloque_titulo.setSpacing(1)
        titulo = QLabel("MONITOR FISIOLÓGICO")
        titulo.setObjectName("tituloPrincipal")
        bloque_titulo.addWidget(titulo)
        subtitulo = QLabel(
            "Adquisición biomédica científica · ECG RAW/filtrado · SpO₂ · Temperatura · EDA/GSR"
        )
        subtitulo.setObjectName("subtituloPrincipal")
        bloque_titulo.addWidget(subtitulo)
        layout_encabezado.addLayout(bloque_titulo)
        layout_encabezado.addStretch()

        self.estado = QLabel("ESP32 · buscando dispositivo")
        self.estado.setAlignment(Qt.AlignCenter)
        self.estado.setMinimumWidth(255)
        layout_encabezado.addWidget(self.estado)
        layout_principal.addWidget(encabezado)


        panel_archivo = QFrame()
        panel_archivo.setObjectName("panel")
        panel_archivo.setFixedHeight(56)
        sombra(panel_archivo)
        layout_archivo = QHBoxLayout(panel_archivo)
        layout_archivo.setContentsMargins(14, 7, 14, 7)
        layout_archivo.setSpacing(8)

        titulo_registro = QLabel("REGISTRO")
        titulo_registro.setObjectName("tituloSeccion")
        layout_archivo.addWidget(titulo_registro)
        layout_archivo.addWidget(QLabel("Nombre"))

        self.entrada_nombre = QLineEdit()
        self.entrada_nombre.setPlaceholderText("Ej. participante_01")
        self.entrada_nombre.setMinimumWidth(190)
        self.entrada_nombre.setMaximumWidth(260)
        layout_archivo.addWidget(self.entrada_nombre)

        self.boton_carpeta = QPushButton("ELEGIR CARPETA")
        self.boton_carpeta.setObjectName("secundario")
        self.boton_carpeta.clicked.connect(self.seleccionar_carpeta)
        layout_archivo.addWidget(self.boton_carpeta)

        self.label_carpeta = QLabel(CARPETA_REGISTROS)
        self.label_carpeta.setStyleSheet(f"color: {COLOR_SECUNDARIO}; font-size: 9px;")
        self.label_carpeta.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout_archivo.addWidget(self.label_carpeta, 1)
        layout_principal.addWidget(panel_archivo)


        control = QFrame()
        control.setObjectName("panel")
        control.setFixedHeight(70)
        sombra(control)
        layout_control = QHBoxLayout(control)
        layout_control.setContentsMargins(14, 8, 14, 8)
        layout_control.setSpacing(8)

        titulo_sesion = QLabel("SESIÓN")
        titulo_sesion.setObjectName("tituloSeccion")
        layout_control.addWidget(titulo_sesion)
        layout_control.addWidget(QLabel("Estabilización"))

        self.spin_estabilizacion = QSpinBox()
        self.spin_estabilizacion.setRange(20, 300)
        self.spin_estabilizacion.setValue(ESTABILIZACION_PREDETERMINADA_S)
        self.spin_estabilizacion.setSuffix(" s")
        self.spin_estabilizacion.setFixedWidth(90)
        layout_control.addWidget(self.spin_estabilizacion)

        layout_control.addWidget(QLabel("Duración"))
        self.spin_minutos = QSpinBox()
        self.spin_minutos.setRange(0, 120)
        self.spin_minutos.setValue(1)
        self.spin_minutos.setSuffix(" min")
        self.spin_minutos.setFixedWidth(90)
        layout_control.addWidget(self.spin_minutos)

        self.spin_segundos = QSpinBox()
        self.spin_segundos.setRange(0, 59)
        self.spin_segundos.setValue(0)
        self.spin_segundos.setSuffix(" s")
        self.spin_segundos.setFixedWidth(75)
        layout_control.addWidget(self.spin_segundos)

        self.boton_iniciar = QPushButton("INICIAR")
        self.boton_iniciar.setObjectName("primario")
        self.boton_iniciar.clicked.connect(self.iniciar_prueba)
        layout_control.addWidget(self.boton_iniciar)

        self.boton_terminar = QPushButton("TERMINAR")
        self.boton_terminar.setObjectName("peligro")
        self.boton_terminar.setEnabled(False)
        self.boton_terminar.clicked.connect(self.terminar_prueba)
        layout_control.addWidget(self.boton_terminar)
        layout_control.addStretch()

        bloque_estado = QVBoxLayout()
        bloque_estado.setSpacing(3)
        fila_estado = QHBoxLayout()
        fila_estado.setSpacing(8)

        self.label_fase = QLabel("LISTO")
        self.label_fase.setAlignment(Qt.AlignCenter)
        self.label_fase.setMinimumWidth(165)
        fila_estado.addWidget(self.label_fase)

        self.label_tiempo = QLabel("--:--")
        self.label_tiempo.setStyleSheet(
            f"color: {COLOR_TEXTO}; font-size: 23px; font-weight: 700;"
        )
        self.label_tiempo.setMinimumWidth(74)
        fila_estado.addWidget(self.label_tiempo)
        bloque_estado.addLayout(fila_estado)

        self.progreso = QProgressBar()
        self.progreso.setRange(0, 1000)
        self.progreso.setValue(0)
        self.progreso.setTextVisible(False)
        self.progreso.setFixedWidth(248)
        bloque_estado.addWidget(self.progreso)
        layout_control.addLayout(bloque_estado)
        layout_principal.addWidget(control)


        grid = QGridLayout()
        grid.setSpacing(7)

        self.tarjeta_fc = TarjetaParametro("FC ECG", "BPM", AZUL)
        self.tarjeta_spo2 = TarjetaParametro("SpO₂", "%", CIAN)
        self.tarjeta_fc_ppg = TarjetaParametro("FC PPG", "BPM", CIAN)
        self.tarjeta_temp = TarjetaParametro("TEMPERATURA", "°C", AMBAR)
        self.tarjeta_eda = TarjetaParametro("EDA · Δ BASAL", "%", ESMERALDA)
        self.tarjeta_rr = TarjetaParametro("INTERVALO RR", "ms", AZUL)

        self.tarjeta_eda_basal = TarjetaParametro("EDA BASAL", "ADC", ESMERALDA)
        self.tarjeta_rmssd = TarjetaParametro("HRV · RMSSD FASE", "ms", VIOLETA)
        self.tarjeta_sdnn = TarjetaParametro("HRV · SDNN FASE", "ms", VIOLETA)
        self.tarjeta_estado = TarjetaParametro("ESTADO ECG", "", VERDE)
        self.tarjeta_calidad = TarjetaParametro("CALIDAD ECG", "", VERDE)
        self.tarjeta_rr_validos = TarjetaParametro("RR VÁLIDOS · FASE", "", AZUL)

        fila1 = [
            self.tarjeta_fc,
            self.tarjeta_spo2,
            self.tarjeta_fc_ppg,
            self.tarjeta_temp,
            self.tarjeta_eda,
            self.tarjeta_rr,
        ]
        fila2 = [
            self.tarjeta_eda_basal,
            self.tarjeta_rmssd,
            self.tarjeta_sdnn,
            self.tarjeta_estado,
            self.tarjeta_calidad,
            self.tarjeta_rr_validos,
        ]

        for col, tarjeta in enumerate(fila1):
            grid.addWidget(tarjeta, 0, col)
        for col, tarjeta in enumerate(fila2):
            grid.addWidget(tarjeta, 1, col)
        for col in range(6):
            grid.setColumnStretch(col, 1)

        layout_principal.addLayout(grid)


        panel_ecg = QFrame()
        panel_ecg.setObjectName("panel")
        sombra(panel_ecg)
        layout_ecg = QVBoxLayout(panel_ecg)
        layout_ecg.setContentsMargins(10, 8, 10, 8)
        layout_ecg.setSpacing(5)

        fila_ecg = QHBoxLayout()
        titulo_ecg = QLabel("ECG FILTRADO EN TIEMPO REAL")
        titulo_ecg.setObjectName("tituloSeccion")
        fila_ecg.addWidget(titulo_ecg)
        fila_ecg.addStretch()

        self.info_ecg = QLabel(
            f"{FRECUENCIA_ECG} Hz · ventana {VENTANA_SEGUNDOS} s · RAW se conserva en CSV"
        )
        self.info_ecg.setStyleSheet(f"color: {COLOR_SECUNDARIO}; font-size: 9px;")
        fila_ecg.addWidget(self.info_ecg)
        layout_ecg.addLayout(fila_ecg)

        self.grafica = pg.PlotWidget()
        self.grafica.setMinimumHeight(350)
        self.grafica.setBackground(FONDO_GRAFICA)
        self.grafica.showGrid(x=True, y=True, alpha=0.14)
        self.grafica.setXRange(-VENTANA_SEGUNDOS, 0, padding=0)
        self.grafica.enableAutoRange(axis="x", enable=False)
        self.grafica.enableAutoRange(axis="y", enable=True)
        self.grafica.setMouseEnabled(x=False, y=False)
        self.grafica.setMenuEnabled(False)
        self.grafica.setLabel("bottom", "Tiempo", units="s")
        self.grafica.setLabel("left", "ECG filtrado", units="ADC")

        eje_x = self.grafica.getAxis("bottom")
        eje_y = self.grafica.getAxis("left")
        eje_x.setTextPen(pg.mkPen("#94A3B8"))
        eje_y.setTextPen(pg.mkPen("#94A3B8"))
        eje_x.setPen(pg.mkPen("#475569"))
        eje_y.setPen(pg.mkPen("#475569"))
        eje_y.enableAutoSIPrefix(False)

        self.curva = self.grafica.plot(pen=pg.mkPen("#5EEAD4", width=1.3))
        self.marcadores_r = pg.ScatterPlotItem(
            size=8,
            symbol="o",
            brush=pg.mkBrush("#F59E0B"),
            pen=pg.mkPen("#FDE68A", width=0.7),
        )
        self.grafica.addItem(self.marcadores_r)
        layout_ecg.addWidget(self.grafica, 1)
        layout_principal.addWidget(panel_ecg, 1)


        pie = QFrame()
        pie.setObjectName("panel")
        pie.setFixedHeight(36)
        layout_pie = QHBoxLayout(pie)
        layout_pie.setContentsMargins(12, 4, 12, 4)

        self.informacion = QLabel("Sistema listo")
        self.informacion.setStyleSheet(f"color: {COLOR_SECUNDARIO}; font-size: 9px;")
        layout_pie.addWidget(self.informacion)
        layout_pie.addStretch()

        self.label_csv = QLabel("Registro CSV: esperando prueba")
        self.label_csv.setStyleSheet(f"color: {COLOR_SECUNDARIO}; font-size: 9px;")
        self.label_csv.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout_pie.addWidget(self.label_csv)
        layout_principal.addWidget(pie)

        self._fase_visual("idle")

        self.timer = QTimer()
        self.timer.timeout.connect(self.actualizar)
        self.timer.start(33)

    def formatear(self, valor, decimales=0):
        if not np.isfinite(valor):
            return "--"
        return f"{valor:.{decimales}f}"

    def _fase_visual(self, fase):
        mapa = {
            "idle": ("LISTO", "#F1F5F9", "#CBD5E1", COLOR_SECUNDARIO),
            "adaptation": ("ADAPTACIÓN", "#FFF7ED", "#FED7AA", AMBAR),
            "baseline": ("BASAL", "#F0FDFA", "#99F6E4", ESMERALDA),
            "baseline_wait": ("EDA NO ESTABLE", "#FFF7ED", "#FED7AA", AMBAR),
            "recording": ("PRUEBA EN CURSO", "#ECFDF5", "#A7F3D0", VERDE),
            "finished": ("PRUEBA FINALIZADA", "#EFF6FF", "#BFDBFE", AZUL),
            "cancelled": ("SESIÓN DETENIDA", "#FEF2F2", "#FECACA", ROJO),
        }
        texto, fondo, borde, color = mapa[fase]
        self.label_fase.setText(texto)
        self.label_fase.setStyleSheet(
            f"""
            color: {color};
            background: {fondo};
            border: 1px solid {borde};
            border-radius: 12px;
            padding: 4px 10px;
            font-size: 10px;
            font-weight: 700;
            """
        )

    def _serial_visual(self):
        conectado = "Conectado:" in estado_serial
        if conectado:
            texto = estado_serial.replace("Conectado:", "ESP32 ·")
            fondo = "rgba(5,150,105,0.23)"
            borde = "rgba(167,243,208,0.35)"
            color = "#D1FAE5"
        else:
            texto = estado_serial
            fondo = "rgba(217,119,6,0.22)"
            borde = "rgba(253,230,138,0.30)"
            color = "#FEF3C7"

        self.estado.setText(texto)
        self.estado.setStyleSheet(
            f"""
            color: {color};
            background: {fondo};
            border: 1px solid {borde};
            border-radius: 16px;
            padding: 7px 13px;
            font-size: 10px;
            font-weight: 700;
            """
        )

    def limpiar_valores_nuevo_sujeto(self):
        self.tarjeta_fc.cambiar_valor("--", "muted")
        self.tarjeta_spo2.cambiar_valor("CALCULANDO", "warning")
        self.tarjeta_fc_ppg.cambiar_valor("CALCULANDO", "warning")
        self.tarjeta_temp.cambiar_valor("--", "muted")
        self.tarjeta_eda.cambiar_valor("ADAPTACIÓN", "warning")
        self.tarjeta_eda_basal.cambiar_valor("--", "muted")
        self.tarjeta_rr.cambiar_valor("--", "muted")
        self.tarjeta_rmssd.cambiar_valor("CALCULANDO", "warning")
        self.tarjeta_sdnn.cambiar_valor("CALCULANDO", "warning")
        self.tarjeta_estado.cambiar_valor("INICIANDO", "warning")
        self.tarjeta_calidad.cambiar_valor("INICIANDO", "warning")
        self.tarjeta_rr_validos.cambiar_valor("0", "muted")
        self.curva.clear()
        self.marcadores_r.clear()

    def seleccionar_carpeta(self):
        global CARPETA_REGISTROS
        carpeta = QFileDialog.getExistingDirectory(
            self,
            "Seleccionar carpeta donde guardar las pruebas",
            CARPETA_REGISTROS,
        )
        if carpeta:
            CARPETA_REGISTROS = carpeta
            self.label_carpeta.setText(carpeta)
            self.label_csv.setText(f"Carpeta · {carpeta}")

    def iniciar_prueba(self):
        global modo_prueba, inicio_estabilizacion
        global duracion_estabilizacion_nominal, duracion_estabilizacion_efectiva
        global duracion_prueba_segundos, nombre_base_archivo
        global temperatura_actual, estado_leads_actual
        global ruta_csv_actual

        nombre = limpiar_nombre_archivo(self.entrada_nombre.text())
        if not nombre:
            QMessageBox.warning(self, "Nombre requerido", "Escribe un nombre para la prueba o participante.")
            return

        duracion = self.spin_minutos.value() * 60 + self.spin_segundos.value()
        if duracion <= 0:
            QMessageBox.warning(self, "Duración inválida", "La duración debe ser mayor que 0 segundos.")
            return

        if "Conectado:" not in estado_serial:
            QMessageBox.warning(self, "ESP32", "El ESP32 todavía no está conectado.")
            return

        if not os.path.isdir(CARPETA_REGISTROS):
            QMessageBox.warning(self, "Carpeta inválida", "Selecciona una carpeta válida para guardar la prueba.")
            return

        nombre_base_archivo = nombre
        duracion_prueba_segundos = duracion
        duracion_estabilizacion_nominal = self.spin_estabilizacion.value()
        duracion_estabilizacion_efectiva = duracion_estabilizacion_nominal

        with bloqueo:
            detector_qrs.reiniciar_completo()
            datos_ecg.clear()
            picos_r.clear()
            temperatura_actual = np.nan
            estado_leads_actual = 0
            reiniciar_ppg_python()
            reiniciar_gsr_python()
            reiniciar_control_sesion()
            ruta_csv_actual = ""

            try:
                abrir_registro_csv()
            except OSError as error:
                QMessageBox.critical(self, "CSV", f"No se pudo crear el archivo:\n{error}")
                return

            modo_prueba = "stabilizing"
            inicio_estabilizacion = time.monotonic()

        self.limpiar_valores_nuevo_sujeto()

        for widget in (
            self.entrada_nombre,
            self.boton_carpeta,
            self.spin_estabilizacion,
            self.spin_minutos,
            self.spin_segundos,
            self.boton_iniciar,
        ):
            widget.setEnabled(False)
        self.boton_terminar.setEnabled(True)

        self._fase_visual("adaptation")
        self.label_csv.setText(f"Guardando sesión completa · {ruta_csv_actual}")

    def comenzar_registro(self, forzar_basal=False):
        global modo_prueba, inicio_registro, timestamp_inicio_prueba_esp32_us


        with bloqueo:
            congelar_basal_gsr(forzar=forzar_basal)

            modo_prueba = "recording"
            inicio_registro = time.monotonic()
            timestamp_inicio_prueba_esp32_us = ultimo_timestamp_esp32_us

        self._fase_visual("recording")
        self.label_csv.setText(f"Guardando prueba · {ruta_csv_actual}")
        return True

    def intentar_comenzar_registro(self):


        with bloqueo:
            estable, pendiente, n = evaluar_estabilidad_gsr()
            suficiente = n >= int(FRECUENCIA_GSR * MIN_SEGUNDOS_BASAL_GSR)

        if estable and suficiente:
            return self.comenzar_registro(forzar_basal=False)


        if suficiente:
            return self.comenzar_registro(forzar_basal=True)


        return self.comenzar_registro(forzar_basal=True)

    def terminar_prueba(self, automatico=False):
        global modo_prueba, inicio_estabilizacion, inicio_registro

        if modo_prueba == "idle":
            return

        estado_anterior = modo_prueba
        archivo_guardado = ruta_csv_actual

        with bloqueo:
            cerrar_registro_csv()
            modo_prueba = "idle"

        inicio_estabilizacion = None
        inicio_registro = None

        for widget in (
            self.entrada_nombre,
            self.boton_carpeta,
            self.spin_estabilizacion,
            self.spin_minutos,
            self.spin_segundos,
            self.boton_iniciar,
        ):
            widget.setEnabled(True)
        self.boton_terminar.setEnabled(False)
        self.label_tiempo.setText("00:00")

        if estado_anterior == "recording":
            self.progreso.setValue(1000)
            self._fase_visual("finished")
            self.label_csv.setText(f"Archivo guardado · {archivo_guardado}")
            QMessageBox.information(
                self,
                "Prueba finalizada",
                f"La prueba terminó correctamente.\n\nArchivo guardado en:\n{archivo_guardado}",
            )
        else:
            self.progreso.setValue(0)
            self._fase_visual("cancelled")
            self.label_csv.setText(f"Sesión parcial guardada · {archivo_guardado}")
            QMessageBox.information(
                self,
                "Sesión detenida",
                (
                    "La sesión se detuvo durante adaptación/basal.\n\n"
                    "Los datos adquiridos hasta ese momento se conservaron en:\n"
                    f"{archivo_guardado}"
                ),
            )

    def actualizar_cronometro(self):
        if modo_prueba == "stabilizing":
            if inicio_estabilizacion is None:
                return

            transcurrido = time.monotonic() - inicio_estabilizacion
            restante = duracion_estabilizacion_efectiva - transcurrido
            fraccion = np.clip(
                transcurrido / max(duracion_estabilizacion_efectiva, 1),
                0,
                1,
            )
            self.progreso.setValue(int(fraccion * 1000))

            fase = fase_actual()
            if fase == "ADAPTACION":
                self._fase_visual("adaptation")
            else:
                with bloqueo:
                    estable, _, n = evaluar_estabilidad_gsr()


                self._fase_visual("baseline")

            if restante <= 0:
                self.intentar_comenzar_registro()
                return

            restante = int(np.ceil(restante))
            self.label_tiempo.setText(f"{restante // 60:02d}:{restante % 60:02d}")

        elif modo_prueba == "recording":
            if inicio_registro is None:
                return

            transcurrido = time.monotonic() - inicio_registro
            restante = duracion_prueba_segundos - transcurrido
            fraccion = np.clip(transcurrido / duracion_prueba_segundos, 0, 1)
            self.progreso.setValue(int(fraccion * 1000))

            if restante <= 0:
                self.terminar_prueba(automatico=True)
                return

            restante = int(np.ceil(restante))
            self.label_tiempo.setText(f"{restante // 60:02d}:{restante % 60:02d}")

    def actualizar(self):
        self._serial_visual()
        self.actualizar_cronometro()

        with bloqueo:
            ecg = list(datos_ecg)
            picos = list(picos_r)
            temperatura = temperatura_actual
            estado_leads = estado_leads_actual
            bpm = detector_qrs.bpm
            rr = detector_qrs.rr_estable_ms
            estado_ppg = estado_ppg_ahora()
            ppg_lista = ppg_listo_ahora()
            spo2 = spo2_actual
            fc_ppg = fc_ppg_actual
            estado_gsr = estado_gsr_ahora()
            basal_gsr = basal_gsr_para_mostrar()
            delta_gsr_pct = gsr_delta_pct
            pendiente_gsr = gsr_pendiente_pct_min
            pendiente_gsr_corta = gsr_pendiente_corta_pct_min
            fase = fase_actual()

            rmssd_fase, sdnn_fase, rr_fase = detector_qrs.metricas_fase(fase)
            rr_basal = len(detector_qrs.intervalos_rr_basal)
            rr_prueba = len(detector_qrs.intervalos_rr_prueba)

            indice_actual = ultimo_numero_ecg
            perdidas = perdidas_tipo.copy()


        if np.isfinite(temperatura):
            self.tarjeta_temp.cambiar_valor(self.formatear(temperatura, 2), "normal")
        else:
            self.tarjeta_temp.cambiar_valor("--", "muted")


        if np.isfinite(basal_gsr):
            estado_basal = "ok" if (modo_prueba == "recording" and gsr_basal_estable) else "warning"
            self.tarjeta_eda_basal.cambiar_valor(self.formatear(basal_gsr, 0), estado_basal)
        else:
            self.tarjeta_eda_basal.cambiar_valor("--", "muted")

        if estado_gsr == "ADAPTANDO":
            self.tarjeta_eda.cambiar_valor("ADAPTACIÓN", "warning")
        elif estado_gsr in ("CALCULANDO BASAL", "BASAL DERIVANDO"):
            self.tarjeta_eda.cambiar_valor("CALCULANDO", "warning")
        elif estado_gsr == "BASAL ESTABLE":
            self.tarjeta_eda.cambiar_valor("ESTABLE", "ok")
        elif estado_gsr == "SIN DATOS":
            self.tarjeta_eda.cambiar_valor("SIN DATOS", "error")
        elif estado_gsr == "SATURADO":
            self.tarjeta_eda.cambiar_valor("SATURADO", "error")
        elif estado_gsr == "SIN BASAL":
            self.tarjeta_eda.cambiar_valor("SIN BASAL", "warning")
        elif estado_gsr == "BASAL INESTABLE":
            if np.isfinite(delta_gsr_pct):
                self.tarjeta_eda.cambiar_valor(f"{delta_gsr_pct:+.1f}", "warning")
            else:
                self.tarjeta_eda.cambiar_valor("INESTABLE", "warning")
        elif np.isfinite(delta_gsr_pct):
            self.tarjeta_eda.cambiar_valor(f"{delta_gsr_pct:+.1f}", "ok")
        else:
            self.tarjeta_eda.cambiar_valor("--", "muted")


        if estado_ppg == "SIN DEDO":
            self.tarjeta_spo2.cambiar_valor("SIN DEDO", "warning")
            self.tarjeta_fc_ppg.cambiar_valor("--", "muted")
        elif estado_ppg == "SIN FLUJO":
            self.tarjeta_spo2.cambiar_valor("SIN FLUJO", "error")
            self.tarjeta_fc_ppg.cambiar_valor("--", "muted")
        elif not ppg_lista:
            self.tarjeta_spo2.cambiar_valor("CALCULANDO", "warning")
            self.tarjeta_fc_ppg.cambiar_valor("CALCULANDO", "warning")
        else:
            self.tarjeta_spo2.cambiar_valor(self.formatear(spo2, 1), "ok")
            self.tarjeta_fc_ppg.cambiar_valor(self.formatear(fc_ppg, 1), "ok")


        calidad = evaluar_calidad(ecg, estado_leads, indice_actual)
        if calidad == "BUENA":
            self.tarjeta_estado.cambiar_valor("ESTABLE", "ok")
            self.tarjeta_calidad.cambiar_valor("BUENA", "ok")
        elif calidad == "INESTABLE":
            self.tarjeta_estado.cambiar_valor("INESTABLE", "warning")
            self.tarjeta_calidad.cambiar_valor("INESTABLE", "warning")
        elif calidad == "SEÑAL BAJA":
            self.tarjeta_estado.cambiar_valor("SEÑAL BAJA", "error")
            self.tarjeta_calidad.cambiar_valor("SEÑAL BAJA", "error")
        else:
            self.tarjeta_estado.cambiar_valor("INICIANDO", "warning")
            self.tarjeta_calidad.cambiar_valor("INICIANDO", "warning")


        if estado_leads == 1 and np.isfinite(bpm):
            self.tarjeta_fc.cambiar_valor(self.formatear(bpm, 0), "normal")
        else:
            self.tarjeta_fc.cambiar_valor("--", "muted")

        if estado_leads == 1 and np.isfinite(rr):
            self.tarjeta_rr.cambiar_valor(self.formatear(rr, 0), "normal")
        else:
            self.tarjeta_rr.cambiar_valor("--", "muted")


        if fase in ("BASAL", "PRUEBA"):
            if np.isfinite(rmssd_fase) and np.isfinite(sdnn_fase):
                self.tarjeta_rmssd.cambiar_valor(self.formatear(rmssd_fase, 1), "normal")
                self.tarjeta_sdnn.cambiar_valor(self.formatear(sdnn_fase, 1), "normal")
            else:
                self.tarjeta_rmssd.cambiar_valor("CALCULANDO", "warning")
                self.tarjeta_sdnn.cambiar_valor("CALCULANDO", "warning")
            self.tarjeta_rr_validos.cambiar_valor(
                str(rr_fase),
                "ok" if rr_fase >= RR_MINIMOS_HRV else "normal",
            )
        else:
            self.tarjeta_rmssd.cambiar_valor("CALCULANDO", "warning")
            self.tarjeta_sdnn.cambiar_valor("CALCULANDO", "warning")
            self.tarjeta_rr_validos.cambiar_valor("0", "muted")


        pendiente_20_txt = (
            f"{pendiente_gsr:+.2f}%/min" if np.isfinite(pendiente_gsr) else "--"
        )
        pendiente_10_txt = (
            f"{pendiente_gsr_corta:+.2f}%/min"
            if np.isfinite(pendiente_gsr_corta)
            else "--"
        )
        self.informacion.setText(
            f"Fase: {fase} · HRV basal: {rr_basal} RR · HRV prueba: {rr_prueba} RR · "
            f"SpO₂: {estado_ppg} · EDA: {estado_gsr} · "
            f"pendiente EDA 10s/20s: {pendiente_10_txt}/{pendiente_20_txt} · "
            f"Pérdidas E/P/G/T: {perdidas['E']}/{perdidas['P']}/{perdidas['G']}/{perdidas['T']}"
        )


        if len(ecg) < 2:
            return

        indices = np.asarray([indice for indice, _ in ecg], dtype=float)
        valores = np.asarray([valor for _, valor in ecg], dtype=float)
        ultimo_indice = indices[-1]
        x = (indices - ultimo_indice) / FRECUENCIA_ECG
        valores_plot = valores - CENTRO_ECG
        self.curva.setData(x, valores_plot)

        primer_indice_visible = ultimo_indice - MAX_MUESTRAS_ECG
        indices_r = []
        valores_r = []

        for indice_pico, valor_pico in picos:
            if indice_pico >= primer_indice_visible:
                indices_r.append(indice_pico)
                valores_r.append(valor_pico - CENTRO_ECG)

        if indices_r:
            indices_r = np.asarray(indices_r, dtype=float)
            x_r = (indices_r - ultimo_indice) / FRECUENCIA_ECG
            self.marcadores_r.setData(x=x_r, y=valores_r)
        else:
            self.marcadores_r.clear()

    def closeEvent(self, event):
        global ejecutando, modo_prueba

        with bloqueo:
            cerrar_registro_csv()
            modo_prueba = "idle"

        ejecutando = False
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    hilo_serial = threading.Thread(target=leer_serial, daemon=True)
    hilo_serial.start()

    ventana = MonitorFisiologico()
    ventana.show()

    sys.exit(app.exec())
