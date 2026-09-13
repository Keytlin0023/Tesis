  if (
    contadorEstabilizacionECG
    <
    MUESTRAS_ESTABILIZACION_ECG
  ) {

    contadorEstabilizacionECG++;


    enviarDatosECG(
      contadorECG,
      timestampUs,
      ecgRaw,
      2048,
      1
    );


    contadorECG++;


    return;
  }


  float visual =
    CENTRO_GRAFICA
    +
    ecg
    *
    GANANCIA_VISUAL;


  int ecgFiltrado =
    constrain(
      (int)
      lroundf(
        visual
      ),
      0,
      4095
    );


  enviarDatosECG(
    contadorECG,
    timestampUs,
    ecgRaw,
    ecgFiltrado,
    1
  );


  contadorECG++;
}


void setup() {


  Serial.begin(
    BAUDIOS
  );


  analogReadResolution(
    12
  );


  pinMode(
    PIN_EKG,
    INPUT
  );


  pinMode(
    PIN_LO_PLUS,
    INPUT
  );


  pinMode(
    PIN_LO_MINUS,
    INPUT
  );


  analogSetPinAttenuation(
    PIN_EKG,
    ADC_11db
  );


  pinMode(
    PIN_GSR,
    INPUT
  );


  analogSetPinAttenuation(
    PIN_GSR,
    ADC_11db
  );


  filtroGsrInicializado =
    false;


  sensorTemperatura.begin();


  sensorTemperatura.setResolution(
    12
  );


  sensorTemperatura.setWaitForConversion(
    false
  );


  sensorTemperatura.requestTemperatures();


  inicioConversionTemp =
    millis();


  ultimaSolicitudTemp =
    millis();


  conversionTemperatura =
    true;


  Wire.begin(
    PIN_SDA,
    PIN_SCL
  );


  Wire.setClock(
    400000
  );


  sensorSpO2Disponible =
    sensorSpO2.begin(
      Wire,
      I2C_SPEED_FAST
    );


  if (
    sensorSpO2Disponible
  ) {

    sensorSpO2.setup(
      SPO2_BRILLO_LED,
      SPO2_PROMEDIO_MUESTRAS,
      SPO2_MODO_LED,
      SPO2_SAMPLE_RATE,
      SPO2_PULSE_WIDTH,
      SPO2_ADC_RANGE
    );


    sensorSpO2.setPulseAmplitudeGreen(
      0
    );


    sensorSpO2.clearFIFO();


    limpiarEstadoPPG();
  }


  const float Q =
    0.70710678f;


  filtroHP.configurarPasaAltas(
    FS_ECG,
    0.5f,
    Q
  );


  filtroNotch.configurarNotch(
    FS_ECG,
    60.0f,
    25.0f
  );


  filtroLP.configurarPasaBajas(
    FS_ECG,
    35.0f,
    Q
  );


  reiniciarECG();


  contadorECG = 0;

  contadorPPG = 0;

  contadorGSR = 0;

  contadorTEMP = 0;


  uint32_t ahoraUs =
    micros();


  siguienteMuestraECG =
    ahoraUs;


  siguienteMuestraGSR =
    ahoraUs;
}


void loop() {


  uint32_t ahoraUs =
    micros();


  while (
    (int32_t)(
      ahoraUs
      -
      siguienteMuestraECG
    )
    >=
    0
  ) {

    siguienteMuestraECG +=
      PERIODO_ECG_US;


    procesarMuestraECG();


    ahoraUs =
      micros();
  }
  actualizarSpO2();
  actualizarGSR();
  actualizarTemperatura();
}
