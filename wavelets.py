import numpy as np
import scipy.io
from scipy import signal
import matplotlib.pyplot as plt
import pywt  # Añadimos la librería para wavelets

# 1. Cargar el archivo .set
set_file = 'eeg+eye+state/EEG_freeviewing_25channels.set'
metadata = scipy.io.loadmat(set_file, struct_as_record=False, squeeze_me=True)
EEG = metadata['EEG']
print("Metadata cargada:", EEG)

fs = EEG.srate
num_channels = EEG.nbchan
num_samples = EEG.pnts
num_trials = EEG.trials
channel_labels = [ch.labels for ch in EEG.chanlocs]

# 2. Leer el archivo .fdt
fdt_file = 'eeg+eye+state/EEG_freeviewing_25channels.fdt'
with open(fdt_file, 'rb') as f:
    if num_trials > 1:
        data = np.fromfile(f, dtype=np.float32).reshape((num_samples, num_channels, num_trials), order='F')
    else:
        data = np.fromfile(f, dtype=np.float32).reshape((num_samples, num_channels), order='F')

print("Dimensiones de los datos:", data.shape)

# 3. Preprocesamiento
window_size = int(4 * fs)  # 4 segundos de ventana
num_windows = data.shape[0] // window_size if num_trials == 1 else data.shape[2] // window_size

# Filtro de paso alto (mantenemos el mismo)
highpass_filter = signal.firwin(65, 0.5, fs=fs, pass_zero=False)

# 4. Configuración de Wavelet
wavelet = 'db4'  # Wavelet Daubechies 4, comúnmente usada en EEG
level = 4  # Nivel de descomposición (ajusta según fs y necesidades)

# 5. Extraer características con Wavelets
feature_vectors = []
labels = []

for i in range(num_windows):
    if num_trials == 1:
        start = i * window_size
        end = start + window_size
        window = data[start:end, :]
    else:
        window = data[:, :, i]

    window_features = []
    for ch in range(num_channels):
        channel_data = window[:, ch]
        # Aplicar filtro de paso alto
        channel_data = signal.lfilter(highpass_filter, 1.0, channel_data)

        # Calcular transformada wavelet discreta
        coeffs = pywt.wavedec(channel_data, wavelet, level=level)

        # Extraer características de los coeficientes
        for coeff in coeffs:
            # Por ejemplo, energía de los coeficientes
            energy = np.sum(coeff ** 2)
            # Otras posibles características
            mean = np.mean(np.abs(coeff))
            std = np.std(coeff)
            window_features.extend([energy, mean, std])  # Añadimos energía, media y desv. estándar

    feature_vectors.append(window_features)

X_features = np.array(feature_vectors)

print("Dimensiones de X_features:", X_features.shape)
print("Ejemplo de feature vector:", X_features[0])

# 6. Visualización (mantenemos la gráfica original, pero opcionalmente puedes graficar coeficientes wavelet)
t = np.linspace(0, 4, window_size)
plt.figure(figsize=(12, 6))
# Ejemplo: graficar la señal original y los coeficientes wavelet para el canal 0
channel_data = signal.lfilter(highpass_filter, 1.0, window[:, 0])
coeffs = pywt.wavedec(channel_data, wavelet, level=level)

# Graficar señal original
plt.subplot(level + 2, 1, 1)
plt.plot(t, channel_data, label='Señal Original')
plt.title(f"Señal EEG (Canal {channel_labels[0]}, Ventana 0, 4s)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud (µV)")
plt.grid(True)

# Graficar coeficientes wavelet
for l in range(level + 1):
    plt.subplot(level + 2, 1, l + 2)
    plt.plot(coeffs[l], label=f'Nivel {l}' if l == 0 else f'Detalle {l}')
    plt.title(f"Coeficientes Wavelet - Nivel {l}" if l == 0 else f"Detalle {l}")
    plt.xlabel("Índice")
    plt.ylabel("Amplitud")
    plt.grid(True)

plt.tight_layout()
plt.savefig('SEGUNDO_TEST_WAVELET.png', dpi=300, bbox_inches='tight')
print("Gráfico exportado como 'SEGUNDO_TEST_WAVELET.png'")
plt.show()
