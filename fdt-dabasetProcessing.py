import numpy as np
import scipy.io
from scipy import signal
import matplotlib.pyplot as plt

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

# 3. Preprocesamiento (similar al código anterior)
window_size = int(4 * fs)  # 4 segundos de ventana
num_windows = data.shape[0] // window_size if num_trials == 1 else data.shape[2] // window_size

bands = {
    'delta': (0.5, 3),
    'theta': (3.5, 7.5),
    'alpha': (7.5, 13),
    'beta': (14, 30)
}

# Filtros FIR (ajusta el orden según necesites, aquí usamos 65 como en tu código)
highpass_filter = signal.firwin(65, 0.5, fs=fs, pass_zero=False)
bandpass_filters = {band: signal.firwin(65, [low, high], fs=fs, pass_zero=False) for band, (low, high) in bands.items()}

# 4. Extraer características
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
        channel_data = signal.lfilter(highpass_filter, 1.0, channel_data)
        for band in bands:
            filtered = signal.lfilter(bandpass_filters[band], 1.0, channel_data)
            energy = np.sum(filtered**2)
            window_features.append(energy)

    feature_vectors.append(window_features)

X_features = np.array(feature_vectors)

print("Dimensiones de X_features:", X_features.shape)
print("Ejemplo de feature vector:", X_features[0])


t = np.linspace(0, 4, window_size)
plt.figure(figsize=(12, 6))
filtered_signals = {}
for band in bands:
    filtered_signals[band] = signal.lfilter(bandpass_filters[band], 1.0, window[:, 0])  # Canal 0 como ejemplo
    plt.plot(t, filtered_signals[band], label=band.capitalize(), linewidth=1.5)

plt.title(f"Señales EEG Filtradas (Canal {channel_labels[0]}, Ventana 0, 4s)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud (µV)")
plt.legend()
plt.grid(True)
plt.savefig('SEGUNDO TEST.png', dpi=300, bbox_inches='tight')
print("Gráfico exportado como 'SEGUNDO TEST.png'")
plt.show()