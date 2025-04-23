from scipy.io import arff
from scipy import signal
import pandas as pd
import numpy as np
import spkit as sp
import matplotlib.pyplot as plt

file_path = 'eeg+eye+state/EEG Eye State.arff'
data, meta = arff.loadarff(file_path)
df = pd.DataFrame(data)

df['eyeDetection'] = df['eyeDetection'].apply(lambda x: int(x.decode('utf-8')) if isinstance(x, bytes) else int(x))

print("Dimensiones del DataFrame:", df.shape)
print("Distribución de eyeDetection:", df['eyeDetection'].value_counts())
print("Estadísticas de los canales EEG:")
print(df.iloc[:, :-1].describe())

# Separar características (14 canales EEG) y etiquetas
X = df.iloc[:, :-1].values
y = df['eyeDetection'].values

# Parámetros
fs = 128  # Frecuencia de muestreo (Hz)
window_size = 1 * fs  # 4 segundos * 128 Hz = 512 muestras
num_channels = X.shape[1]  # 14 canales
bands = {
    'delta': (0.5, 3),
    'theta': (3.5, 7.5),
    'alpha': (7.5, 13),
    'beta': (14, 30)
}

# 2. Preprocesamiento: Remoción de artefactos
X_clean = np.zeros_like(X)
for ch in range(num_channels):
    X_clean[:, ch] = sp.eeg.ATAR_1Ch(X[:, ch], verbose=False)

# 3. Diseñar filtros FIR
nyquist = fs / 2
highpass_filter = signal.firwin(65, 0.5, fs=fs, pass_zero='highpass')  # High-pass 0.5 Hz
bandpass_filters = {
    band: signal.firwin(65, [low / nyquist, high / nyquist], fs=fs, pass_zero='bandpass')
    for band, (low, high) in bands.items()
}

# 4. Visualizar las ondas fuente para 4 canales diferentes (AF3, O1, F7, T8)
channels_to_plot = [0, 6, 1, 9]  # Índices de los canales: AF3 (0), O1 (6), F7 (1), T8 (9)
channel_names = [df.columns[i] for i in channels_to_plot]  # Nombres de los canales
window_idx = 10  # Primera ventana de 4 segundos
start = window_idx * window_size
end = start + window_size

# Colores para las bandas
colors = {
    'delta': '#1f77b4',  # Azul
    'theta': '#2ca02c',  # Verde
    'alpha': '#d62728',  # Rojo
    'beta': '#9467bd'  # Morado
}

# Preparar señales para todos los canales seleccionados
t = np.arange(window_size) / fs  # Vector de tiempo (0 a 4 segundos)
filtered_signals_by_channel = {}
for ch_idx in channels_to_plot:
    window = X_clean[start:end, ch_idx]
    filtered_signals = {}
    window = signal.lfilter(highpass_filter, 1.0, window)  # Aplicar high-pass
    for band in bands:
        filtered_signals[band] = signal.lfilter(bandpass_filters[band], 1.0, window)
    filtered_signals_by_channel[ch_idx] = filtered_signals


for ch_idx, ch_name in zip(channels_to_plot, channel_names):
    filtered_signals = filtered_signals_by_channel[ch_idx]

    plt.figure(figsize=(12, 6))
    for band, signal_data in filtered_signals.items():
        plt.plot(t, signal_data, label=band.capitalize(), color=colors[band], linewidth=1.5, alpha=0.8)
    plt.title(f"Señales EEG Filtradas (Canal {ch_name}, Ventana {window_idx + 1}, 4s)", fontsize=14)
    plt.xlabel("Tiempo (s)", fontsize=12)
    plt.ylabel("Amplitud (µV)", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().set_facecolor('#f5f5f5')

    filename = f'eeg_waves_{ch_name}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Gráfico exportado como '{filename}'")
    plt.show()

# 4.2. Generar un solo gráfico con 4 subplots
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
for ax, ch_idx, ch_name in zip(axes, channels_to_plot, channel_names):
    filtered_signals = filtered_signals_by_channel[ch_idx]
    for band, signal_data in filtered_signals.items():
        ax.plot(t, signal_data, label=band.capitalize(), color=colors[band], linewidth=1.5, alpha=0.8)
    ax.set_title(f"Canal {ch_name}", fontsize=12, pad=10)
    ax.set_ylabel("Amplitud (µV)", fontsize=10)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_facecolor('#f5f5f5')
axes[-1].set_xlabel("Tiempo (s)", fontsize=12)
plt.suptitle(f"Señales EEG Filtradas (Ventana {window_idx + 1}, 4s)", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig('eeg_waves_all_channels.png', dpi=300, bbox_inches='tight')
print("Gráfico con subplots exportado como 'eeg_waves_all_channels.png'")
plt.show()

# 5. Segmentar datos en ventanas de 4 segundos y extraer energías
feature_vectors = []
labels = []
num_windows = len(X_clean) // window_size

for i in range(num_windows):
    start = i * window_size
    end = start + window_size
    window = X_clean[start:end, :]  # Shape: (512, 14)

    # Extraer energía por banda y canal
    window_features = []
    for ch in range(num_channels):
        channel_data = window[:, ch]
        channel_data = signal.lfilter(highpass_filter, 1.0, channel_data)
        for band in bands:
            filtered = signal.lfilter(bandpass_filters[band], 1.0, channel_data)
            energy = np.sum(filtered ** 2)
            window_features.append(energy)

    feature_vectors.append(window_features)
    labels.append(y[end - 1])

# Convertir a arrays
X_features = np.array(feature_vectors)
y_labels = np.array(labels)

print("Dimensiones de X_features:", X_features.shape)
print("Dimensiones de y_labels:", y_labels.shape)

print("Ejemplo de feature vector:", X_features[1])