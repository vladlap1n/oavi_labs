import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks


def analyze_sound(file_path, sound_name):
    y, sr = librosa.load(file_path, sr=None)

    plt.figure(figsize=(10, 6))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y, window='hann')), ref=np.max)
    librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Спектрограмма звука "{sound_name}"')
    plt.savefig(f'{sound_name}_spectrogram.png')
    plt.close()

    fft = np.abs(librosa.stft(y))
    frequencies = librosa.fft_frequencies(sr=sr)
    magnitude = np.mean(fft, axis=1)

    threshold = 0.1 * np.max(magnitude)
    significant_freqs = frequencies[magnitude > threshold]
    min_freq = np.min(significant_freqs) if len(significant_freqs) > 0 else 0
    max_freq = np.max(significant_freqs) if len(significant_freqs) > 0 else 0

    spectral_flatness = librosa.feature.spectral_flatness(y=y)
    frame_index = np.argmin(spectral_flatness)
    frame = fft[:, frame_index]

    peaks, _ = find_peaks(frame, height=0.1 * np.max(frame))
    peak_freqs = frequencies[peaks]
    peak_mags = frame[peaks]

    if len(peak_freqs) > 0:
        fundamental_idx = np.argmax(peak_mags)
        fundamental_freq = peak_freqs[fundamental_idx]
    else:
        fundamental_freq = 0

    mean_magnitude = np.mean(fft, axis=1)
    formant_peaks, _ = find_peaks(mean_magnitude, distance=5)  # distance=5 соответствует ~50 Гц при sr=22050
    formant_peaks = formant_peaks[np.argsort(mean_magnitude[formant_peaks])[-3:]]  # Три самых высоких пика
    formant_freqs = frequencies[formant_peaks]

    return {
        'min_freq': min_freq,
        'max_freq': max_freq,
        'fundamental_freq': fundamental_freq,
        'formants': sorted(formant_freqs.tolist())
    }


sounds = {
    'А': '1.wav',
    'И': '2.wav',
    'лай': '3.wav'
}

results = {}
for name, file in sounds.items():
    results[name] = analyze_sound(file, name)

with open('lab_10_2.md', 'w', encoding='utf-8') as f:
    f.write('# Лабораторная работа №10. Обработка голоса\n\n')
    for name, data in results.items():
        f.write(f'## Звук "{name}"\n')
        f.write(f'- Минимальная частота: {data["min_freq"]:.2f} Гц\n')
        f.write(f'- Максимальная частота: {data["max_freq"]:.2f} Гц\n')
        f.write(f'- Основной тон с наибольшим количеством обертонов: {data["fundamental_freq"]:.2f} Гц\n')
        f.write(f'- Три самые сильные форманты: {", ".join(f"{f:.2f}" for f in data["formants"])} Гц\n\n')
        f.write(f'![Спектрограмма]({name}_spectrogram.png)\n\n')

print("Анализ завершен. Результаты сохранены в lab_10_2.md")