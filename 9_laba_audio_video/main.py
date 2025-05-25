import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from scipy.signal import savgol_filter, wiener
import soundfile as sf
import os


class AudioLabAnalyzer:
    def __init__(self, filename):
        self.filename = filename
        self.fs, self.audio = wavfile.read(filename)

        if len(self.audio.shape) > 1:
            self.audio = self.audio.mean(axis=1)

        self.audio = self.audio / np.max(np.abs(self.audio))
        self.duration = len(self.audio) / self.fs
        self.noise_level = None
        self.filtered_audio = {
            'savgol': None,
            'wiener': None,
            'lowpass': None
        }
        self.energy_results = {
            'max_times': [],
            'energy_plot': 'energy_analysis.png'
        }

    def analyze(self):
        self._plot_spectrogram(self.audio, 'original_spectrogram.png')
        self._estimate_noise()
        self._apply_filters()
        self._analyze_energy()
        self._save_filtered_audio()
        self._generate_report()

    def _plot_spectrogram(self, audio, save_to, title=None):
        plt.figure(figsize=(12, 8))
        f, t, Sxx = signal.spectrogram(audio, self.fs, window='hann',
                                       nperseg=1024, noverlap=512)
        plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
        plt.yscale('log')
        plt.ylabel('Частота [Гц] (лог. шкала)')
        plt.xlabel('Время [сек]')
        if title:
            plt.title(title)
        plt.colorbar(label='Интенсивность [дБ]')
        plt.savefig(save_to)
        plt.close()

    def _estimate_noise(self):
        noise_est = self.audio[:int(0.1 * self.fs)]
        self.noise_level = np.std(noise_est)

    def _apply_filters(self):

        self.filtered_audio['savgol'] = savgol_filter(self.audio, 101, 3)
        self._plot_spectrogram(self.filtered_audio['savgol'],
                               'savgol_spectrogram.png',
                               'После фильтра Савицкого-Голея')

        self.filtered_audio['wiener'] = wiener(self.audio)
        self._plot_spectrogram(self.filtered_audio['wiener'],
                               'wiener_spectrogram.png',
                               'После фильтра Винера')

        b, a = signal.butter(4, 1000 / (self.fs / 2), 'low')
        self.filtered_audio['lowpass'] = signal.filtfilt(b, a, self.audio)
        self._plot_spectrogram(self.filtered_audio['lowpass'],
                               'lowpass_spectrogram.png',
                               'После НЧ-фильтра')

    def _analyze_energy(self, delta_t=0.1, freq_range=(40, 50)):
        samples_per_segment = int(delta_t * self.fs)
        num_segments = len(self.audio) // samples_per_segment

        energy = []
        for i in range(num_segments):
            segment = self.audio[i * samples_per_segment: (i + 1) * samples_per_segment]
            energy.append(np.sum(segment ** 2))

        self.energy_results['max_times'] = [
            i * delta_t for i in np.argsort(energy)[-3:][::-1]
        ]

        f, t, Sxx = signal.spectrogram(self.audio, self.fs, window='hann', nperseg=1024)
        freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])
        energy_in_range = np.sum(Sxx[freq_mask, :], axis=0)

        plt.figure(figsize=(12, 6))
        plt.plot(t, energy_in_range)
        plt.xlabel('Время [сек]')
        plt.ylabel(f'Энергия в диапазоне {freq_range[0]}-{freq_range[1]} Гц')
        plt.title('Энергия сигнала в заданном частотном диапазоне')
        plt.savefig(self.energy_results['energy_plot'])
        plt.close()

    def _save_filtered_audio(self):
        """Сохранение отфильтрованных версий"""
        for name, audio in self.filtered_audio.items():
            sf.write(f"audio_{name}.wav", audio, self.fs)

    def _generate_report(self):
        """Генерация отчета"""
        report = f"""# Лабораторная работа №9. Анализ шума

## 1. Исходные данные
- Анализируемый файл: `{os.path.basename(self.filename)}`
- Частота дискретизации: {self.fs} Гц
- Длительность: {self.duration:.2f} сек
- Уровень шума: {self.noise_level:.4f}

## 2. Спектральный анализ

### Оригинальный сигнал
![Спектрограмма](original_spectrogram.png)

### Результаты фильтрации
| Фильтр | Спектрограмма |
|--------|---------------|
| Савицкого-Голея | ![Спектрограмма](savgol_spectrogram.png) |
| Винера | ![Спектрограмма](wiener_spectrogram.png) |
| Низких частот | ![Спектрограмма](lowpass_spectrogram.png) |

## 3. Анализ энергии
- Шаг анализа: {0.1} сек
- Анализируемый частотный диапазон: 40-50 Гц
- Моменты с максимальной энергией: {self.energy_results['max_times']} сек

![График энергии]({self.energy_results['energy_plot']})

## 4. Выводы
В ходе работы были проанализированы три метода фильтрации:
1. **Фильтр Савицкого-Голея** - хорошо сохраняет форму сигнала
2. **Фильтр Винера** - эффективен против аддитивного шума
3. **НЧ-фильтр** - удаляет высокочастотные помехи

"""

        with open("lab_9.md", "w", encoding="utf-8") as f:
            f.write(report)

        print("Анализ завершен. Отчет сохранен в lab_9.md")


# Запуск анализа
if __name__ == "__main__":
    analyzer = AudioLabAnalyzer("sound.wav")
    analyzer.analyze()