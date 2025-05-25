import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime


# =============================================
# 0. Создание папки для результатов
# =============================================
def create_results_folder():
    if not os.path.exists("lab8_results"):
        os.makedirs("lab8_results")
    return "lab8_results"


# =============================================
# 1. Загрузка изображения и перевод в HSL
# =============================================
def load_and_convert_to_hsl(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Изображение {image_path} не найдено!")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_hsl = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    H, L, S = cv2.split(image_hsl)
    return image_rgb, H, L, S


# =============================================
# 2. Гамма-коррекция
# =============================================
def gamma_correction(channel, gamma=1.5):
    corrected = np.power(channel / 255.0, gamma) * 255.0
    return corrected.astype(np.uint8)


# =============================================
# 3. Построение гистограммы
# =============================================
def plot_histogram(data, title, filename, xlabel="Яркость", ylabel="Частота"):
    plt.figure()
    plt.hist(data.ravel(), bins=256, range=(0, 256), color='blue', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.close()


# =============================================
# 4. Ручная реализация LBP
# =============================================
def manual_lbp(image, radius=1, neighbors=8):
    height, width = image.shape
    lbp = np.zeros_like(image, dtype=np.uint8)

    for y in range(radius, height - radius):
        for x in range(radius, width - radius):
            center = image[y, x]
            binary_code = 0
            for n in range(neighbors):
                angle = 2 * np.pi * n / neighbors
                xn = int(x + radius * np.cos(angle))
                yn = int(y - radius * np.sin(angle))
                pixel = image[yn, xn]
                binary_code |= (1 << (neighbors - 1 - n)) if pixel >= center else 0
            lbp[y, x] = binary_code
    return lbp


# Улучшенная версия LBP (с использованием униформных шаблонов)
def uniform_lbp(image, radius=1, neighbors=8):
    lbp = manual_lbp(image, radius, neighbors)

    # Преобразование в униформные коды
    uniform_table = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        binary = np.binary_repr(i, width=8)
        transitions = 0
        for j in range(1, len(binary)):
            if binary[j] != binary[j - 1]:
                transitions += 1
        if transitions <= 2:
            uniform_table[i] = np.sum(np.unpackbits(np.array([i], dtype=np.uint8)))
        else:
            uniform_table[i] = neighbors + 1  # Неуниформный код

    return uniform_table[lbp]


# =============================================
# 5. Построение гистограммы LBP
# =============================================
def plot_lbp_histogram(hist, title, filename):
    plt.figure()
    plt.bar(np.arange(len(hist)), hist, color='green', alpha=0.7)
    plt.title(title)
    plt.xlabel("LBP код")
    plt.ylabel("Частота")
    plt.savefig(filename)
    plt.close()


# =============================================
# 6. Энтропия LBP
# =============================================
def compute_entropy(hist):
    hist_normalized = hist / hist.sum()
    return -np.sum(hist_normalized * np.log2(hist_normalized + 1e-10))


# =============================================
# 7. Генерация отчёта в Markdown
# =============================================
def generate_md_report(results_dir, image_name, gamma, entropy_orig, entropy_corr, lbp_radius):
    report_path = os.path.join(results_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Отчёт по лабораторной работе №8 (Вариант 5)\n\n")
        f.write(f"**Дата выполнения:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Исходное изображение:** `{image_name}`\n")
        f.write(f"**Параметры:**\n")
        f.write(f"- Гамма-коррекция: γ = {gamma}\n")
        f.write(f"- Радиус LBP: {lbp_radius}\n\n")

        f.write("## Результаты\n\n")
        f.write("### 1. Исходное и контрастированное изображение\n")
        f.write(f"![Исходное](original.png) | ![Контрастированное](corrected.png)\n")
        f.write(":-:|:-:\n")
        f.write("Исходное (RGB) | После гамма-коррекции\n\n")

        f.write("### 2. Гистограммы яркости\n")
        f.write(f"![Гистограмма исходная](hist_original.png) | ![Гистограмма после коррекции](hist_corrected.png)\n")
        f.write(":-:|:-:\n")
        f.write("До преобразования | После преобразования\n\n")

        f.write("### 3. LBP-анализ\n")
        f.write(f"![LBP исходное](lbp_original.png) | ![LBP контрастированное](lbp_corrected.png)\n")
        f.write(":-:|:-:\n")
        f.write("LBP-карта (оригинал) | LBP-карта (после коррекции)\n\n")

        f.write(
            f"![Гистограмма LBP (оригинал)](lbp_hist_original.png) | ![Гистограмма LBP (коррекция)](lbp_hist_corrected.png)\n")
        f.write(":-:|:-:\n")
        f.write("Гистограмма LBP-кодов (оригинал) | Гистограмма LBP-кодов (коррекция)\n\n")

        f.write("### 4. Текстурные признаки\n")
        f.write("| Метрика       | Исходное | После коррекции |\n")
        f.write("|--------------|----------|-----------------|\n")
        f.write(f"| Энтропия LBP | {entropy_orig:.4f} | {entropy_corr:.4f} |\n\n")

        f.write("## Выводы\n")
        f.write(
            f"- После гамма-коррекции (γ = {gamma}) яркость изображения {'увеличилась' if gamma > 1 else 'уменьшилась'}.\n")
        f.write(
            f"- Энтропия LBP {'увеличилась' if entropy_corr > entropy_orig else 'уменьшилась'}, что означает {'усиление' if entropy_corr > entropy_orig else 'ослабление'} текстурных особенностей.\n")


# =============================================
# 8. Основная функция
# =============================================
# ... (предыдущий код остаётся без изменений до функции main())

def main():
    # Параметры
    image_paths = ["cat.png", "cartoon.png", "atlas.png"]  # Список изображений для обработки
    gamma = 1.5
    lbp_radius = 1  # Радиус для LBP (1 или 2)
    neighbors = 8  # Количество соседей в LBP

    # Создание папки для результатов
    results_dir = create_results_folder()

    # Создаём общий отчёт
    report_path = os.path.join(results_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Отчёт по лабораторной работе №8 (Вариант 5)\n\n")
        f.write(f"**Дата выполнения:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Параметры:**\n")
        f.write(f"- Гамма-коррекция: γ = {gamma}\n")
        f.write(f"- Радиус LBP: {lbp_radius}\n\n")

    # Обрабатываем каждое изображение
    for image_path in image_paths:
        if not os.path.exists(image_path):
            print(f"Изображение {image_path} не найдено, пропускаем...")
            continue

        # Создаём подпапку для каждого изображения
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        image_dir = os.path.join(results_dir, image_name)
        os.makedirs(image_dir, exist_ok=True)

        # 1. Загрузка и конвертация
        try:
            image_rgb, H, L, S = load_and_convert_to_hsl(image_path)
        except Exception as e:
            print(f"Ошибка при загрузке {image_path}: {e}")
            continue

        # 2. Гамма-коррекция
        L_corrected = gamma_correction(L, gamma=gamma)
        image_hsl_corrected = cv2.merge([H, L_corrected, S])
        image_rgb_corrected = cv2.cvtColor(image_hsl_corrected, cv2.COLOR_HLS2RGB)

        # 3. Сохранение изображений
        cv2.imwrite(os.path.join(image_dir, "original.png"), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(image_dir, "corrected.png"), cv2.cvtColor(image_rgb_corrected, cv2.COLOR_RGB2BGR))

        # 4. Гистограммы яркости
        plot_histogram(L, "Гистограмма яркости (исходная)", os.path.join(image_dir, "hist_original.png"))
        plot_histogram(L_corrected, "Гистограмма яркости (после коррекции)",
                       os.path.join(image_dir, "hist_corrected.png"))

        # 5. LBP-анализ
        lbp_original = uniform_lbp(L, radius=lbp_radius, neighbors=neighbors)
        lbp_corrected = uniform_lbp(L_corrected, radius=lbp_radius, neighbors=neighbors)

        # Гистограммы LBP
        hist_original, _ = np.histogram(lbp_original.ravel(), bins=np.arange(0, neighbors + 3),
                                        range=(0, neighbors + 2))
        hist_corrected, _ = np.histogram(lbp_corrected.ravel(), bins=np.arange(0, neighbors + 3),
                                         range=(0, neighbors + 2))

        # Визуализация LBP-карт
        plt.imsave(os.path.join(image_dir, "lbp_original.png"), lbp_original, cmap="gray")
        plt.imsave(os.path.join(image_dir, "lbp_corrected.png"), lbp_corrected, cmap="gray")

        # Гистограммы LBP
        plot_lbp_histogram(hist_original, "Гистограмма LBP (исходная)",
                           os.path.join(image_dir, "lbp_hist_original.png"))
        plot_lbp_histogram(hist_corrected, "Гистограмма LBP (после коррекции)",
                           os.path.join(image_dir, "lbp_hist_corrected.png"))

        # 6. Энтропия LBP
        entropy_original = compute_entropy(hist_original)
        entropy_corrected = compute_entropy(hist_corrected)

        # Добавляем результаты в отчёт
        with open(report_path, "a", encoding="utf-8") as f:
            f.write(f"## Результаты для изображения: {image_name}\n\n")

            f.write("### 1. Исходное и контрастированное изображение\n")
            f.write(f"![Исходное]({image_name}/original.png) | ![Контрастированное]({image_name}/corrected.png)\n")
            f.write(":-:|:-:\n")
            f.write(f"Исходное (RGB) | После гамма-коррекции (γ = {gamma})\n\n")

            f.write("### 2. Гистограммы яркости\n")
            f.write(
                f"![Гистограмма исходная]({image_name}/hist_original.png) | ![Гистограмма после коррекции]({image_name}/hist_corrected.png)\n")
            f.write(":-:|:-:\n")
            f.write("До преобразования | После преобразования\n\n")

            f.write("### 3. LBP-анализ\n")
            f.write(
                f"![LBP исходное]({image_name}/lbp_original.png) | ![LBP контрастированное]({image_name}/lbp_corrected.png)\n")
            f.write(":-:|:-:\n")
            f.write("LBP-карта (оригинал) | LBP-карта (после коррекции)\n\n")

            f.write(
                f"![Гистограмма LBP (оригинал)]({image_name}/lbp_hist_original.png) | ![Гистограмма LBP (коррекция)]({image_name}/lbp_hist_corrected.png)\n")
            f.write(":-:|:-:\n")
            f.write("Гистограмма LBP-кодов (оригинал) | Гистограмма LBP-кодов (коррекция)\n\n")

            f.write("### 4. Текстурные признаки\n")
            f.write("| Метрика       | Исходное | После коррекции |\n")
            f.write("|--------------|----------|-----------------|\n")
            f.write(f"| Энтропия LBP | {entropy_original:.4f} | {entropy_corrected:.4f} |\n\n")

            f.write(f"### Выводы для {image_name}\n")
            f.write(
                f"- После гамма-коррекции (γ = {gamma}) яркость изображения {'увеличилась' if gamma > 1 else 'уменьшилась'}.\n")
            f.write(
                f"- Энтропия LBP {'увеличилась' if entropy_corrected > entropy_original else 'уменьшилась'}, что означает {'усиление' if entropy_corrected > entropy_original else 'ослабление'} текстурных особенностей.\n\n")

    print(f"Отчёт сгенерирован в папке: {os.path.abspath(results_dir)}")


if __name__ == "__main__":
    main()