import cv2
import numpy as np
import os
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

ALPHABET = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
THRESHOLD = 128
MIN_SYMBOL_WIDTH = 5
MAX_SYMBOL_WIDTH = 40
TARGET_SIZE = (32, 32)

def auto_crop(image):
    coords = cv2.findNonZero(image)
    if coords is None:
        return image
    x, y, w, h = cv2.boundingRect(coords)
    return image[y:y + h, x:x + w]

def add_padding(image, padding=2):
    return cv2.copyMakeBorder(image, padding, padding, padding, padding,
                              cv2.BORDER_CONSTANT, value=0)

def resize_image(img, target_size=TARGET_SIZE):
    return cv2.resize(img, target_size)

def rename_reference_images(reference_dir):
    if not os.path.exists(reference_dir):
        print(f"Ошибка: папка '{reference_dir}' не найдена")
        return
    renamed_count = 0
    for idx, char in enumerate(ALPHABET):
        old_path = os.path.join(reference_dir, f"{char}.png")
        new_path = os.path.join(reference_dir, f"{idx}.png")
        if os.path.exists(old_path):
            os.rename(old_path, new_path)
            renamed_count += 1
        else:
            print(f"Файл не найден: {old_path}")
    print(f"Переименовано файлов: {renamed_count}")

def load_reference_images(reference_dir):
    reference = {}
    for idx, char in enumerate(ALPHABET):
        img_path = os.path.join(reference_dir, f"{idx}.png")
        if os.path.exists(img_path):
            img = cv2.imread(img_path, 0)
            if img is not None:
                _, binary = cv2.threshold(img, THRESHOLD, 255, cv2.THRESH_BINARY_INV)
                cropped = auto_crop(binary)
                padded = add_padding(cropped)
                resized = resize_image(padded)
                reference[char] = resized
                cv2.imshow(f"Reference {char}", resized)
                cv2.waitKey(50)
    cv2.destroyAllWindows()
    return reference

def calculate_features(image):
    binary = np.where(image > 127, 0, 1).astype(np.float32)
    m00 = np.sum(binary)
    if m00 < 1:
        return None
    h, w = binary.shape
    y, x = np.indices((h, w))
    x_c = np.sum(x * binary) / m00
    y_c = np.sum(y * binary) / m00

    mu20 = np.sum((x - x_c)**2 * binary) / m00
    mu02 = np.sum((y - y_c)**2 * binary) / m00
    mu11 = np.sum((x - x_c)*(y - y_c)*binary) / m00

    # Центральные моменты
    mu20_centered = np.sum((x - x_c)**2 * binary)
    mu02_centered = np.sum((y - y_c)**2 * binary)
    mu11_centered = np.sum((x - x_c)*(y - y_c)*binary)

    # Соотношение сторон
    aspect_ratio = w / h if h != 0 else 0

    # Подсчёт количества компонент
    num_labels, _ = cv2.connectedComponents(binary.astype(np.uint8))

    return np.array([
        m00 / (w * h),
        x_c / w,
        y_c / h,
        mu20 / (w ** 2),
        mu02 / (h ** 2),
        mu11 / (w * h),
        mu20_centered / (w ** 2),
        mu02_centered / (h ** 2),
        mu11_centered / (w * h),
        aspect_ratio,
        num_labels
    ])

def preprocess_image(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    processed = cv2.adaptiveThreshold(enhanced, 255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((2, 2), np.uint8)
    processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
    return processed

def segment_characters(image):
    inverted = 255 - image
    vertical_proj = np.sum(inverted, axis=0) / 255
    plt.figure(figsize=(10, 3))
    plt.plot(vertical_proj)
    plt.title('Вертикальная проекция')
    plt.show()

    segments = []
    start = None
    for i, val in enumerate(vertical_proj):
        if val > 0 and start is None:
            start = i
        elif val == 0 and start is not None:
            width = i - start
            if width > MAX_SYMBOL_WIDTH:
                parts = int(np.ceil(width / (MAX_SYMBOL_WIDTH / 2)))
                for p in range(parts):
                    seg_start = start + p * width // parts
                    seg_end = start + (p + 1) * width // parts
                    segments.append((seg_start, seg_end))
            else:
                segments.append((start, i))
            start = None
    if start is not None:
        segments.append((start, len(vertical_proj)))

    characters = []
    for x1, x2 in segments:
        if x2 - x1 < MIN_SYMBOL_WIDTH:
            continue
        column = image[:, x1:x2]
        horizontal_proj = np.sum(column, axis=1) / 255
        y_indices = np.where(horizontal_proj > 0)[0]
        if len(y_indices) == 0:
            continue
        y1, y2 = max(0, y_indices[0] - 2), min(image.shape[0], y_indices[-1] + 2)
        char_img = image[y1:y2, x1:x2]
        if char_img.size == 0:
            continue
        resized = resize_image(char_img)
        characters.append(resized)

    plt.figure(figsize=(10, 2))
    for i, c in enumerate(characters):
        plt.subplot(1, len(characters), i + 1)
        plt.imshow(c, cmap='gray')
        plt.axis('off')
    plt.show()
    return characters

def recognize_characters(image_path, reference):
    if not os.path.exists(image_path):
        print(f"Ошибка: файл {image_path} не найден")
        return []
    img = cv2.imread(image_path, 0)
    if img is None:
        print(f"Не удалось загрузить изображение {image_path}")
        return []

    processed = preprocess_image(img)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1), plt.imshow(img, cmap='gray'), plt.title('Original')
    plt.subplot(1, 2, 2), plt.imshow(processed, cmap='gray'), plt.title('Processed')
    plt.show()

    characters = segment_characters(processed)
    print(f"Найдено символов: {len(characters)}")

    all_features = []
    ref_features = {}
    for char, img_ref in reference.items():
        features = calculate_features(img_ref)
        if features is not None:
            all_features.append(features)
            ref_features[char] = features

    if len(all_features) == 0:
        print("Нет эталонных признаков для нормализации")
        return []

    scaler = MinMaxScaler()
    scaled_all = scaler.fit_transform(all_features)

    for i, char in enumerate(ref_features.keys()):
        ref_features[char] = scaled_all[i]

    results = []
    for i, char_img in enumerate(characters):
        features = calculate_features(char_img)
        if features is None:
            results.append((i + 1, [("?", 0)]))
            continue
        scaled = scaler.transform([features])[0]
        hypotheses = []
        for char, ref_feat in ref_features.items():
            sim = cosine_similarity([scaled], [ref_feat])[0][0]
            hypotheses.append((char, sim))
        hypotheses.sort(key=lambda x: x[1], reverse=True)
        results.append((i + 1, hypotheses))
    return results

def save_results(results, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for idx, hypotheses in results:
            line = f"{idx}: ["
            for char, score in hypotheses[:10]:
                line += f"('{char}', {score:.3f}), "
            line = line.rstrip(", ") + "]\n"
            f.write(line)

def print_recognition_summary(results, expected_phrase=""):
    if not results:
        print("Нет результатов для вывода")
        return
    recognized = "".join([hyp[0][0] for _, hyp in results])
    print(f"\nРаспознанная строка: {recognized}")
    if expected_phrase:
        recognized = recognized.ljust(len(expected_phrase))[:len(expected_phrase)]
        correct = sum(1 for r, e in zip(recognized, expected_phrase) if r == e)
        accuracy = correct / len(expected_phrase) * 100
        print(f"Ожидаемая строка: {expected_phrase}")
        print(f"Точность: {accuracy:.1f}% ({correct}/{len(expected_phrase)})")

def generate_md_report(results_dict, expected_phrase, image_names):
    report = """# Лабораторная работа №7
## Распознавание русских заглавных курсивных символов
**Вариант:** Русский заглавный курсив
### Цель работы
Реализация системы распознавания символов на основе признаков.
### Входные данные
- Ожидаемая строка: `МОЙ ЛЮБИМЫЙ`
- Размер алфавита: 33 символов
### Этапы реализации
1. **Предобработка изображения**:
   - Адаптивная бинаризация
   - Удаление шумов с помощью морфологической операции открытия
2. **Сегментация символов**:
   - Анализ вертикальной проекции
   - Разделение слипшихся символов при превышении максимальной ширины
3. **Извлечение признаков**:
   - Нормированная масса
   - Координаты центра тяжести
   - Осевые моменты инерции
4. **Классификация**:
   - Евклидово расстояние в пространстве признаков
5. **Постобработка**:
   - Ранжирование гипотез по мере близости
   - Формирование итоговой строки
"""

    for name in image_names:
        data = results_dict[name]
        report += f"""
### Изображение: {name} (`{data['image_path']}`)
**Результаты распознавания:**
| Метод | Распознанная строка | Точность |
|-------|----------------------|----------|
| Без профилей | `{"".join([hyp[0][0] for _, hyp in data['results']])}` | {data['accuracy_without']:.1f}% |
**Подробные гипотезы для символов:**
"""
        for idx, hypotheses in data['results']:
            report += f"\n#### Символ {idx}:\n"
            report += "| Символ | Мера близости |\n"
            report += "|--------|-------------|\n"
            for char, score in hypotheses[:5]:  # Только топ-5 гипотез
                report += f"| {char} | {score:.3f} |\n"

    report += """
### Анализ ошибок и сравнение результатов
**Сравнение точности для разных изображений:**
| Изображение | Точность |
|-------------|----------|"""
    for name in image_names:
        data = results_dict[name]
        report += f"\n| {name} | {data['accuracy_without']:.1f}% |"

    total_accuracy = sum(data['accuracy_without'] for data in results_dict.values()) / len(results_dict)
    report += f"""
### Общий анализ эффективности методов
**Основные характеристики:**
- Количество протестированных изображений: {len(image_names)}
- Средняя точность: {total_accuracy:.1f}%
**Преимущества текущего подхода:**
1. Простой и понятный способ классификации
2. Хорошая устойчивость к изменениям масштаба благодаря масштабированию
3. Эффективное использование статистических признаков
**Недостатки текущей реализации:**
1. Зависимость точности от качества сегментации
2. Чувствительность к шумам и дефектам изображения
3. Ограниченная способность различать визуально похожие символы
### Выводы
1. Реализована система распознавания русских заглавных курсивных символов
2. Использован метод евклидова расстояния в признаковом пространстве
3. Эксперименты показали, что метод демонстрирует хорошие результаты на стандартном размере шрифта
4. Для повышения точности рекомендуется:
   - Улучшить алгоритмы предобработки изображения
   - Добавить адаптивную нормализацию размера символов
   - Расширить набор используемых признаков
"""

    with open("lab7_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("\nОтчет успешно сохранен в файл 'lab7_report.md'")

def main():
    reference_dir = "reference"
    if not os.path.exists(reference_dir):
        print(f"Ошибка: папка '{reference_dir}' не найдена")
        print(f"Создайте папку '{reference_dir}' и поместите туда изображения")
        return

    print("Переименование эталонных изображений...")
    rename_reference_images(reference_dir)
    print("Загрузка эталонных изображений...")
    reference = load_reference_images(reference_dir)

    test_images = {
        "Основное изображение": "input.bmp",
        "Другой размер": "input_alt_size.bmp"
    }

    expected_phrase = input("Введите ожидаемую строку для теста (или Enter): ").upper()
    results_dict = {}
    image_names = []

    for name, path in test_images.items():
        if not os.path.exists(path):
            print(f"\nФайл не найден: {path}, пропускаем...")
            continue
        image_names.append(name)
        print(f"\n--- Распознавание: {name} ---")
        results = recognize_characters(path, reference)
        save_results(results, f"results_{name.replace(' ', '_')}.txt")
        print_recognition_summary(results, expected_phrase)

        accuracy_without = 0
        if expected_phrase:
            recognized = "".join([hyp[0][0] for _, hyp in results])
            recognized = recognized.ljust(len(expected_phrase))[:len(expected_phrase)]
            correct = sum(1 for r, e in zip(recognized, expected_phrase) if r == e)
            accuracy_without = correct / len(expected_phrase) * 100 if expected_phrase else 0

        results_dict[name] = {
            'results': results,
            'accuracy_without': accuracy_without,
            'image_path': path
        }

    if results_dict:
        generate_md_report(results_dict, expected_phrase, image_names)

if __name__ == "__main__":
    main()