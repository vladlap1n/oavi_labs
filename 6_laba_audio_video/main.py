import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

IMAGE_PATH = 'phrase.bmp'

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError("❌ Файл phrase.bmp не найден.")

os.makedirs('results', exist_ok=True)
os.makedirs('profiles', exist_ok=True)
os.makedirs('results/characters', exist_ok=True)

# Загрузка и предобработка
img = cv2.imread(IMAGE_PATH, 0)
_, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

def compute_profiles(img_bin):
    """Вычисление горизонтального и вертикального профилей"""
    return np.sum(img_bin, axis=0), np.sum(img_bin, axis=1)

def rotate_image(image, angle):
    """Поворот изображения с компенсацией фона"""
    center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1],
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0)
    _, result = cv2.threshold(result, 127, 255, cv2.THRESH_BINARY)
    return result, rot_mat

def unrotate_boxes(boxes, rot_mat, original_shape):
    """Обратное преобразование координат после поворота"""
    transformed_boxes = []
    inv_rot_mat = cv2.invertAffineTransform(rot_mat)
    for x1, y1, x2, y2 in boxes:
        # Центр прямоугольника
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        # Преобразование координат
        coords = np.array([[cx, cy, 1]]).T
        new_coords = inv_rot_mat @ coords
        nx, ny = new_coords[:, 0].astype(int)
        # Коррекция размеров
        w, h = x2 - x1, y2 - y1
        nx = np.clip(nx, 0, original_shape[1]-1)
        ny = np.clip(ny, 0, original_shape[0]-1)
        transformed_boxes.append((nx - w//2, ny - h//2, nx + w//2, ny + h//2))
    return transformed_boxes

def segment_cursive(img_bin, min_thresh=3, min_char_width=11, max_char_width=70):
    """Улучшенный алгоритм сегментации с прореживанием"""
    shifted = np.zeros_like(img_bin)
    for row in range(img_bin.shape[0]):
        shift = 1 * (row // 25)
        shifted[row, max(0, shift):] = img_bin[row, 0:img_bin.shape[1] - shift]

    blurred = cv2.GaussianBlur(shifted, (3, 3), 0)
    profile = np.sum(blurred, axis=0)

    threshold = max(min_thresh, np.percentile(profile[profile > 0], 25))

    segments = []
    start, in_char = 0, False
    for x in range(len(profile)):
        if profile[x] > threshold and not in_char:
            start = max(0, x - 2)
            in_char = True
        elif profile[x] <= threshold and in_char:
            end = min(x + 2, len(profile))
            if end - start >= min_char_width:
                segments.append((start, end))
            in_char = False
    if in_char:
        segments.append((start, len(profile)))

    refined = []
    for (s, e) in segments:
        sub_profile = profile[s:e]
        min_val = np.min(sub_profile)

        if min_val > threshold * 0.6 or (e - s) < max_char_width:
            refined.append((s, e))
            continue

        split_pos = np.argmin(sub_profile) + s
        refined.extend([(s, split_pos), (split_pos, e)])

    filtered = [(s, e) for s, e in refined
                if min_char_width <= (e - s) <= max_char_width]

    boxes = []
    for x1, x2 in filtered:
        region = img_bin[:, x1:x2]
        rows = np.any(region, axis=1)
        if not np.any(rows): continue

        y1 = np.where(rows)[0][0]
        y2 = np.where(rows)[0][-1]

        boxes.append((
            max(0, x1 - 1), max(0, y1 - 2),
            min(img_bin.shape[1], x2 + 1),
            min(img_bin.shape[0], y2 + 2)
        ))

    return boxes

# Вычисление и визуализация общих профилей
horizontal_profile, vertical_profile = compute_profiles(img_bin)

plt.figure(figsize=(10, 4))
plt.plot(horizontal_profile, color='black')
plt.title('Горизонтальный профиль изображения')
plt.grid(True)
plt.savefig('profiles/horizontal_full.png')
plt.close()

plt.figure(figsize=(4, 10))
plt.plot(vertical_profile, color='black')
plt.title('Вертикальный профиль изображения')
plt.grid(True)
plt.savefig('profiles/vertical_full.png')
plt.close()

# Компенсация курсива через поворот
angle = -12  # Оптимальный угол наклона для курсива
rotated_img, rot_mat = rotate_image(img_bin, angle)

# Сегментация на повернутом изображении
boxes_rotated = segment_cursive(rotated_img)

# Обратное преобразование координат
boxes = unrotate_boxes(boxes_rotated, rot_mat, img_bin.shape)

print(f"✅ Найдено {len(boxes)} символов.")

# Визуализация результатов
output_img = cv2.cvtColor(cv2.bitwise_not(img_bin), cv2.COLOR_GRAY2BGR)
for i, (x1, y1, x2, y2) in enumerate(boxes):
    cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cv2.putText(output_img, str(i), (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

cv2.imwrite('results/segmented_text.png', output_img)

# Сохранение символов и профилей
for i, (x1, y1, x2, y2) in enumerate(boxes):
    char_img = img_bin[y1:y2, x1:x2]
    if char_img.size == 0:
        continue

    cv2.imwrite(f'results/characters/char_{i}.png', cv2.bitwise_not(char_img))

    h_prof, v_prof = compute_profiles(char_img)

    plt.figure(figsize=(6, 2))
    plt.plot(h_prof, color='black')
    plt.title(f'char_{i}_horiz')
    plt.grid(True)
    plt.savefig(f'profiles/char_{i}_horiz.png')
    plt.close()

    plt.figure(figsize=(2, 6))
    plt.plot(v_prof, color='black')
    plt.title(f'char_{i}_vert')
    plt.grid(True)
    plt.savefig(f'profiles/char_{i}_vert.png')
    plt.close()

# Общий обрамляющий прямоугольник
coords = np.column_stack(np.where(img_bin == 255))
if coords.size > 0:
    y1, x1 = coords.min(axis=0)
    y2, x2 = coords.max(axis=0)
    full_output = cv2.cvtColor(cv2.bitwise_not(img_bin), cv2.COLOR_GRAY2BGR)
    cv2.rectangle(full_output, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imwrite('results/bounding_rect.png', full_output)

# Генерация отчета
CHAR_COUNT = len([f for f in os.listdir("results/characters") if f.startswith("char_") and f.endswith(".png")])

MD_FILENAME = "report.md"

md_content = [
    "# Лабораторная работа №6",
    "## *Сегментация текста и анализ символов*",
    "",
    "### Цель работы:",
    "Исследование методов сегментации строк на символы и построение профилей символов алфавита.",
    "Реализация алгоритма сегментации с учётом особенностей курсивного шрифта.",
    "",
    "## 1. Подготовка изображения",
    "Для работы была подготовлена фраза на русском языке (заглавные буквы) в Microsoft Word ",
    "с использованием курсивного шрифта. Скриншот был сохранён в файл `phrase.bmp` в монохромном формате без белого фона вокруг строки.",
    "",
    "> Пример исходного изображения:",
    "",
    "![Фраза в BMP](phrase.bmp)",
    "",
    "## 2. Бинаризация и инвертирование",
    "Изображение было загружено в градациях серого и преобразовано в бинарное изображение методом пороговой обработки (`cv2.THRESH_BINARY_INV`). Это позволило выделить символы как светлые области на тёмном фоне.",
    "",
    "## 3. Расчёт профилей изображения",
    "Для всего изображения были вычислены профили активности пикселей:",
    "- **Горизонтальный профиль** — сумма пикселей по столбцам",
    "- **Вертикальный профиль** — сумма пикселей по строкам",
    "",
    "![Горизонтальный профиль](profiles/horizontal_full.png)",
    "![Вертикальный профиль](profiles/vertical_full.png)",
    "",
    "## 4. Улучшенный алгоритм сегментации курсивного текста",
    "Для компенсации наклона курсива реализован следующий подход:",
    "1. Поворот изображения на -12° для выравнивания символов",
    "2. Сегментация на повернутом изображении",
    "3. Обратное преобразование координат для возврата к исходной ориентации",
    "",
    "> Результат сегментации:",
    "",
    "![Сегментированный текст](results/segmented_text.png)",
    "",

]

md_content.extend([
    "",
    "## 5. [Дополнительно] Общий обрамляющий прямоугольник",
    "Рассчитан и отрисован общий ограничивающий прямоугольник для всей строки текста:",
    "",
    "![Общий прямоугольник](results/bounding_rect.png)",
    "",
    "## Заключение",
    "Алгоритм успешно справляется с задачей сегментации строки на отдельные символы даже в случае курсивного шрифта.",
    "Благодаря применению динамического порога и предварительной компенсации наклона удалось минимизировать ошибки склеивания или разрыва букв.",
    "Построенные профили могут быть использованы в дальнейших задачах классификации и распознавания символов."
])

with open(MD_FILENAME, 'w', encoding='utf-8') as f:
    f.write('\n'.join(md_content))

print(f"✅ Отчет сохранён в файл: {MD_FILENAME}")