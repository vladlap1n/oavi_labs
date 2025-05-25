import numpy as np
from PIL import Image

def apply_median_filter(image, mask):
    pixels = np.array(image, dtype=np.uint8)
    height, width = pixels.shape
    filtered_pixels = np.zeros_like(pixels)

    for y in range(height):
        for x in range(width):
            neighbors = []
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if mask[dy + 1, dx + 1]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            neighbors.append(pixels[ny, nx])
            if neighbors:
                filtered_pixels[y, x] = np.median(neighbors)
            else:
                filtered_pixels[y, x] = pixels[y, x]

    filtered_image = Image.fromarray(filtered_pixels, mode="L")
    return filtered_image

def create_difference_image(original, filtered):
    original_pixels = np.array(original, dtype=np.int16)
    filtered_pixels = np.array(filtered, dtype=np.int16)
    difference_pixels = np.abs(original_pixels - filtered_pixels).astype(np.uint8)
    difference_image = Image.fromarray(difference_pixels, mode="L")
    return difference_image


input_image_path = "test_image.png"
original_image = Image.open(input_image_path).convert("L")
mask = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]
], dtype=bool)

filtered_image = apply_median_filter(original_image, mask)
filtered_image.save("output_filtered.bmp")
difference_image = create_difference_image(original_image, filtered_image)
difference_image.save("output_difference.bmp")

input_image_path="atlas.png"
original_image = Image.open(input_image_path).convert("L")

filtered_image = apply_median_filter(original_image, mask)
filtered_image.save("output_filtered_atlas.bmp")
difference_image = create_difference_image(original_image, filtered_image)
difference_image.save("output_difference_atlas.bmp")

input_image_path="book.png"
original_image = Image.open(input_image_path).convert("L")

filtered_image = apply_median_filter(original_image, mask)
filtered_image.save("output_filtered_book.bmp")
difference_image = create_difference_image(original_image, filtered_image)
difference_image.save("output_difference_book.bmp")

input_image_path="cartoon.png"
original_image = Image.open(input_image_path).convert("L")

filtered_image = apply_median_filter(original_image, mask)
filtered_image.save("output_filtered_cartoon.bmp")
difference_image = create_difference_image(original_image, filtered_image)
difference_image.save("output_difference_cartoon.bmp")

input_image_path="cat.png"
original_image = Image.open(input_image_path).convert("L")

filtered_image = apply_median_filter(original_image, mask)
filtered_image.save("output_filtered_cat.bmp")
difference_image = create_difference_image(original_image, filtered_image)
difference_image.save("output_difference_cat.bmp")

input_image_path="fingers.png"
original_image = Image.open(input_image_path).convert("L")

filtered_image = apply_median_filter(original_image, mask)
filtered_image.save("output_filtered_fingers.bmp")
difference_image = create_difference_image(original_image, filtered_image)
difference_image.save("output_difference_fingers.bmp")

input_image_path="rentgen.png"
original_image = Image.open(input_image_path).convert("L")

filtered_image = apply_median_filter(original_image, mask)
filtered_image.save("output_filtered_rentgen.bmp")
difference_image = create_difference_image(original_image, filtered_image)
difference_image.save("output_difference_rentgen.bmp")

input_image_path="output_binary_cat.bmp"
original_image = Image.open(input_image_path).convert("L")

filtered_image = apply_median_filter(original_image, mask)
filtered_image.save("output_filtered_binary_cat.bmp")
difference_image = create_difference_image(original_image, filtered_image)
difference_image.save("output_difference_binary_cat.bmp")

report_text = f"""# Отчет по лабораторной работе №3

## Задание: Фильтрация изображений и морфологические операции
В данной работе применялся медианный фильтр с разреженной маской в виде прямого креста размером \( 3 \times 3 \). Маска имеет следующий вид:

\[
010
111
010
\]

Для каждого изображения были получены:
1. Отфильтрованное монохромное изображение.
2. Разностное изображение (модуль разности между исходным и отфильтрованным изображениями).

---

## Результаты обработки изображений

### 1. Изображение: `test_image.png`
- **Исходное изображение**:
  ![Исходное изображение](test_image.png)
- **Отфильтрованное изображение**:
  ![Отфильтрованное изображение](output_filtered.bmp)
- **Разностное изображение**:
  ![Разностное изображение](output_difference.bmp)

### 2. Изображение: `atlas.png`
- **Исходное изображение**:
  ![Исходное изображение](atlas.png)
- **Отфильтрованное изображение**:
  ![Отфильтрованное изображение](output_filtered_atlas.bmp)
- **Разностное изображение**:
  ![Разностное изображение](output_difference_atlas.bmp)

### 3. Изображение: `book.png`
- **Исходное изображение**:
  ![Исходное изображение](book.png)
- **Отфильтрованное изображение**:
  ![Отфильтрованное изображение](output_filtered_book.bmp)
- **Разностное изображение**:
  ![Разностное изображение](output_difference_book.bmp)

### 4. Изображение: `cartoon.png`
- **Исходное изображение**:
  ![Исходное изображение](cartoon.png)
- **Отфильтрованное изображение**:
  ![Отфильтрованное изображение](output_filtered_cartoon.bmp)
- **Разностное изображение**:
  ![Разностное изображение](output_difference_cartoon.bmp)

### 5. Изображение: `cat.png`
- **Исходное изображение**:
  ![Исходное изображение](cat.png)
- **Отфильтрованное изображение**:
  ![Отфильтрованное изображение](output_filtered_cat.bmp)
- **Разностное изображение**:
  ![Разностное изображение](output_difference_cat.bmp)

### 6. Изображение: `fingers.png`
- **Исходное изображение**:
  ![Исходное изображение](fingers.png)
- **Отфильтрованное изображение**:
  ![Отфильтрованное изображение](output_filtered_fingers.bmp)
- **Разностное изображение**:
  ![Разностное изображение](output_difference_fingers.bmp)

### 7. Изображение: `rentgen.png`
- **Исходное изображение**:
  ![Исходное изображение](rentgen.png)
- **Отфильтрованное изображение**:
  ![Отфильтрованное изображение](output_filtered_rentgen.bmp)
- **Разностное изображение**:
  ![Разностное изображение](output_difference_rentgen.bmp)
  
### 8. Изображение: `output_binary_cat.png`
- **Исходное изображение**:
  ![Исходное изображение](output_binary_cat.bmp)
- **Отфильтрованное изображение**:
  ![Отфильтрованное изображение](output_filtered_binary_cat.bmp)
- **Разностное изображение**:
  ![Разностное изображение](output_difference_binary_cat.bmp)

---

## Параметры обработки
- **Метод фильтрации**: Медианный фильтр.
- **Маска**: Прямой крест размером \( 3 \times 3 \):
  \[
  010
  111
  010
  \]
- **Ранг фильтра**: 3/5 """

with open('report_lab3_исп__.md', 'w', encoding='utf-8') as report_file:
    report_file.write(report_text)