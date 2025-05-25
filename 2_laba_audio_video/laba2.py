import numpy as np
from PIL import Image

def convert_to_grayscale(image):
    pixels = np.array(image)
    grayscale_pixels = np.dot(pixels[..., :3], [0.3, 0.59, 0.11]).astype(np.uint8)
    grayscale_image = Image.fromarray(grayscale_pixels, mode="L")
    return grayscale_image

def bradley_roth_binarization(image, window_size=5, threshold=0.15):
    pixels = np.array(image, dtype=np.float64)
    height, width = pixels.shape
    integral_image = np.zeros_like(pixels, dtype=np.float64)

    for y in range(height):
        for x in range(width):
            integral_image[y, x] = pixels[y, x]
            if y > 0:
                integral_image[y, x] += integral_image[y - 1, x]
            if x > 0:
                integral_image[y, x] += integral_image[y, x - 1]
            if y > 0 and x > 0:
                integral_image[y, x] -= integral_image[y - 1, x - 1]

    binary_pixels = np.zeros_like(pixels, dtype=np.uint8)
    r = window_size // 2

    for y in range(height):
        for x in range(width):
            y1 = max(0, y - r)
            y2 = min(height - 1, y + r)
            x1 = max(0, x - r)
            x2 = min(width - 1, x + r)

            total = integral_image[y2, x2]
            if y1 > 0:
                total -= integral_image[y1 - 1, x2]
            if x1 > 0:
                total -= integral_image[y2, x1 - 1]
            if y1 > 0 and x1 > 0:
                total += integral_image[y1 - 1, x1 - 1]

            count = (y2 - y1 + 1) * (x2 - x1 + 1)
            mean_brightness = total / count

            if pixels[y, x] < mean_brightness * (1 - threshold):
                binary_pixels[y, x] = 0
            else:
                binary_pixels[y, x] = 255

    binary_image = Image.fromarray(binary_pixels, mode="L")
    return binary_image


test_image1 = "test_image.png"
test_image = Image.open(test_image1).convert("RGB")
grayscale_image = convert_to_grayscale(test_image)
grayscale_image.save("output_grayscale.bmp")
binary_image = bradley_roth_binarization(grayscale_image, window_size=5, threshold=0.15)
binary_image.save("output_binary.bmp")
test_image.show(title="Исходное изображение")
grayscale_image.show(title="Полутоновое изображение")
binary_image.show(title="Бинаризованное изображение")

test_image1 = "atlas.png"
test_image = Image.open(test_image1).convert("RGB")
grayscale_image = convert_to_grayscale(test_image)
grayscale_image.save("output_grayscale_atlas.bmp")
binary_image = bradley_roth_binarization(grayscale_image, window_size=5, threshold=0.15)
binary_image.save("output_binary_atlas.bmp")
test_image.show(title="Исходное изображение")
grayscale_image.show(title="Полутоновое изображение")
binary_image.show(title="Бинаризованное изображение")

test_image1 = "book.png"
test_image = Image.open(test_image1).convert("RGB")
grayscale_image = convert_to_grayscale(test_image)
grayscale_image.save("output_grayscale_book.bmp")
binary_image = bradley_roth_binarization(grayscale_image, window_size=5, threshold=0.15)
binary_image.save("output_binary_book.bmp")
test_image.show(title="Исходное изображение")
grayscale_image.show(title="Полутоновое изображение")
binary_image.show(title="Бинаризованное изображение")

test_image1 = "cartoon.png"
test_image = Image.open(test_image1).convert("RGB")
grayscale_image = convert_to_grayscale(test_image)
grayscale_image.save("output_grayscale_cartoon.bmp")
binary_image = bradley_roth_binarization(grayscale_image, window_size=5, threshold=0.15)
binary_image.save("output_binary_cartoon.bmp")
test_image.show(title="Исходное изображение")
grayscale_image.show(title="Полутоновое изображение")
binary_image.show(title="Бинаризованное изображение")

test_image1 = "cat.png"
test_image = Image.open(test_image1).convert("RGB")
grayscale_image = convert_to_grayscale(test_image)
grayscale_image.save("output_grayscale_cat.bmp")
binary_image = bradley_roth_binarization(grayscale_image, window_size=5, threshold=0.15)
binary_image.save("output_binary_cat.bmp")
test_image.show(title="Исходное изображение")
grayscale_image.show(title="Полутоновое изображение")
binary_image.show(title="Бинаризованное изображение")

test_image1 = "fingers.png"
test_image = Image.open(test_image1).convert("RGB")
grayscale_image = convert_to_grayscale(test_image)
grayscale_image.save("output_grayscale_fingers.bmp")
binary_image = bradley_roth_binarization(grayscale_image, window_size=5, threshold=0.15)
binary_image.save("output_binary_fingers.bmp")
test_image.show(title="Исходное изображение")
grayscale_image.show(title="Полутоновое изображение")
binary_image.show(title="Бинаризованное изображение")

test_image1 = "rentgen.png"
test_image = Image.open(test_image1).convert("RGB")
grayscale_image = convert_to_grayscale(test_image)
grayscale_image.save("output_grayscale_rentgen.bmp")
binary_image = bradley_roth_binarization(grayscale_image, window_size=5, threshold=0.15)
binary_image.save("output_binary_rentgen.bmp")
test_image.show(title="Исходное изображение")
grayscale_image.show(title="Полутоновое изображение")
binary_image.show(title="Бинаризованное изображение")


report_text = f"""#Отчет по лабораторной работе №2

## Задание 1: Приведение полноцветного изображения к полутоновому
Полноцветное изображение было преобразовано в полутоновое с использованием взвешенного усреднения каналов. Формула для преобразования:
\[
I = 0.3 \cdot R + 0.59 \cdot G + 0.11 \cdot B
\]
где \( R, G, B \) — каналы исходного изображения, а \( I \) — яркость пикселя в полутоновом изображении.

## Задание 2: Приведение полутонового изображения к монохромному методом адаптивной бинаризации Брэдли и Рота
Для бинаризации использовался метод адаптивной бинаризации Брэдли и Рота с окном \( 5 \times 5 \) и порогом \( 0.15 \). Этот метод позволяет адаптивно вычислять порог для каждого пикселя на основе локальной яркости в окрестности.

---

## Результаты обработки изображений

### 1. Изображение: `test_image.png`
- **Исходное изображение**:
  ![Исходное изображение](test_image.png)
- **Полутоновое изображение**:
  ![Полутоновое изображение](output_grayscale.bmp)
- **Бинаризованное изображение**:
  ![Бинаризованное изображение](output_binary.bmp)

### 2. Изображение: `atlas.png`
- **Исходное изображение**:
  ![Исходное изображение](atlas.png)
- **Полутоновое изображение**:
  ![Полутоновое изображение](output_grayscale_atlas.bmp)
- **Бинаризованное изображение**:
  ![Бинаризованное изображение](output_binary_atlas.bmp)

### 3. Изображение: `book.png`
- **Исходное изображение**:
  ![Исходное изображение](book.png)
- **Полутоновое изображение**:
  ![Полутоновое изображение](output_grayscale_book.bmp)
- **Бинаризованное изображение**:
  ![Бинаризованное изображение](output_binary_book.bmp)

### 4. Изображение: `cartoon.png`
- **Исходное изображение**:
  ![Исходное изображение](cartoon.png)
- **Полутоновое изображение**:
  ![Полутоновое изображение](output_grayscale_cartoon.bmp)
- **Бинаризованное изображение**:
  ![Бинаризованное изображение](output_binary_cartoon.bmp)

### 5. Изображение: `cat.png`
- **Исходное изображение**:
  ![Исходное изображение](cat.png)
- **Полутоновое изображение**:
  ![Полутоновое изображение](output_grayscale_cat.bmp)
- **Бинаризованное изображение**:
  ![Бинаризованное изображение](output_binary_cat.bmp)

### 6. Изображение: `fingers.png`
- **Исходное изображение**:
  ![Исходное изображение](fingers.png)
- **Полутоновое изображение**:
  ![Полутоновое изображение](output_grayscale_fingers.bmp)
- **Бинаризованное изображение**:
  ![Бинаризованное изображение](output_binary_fingers.bmp)

### 7. Изображение: `rentgen.png`
- **Исходное изображение**:
  ![Исходное изображение](rentgen.png)
- **Полутоновое изображение**:
  ![Полутоновое изображение](output_grayscale_rentgen.bmp)
- **Бинаризованное изображение**:
  ![Бинаризованное изображение](output_binary_rentgen.bmp)

---

## Параметры обработки
- **Метод преобразования в полутоновое изображение**: Взвешенное усреднение каналов (\( 0.3 \cdot R + 0.59 \cdot G + 0.11 \cdot B \)).
- **Метод бинаризации**: Адаптивная бинаризация Брэдли и Рота.
- **Размер окна**: \( 5 \times 5 \).
- **Порог**: \( 0.15 \)."""
with open('report_lab2_исп__.md', 'w', encoding='utf-8') as report_file:
    report_file.write(report_text)