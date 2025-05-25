import numpy as np
import cv2

image = cv2.imread('page.png', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('atlas.png', cv2.IMREAD_GRAYSCALE)
image3 = cv2.imread('cat.png', cv2.IMREAD_GRAYSCALE)
image4 = cv2.imread('cartoon.png', cv2.IMREAD_GRAYSCALE)

def apply_convolution(image, kernel):
    kernel_height, kernel_width = kernel.shape
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
    result = np.zeros_like(image, dtype=np.float32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            result[i, j] = np.sum(padded_image[i:i + kernel_height, j:j + kernel_width] * kernel)
    return result

def process_image(image, prefix):
    kernel_x = np.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]])
    kernel_y = np.array([[3, 0, -3], [10, 0, -10], [3, 0, -3]])
    Gx = apply_convolution(image, kernel_x)
    Gy = apply_convolution(image, kernel_y)
    G = np.sqrt(Gx**2 + Gy**2)
    def normalize(matrix):
        return ((matrix - np.min(matrix)) * (255 / (np.max(matrix) - np.min(matrix)))).astype(np.uint8)
    Gx_norm = normalize(Gx)
    Gy_norm = normalize(Gy)
    G_norm = normalize(G)
    threshold = 50
    G_bin = (G_norm > threshold).astype(np.uint8) * 255
    cv2.imwrite(f'{prefix}_gray_image.jpg', image)
    cv2.imwrite(f'{prefix}_Gx.jpg', Gx_norm)
    cv2.imwrite(f'{prefix}_Gy.jpg', Gy_norm)
    cv2.imwrite(f'{prefix}_G.jpg', G_norm)
    cv2.imwrite(f'{prefix}_G_bin.jpg', G_bin)
    print(f"Изображения для {prefix} успешно сохранены!")

process_image(image, 'page')
process_image(image2, 'atlas')
process_image(image3, 'cat')
process_image(image4, 'cartoon')

report_text = f"""# Лабораторная работа №4
## Обработка изображений с использованием оператора Шарра
## Вариант 5
### Цель работы
Изучение методов обработки изображений, применение оператора Шарра для выделения градиентов и контуров на изображении.

### Задачи
1. Преобразовать цветное изображение в полутоновое.
2. Применить оператор Шарра для вычисления градиентов по осям X и Y.
3. Вычислить общий градиент и нормализовать его.
4. Провести бинаризацию градиентной матрицы.
5. Сохранить результаты обработки.

### Исходные данные
- Изображения: `page.png`, `atlas.png`, `cat.png`, `cartoon.png`.

### Методика выполнения
1. **Загрузка изображений**: Использована функция `cv2.imread` для загрузки изображений в полутоновом формате.
2. **Применение оператора Шарра**:
   - Ядра оператора Шарра:
     ```
     Gx = [[ 3, 10,  3],
           [ 0,  0,  0],
           [-3,-10, -3]]
     
     Gy = [[ 3,  0, -3],
           [10,  0,-10],
           [ 3,  0, -3]]
     ```
   - Свертка изображения с ядрами для получения градиентов по осям X и Y.
3. **Вычисление общего градиента**:
   - Формула: `G = sqrt(Gx^2 + Gy^2)`.
4. **Нормализация**:
   - Приведение значений градиентов к диапазону [0, 255].
5. **Бинаризация**:
   - Порог бинаризации: 50.
   - Значения выше порога устанавливаются в 255, ниже — в 0.
6. **Сохранение результатов**:
   - Для каждого изображения сохранены:
     - Полутоновое изображение.
     - Нормализованные градиенты по осям X и Y.
     - Нормализованный общий градиент.
     - Бинаризованный градиент.

### Результаты
Для каждого изображения получены следующие результаты:

#### Изображение `page.png`
- Исходное изображение:
  ![page.png](page.png)
- Полутоновое изображение:
  ![page_gray_image.jpg](page_gray_image.jpg)
- Градиент по оси X:
  ![page_Gx.jpg](page_Gx.jpg)
- Градиент по оси Y:
  ![page_Gy.jpg](page_Gy.jpg)
- Общий градиент:
  ![page_G.jpg](page_G.jpg)
- Бинаризованный градиент:
  ![page_G_bin.jpg](page_G_bin.jpg)

#### Изображение `atlas.png`
- Исходное изображение:
  ![atlas.png](atlas.png)
- Полутоновое изображение:
  ![atlas_gray_image.jpg](atlas_gray_image.jpg)
- Градиент по оси X:
  ![atlas_Gx.jpg](atlas_Gx.jpg)
- Градиент по оси Y:
  ![atlas_Gy.jpg](atlas_Gy.jpg)
- Общий градиент:
  ![atlas_G.jpg](atlas_G.jpg)
- Бинаризованный градиент:
  ![atlas_G_bin.jpg](atlas_G_bin.jpg)

#### Изображение `cat.png`
- Исходное изображение:
  ![cat.png](cat.png)
- Полутоновое изображение:
  ![cat_gray_image.jpg](cat_gray_image.jpg)
- Градиент по оси X:
  ![cat_Gx.jpg](cat_Gx.jpg)
- Градиент по оси Y:
  ![cat_Gy.jpg](cat_Gy.jpg)
- Общий градиент:
  ![cat_G.jpg](cat_G.jpg)
- Бинаризованный градиент:
  ![cat_G_bin.jpg](cat_G_bin.jpg)

#### Изображение `cartoon.png`
- Исходное изображение:
  ![cartoon.png](cartoon.png)
- Полутоновое изображение:
  ![cartoon_gray_image.jpg](cartoon_gray_image.jpg)
- Градиент по оси X:
  ![cartoon_Gx.jpg](cartoon_Gx.jpg)
- Градиент по оси Y:
  ![cartoon_Gy.jpg](cartoon_Gy.jpg)
- Общий градиент:
  ![cartoon_G.jpg](cartoon_G.jpg)
- Бинаризованный градиент:
  ![cartoon_G_bin.jpg](cartoon_G_bin.jpg)
"""

with open('report_lab4_исп__.md', 'w', encoding='utf-8') as report_file:
    report_file.write(report_text)
