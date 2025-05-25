from pickletools import uint8

import cv2
import numpy as np
import os

def extract_rgb_components(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not open or find the image at {image_path}")
        return

    # Создаем пустые изображения того же размера, что и исходное
    red_img = np.zeros_like(img)
    green_img = np.zeros_like(img)
    blue_img = np.zeros_like(img)

    # Копируем соответствующий канал, оставляя остальные нулевыми
    red_img[:, :, 2] = img[:, :, 2]   # Красный канал в BGR
    green_img[:, :, 1] = img[:, :, 1]  # Зеленый канал в BGR
    blue_img[:, :, 0] = img[:, :, 0]   # Синий канал в BGR

    cv2.imwrite("red_component.png", red_img)
    cv2.imwrite("green_component.png", green_img)
    cv2.imwrite("blue_component.png", blue_img)
    print("RGB components extracted and saved.")
    return img

def rgb_to_hsi(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not open or find the image at {image_path}")
        return

    img = img.astype(np.float32) / 255.0

    blue = img[:, :, 0]
    green = img[:, :, 1]
    red = img[:, :, 2]

    intensity = (red + green + blue) / 3.0

    min_val = np.minimum(np.minimum(red, green), blue)
    saturation = 1 - 3 * min_val / (red + green + blue + 1e-6)
    saturation[saturation < 0] = 0

    numerator = 0.5 * ((red - green) + (red - blue))
    denominator = np.sqrt((red - green) ** 2 + (red - blue) * (green - blue) + 1e-6)
    hue = np.arccos(numerator / denominator)

    hue[blue > green] = 2 * np.pi - hue[blue > green]
    hue = hue / (2 * np.pi)

    intensity = np.uint8(intensity * 255)
    cv2.imwrite("intensity_component.png", intensity)

    print("HSI conversion complete. Intensity component saved.")

def invert_intensity(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not open or find the image at {image_path}")
        return

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted_img = cv2.bitwise_not(gray_img)

    cv2.imwrite("inverted_intensity.png", inverted_img)
    print("Intensity inverted and saved.")

def stretch_image(image_path, scale_factor_x, scale_factor_y,output_path=""):
    # Загружаем изображение
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not open or find the image at {image_path}")
        return

    # Получаем размеры исходного изображения
    height, width = img.shape[:2]
    print(height, width)
    # Вычисляем новые размеры изображения
    new_width = int(width * scale_factor_x)
    new_height = int(height * scale_factor_y)
    print(new_width,new_height)
    # Создаем пустое изображение для результата
    stretched_img = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    # Растягиваем изображение
    for i in range(new_height):
        for j in range(new_width):
            # Вычисляем координаты в исходном изображении
            src_i = i / scale_factor_y
            src_j = j / scale_factor_x

            # Округляем координаты до ближайшего целого
            src_i_rounded = int(round(src_i))
            src_j_rounded = int(round(src_j))

            # Проверяем, чтобы координаты не выходили за пределы исходного изображения
            src_i_rounded = min(src_i_rounded, height-1)
            src_j_rounded = min(src_j_rounded, width-1)

            # Копируем пиксель из исходного изображения в новое
            stretched_img[i, j] = img[src_i_rounded, src_j_rounded]

    # Сохраняем результат
    cv2.imwrite(output_path, stretched_img)
    print(f"Image stretched by factor of {scale_factor_x}x{scale_factor_y} and saved.")
    print(stretched_img.shape[:2])
    return stretched_img

def decimate_image(image_path, factor, output_path=""):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not open or find the image at {image_path}")
        return

    height, width = img.shape[:2]
    new_width = int(width / factor)
    new_height = int(height / factor)

    decimated_img = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    for i in range(new_height):
        for j in range(new_width):
            src_i = int(i * factor)
            src_j = int(j * factor)
            decimated_img[i, j] = img[src_i, src_j]

    cv2.imwrite(output_path, decimated_img)
    print(f"Image decimated by factor of {factor} and saved as {output_path}.")
    return decimated_img

def perediscretisation_two_pass(image_path, scale_factor_x,scale_factor_y):
    size_of_begin=stretch_image(image_path, scale_factor_x, scale_factor_x,"stretched_image2.png")
    miumiu="stretched_image2.png"
    size_of_end=decimate_image(miumiu, scale_factor_y, "resampled_two_pass.png")
    return size_of_end

def perediscretisation_one_pass(image_path, scale_factor_x, scale_factor_y):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not open or find the image at {image_path}")
        return
    height, width = img.shape[:2]
    new_width = int(width * scale_factor_x)
    new_height = int(height * scale_factor_y)
    resampled_img = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    for i in range(new_height):
        for j in range(new_width):
            src_i = int(i / scale_factor_y)
            src_j = int(j / scale_factor_x)
            resampled_img[i, j] = img[src_i, src_j]

    cv2.imwrite("resampled_image_one_pass.png", resampled_img)
    print(f"Image resampled by factor of {scale_factor_x}x{scale_factor_y} in one pass and saved.")
    return resampled_img

image_path = "test_image.png"

if not os.path.exists(image_path):
    print(f"Error: The image at {image_path} does not exist.")
else:
    img_main=extract_rgb_components(image_path)
    rgb_to_hsi(image_path)
    invert_intensity(image_path)

    M = 4
    N = 2
    K = 2

    stretch=stretch_image(image_path, M, M,"stretched_image.png")
    print("------")
    print(stretch.shape[:2])
    decimate=decimate_image(image_path, N, "decimated_image.png")
    two_pass=perediscretisation_two_pass(image_path, M,N)
    one_pass=perediscretisation_one_pass(image_path, K, K)

    # Получаем размеры изображений
    def get_image_size(image_path):
        img = cv2.imread(image_path)
        if img is not None:
            return img.shape[1], img.shape[0]  # width, height
        return None

    red_size = get_image_size("red_component.png")
    green_size = get_image_size("green_component.png")
    blue_size = get_image_size("blue_component.png")
    intensity_size = get_image_size("intensity_component.png")
    inverted_intensity_size = get_image_size("inverted_intensity.png")
    stretched_size = get_image_size("stretched_image.png")
    print("------")
    print(stretched_size)
    decimated_size = get_image_size("decimated_image.png")
    two_pass_size = get_image_size("resampled_two_pass.png")
    one_pass_size = get_image_size("resampled_image_one_pass.png")

    report_text = f"""# Отчет по лабораторной работе

## 1. Выделение компонент R, G, B
Были разделены каналы изображения. Ниже представлены результаты:
- Канал R: ![Red Component](red_component.png) (Размер: {red_size[0]}x{red_size[1]})
- Канал G: ![Green Component](green_component.png) (Размер: {green_size[0]}x{green_size[1]})
- Канал B: ![Blue Component](blue_component.png) (Размер: {blue_size[0]}x{blue_size[1]})

## 2. Преобразование изображения в HSI
Ниже представлены компоненты:
- Яркостная компонента: ![Intensity Component](intensity_component.png) (Размер: {intensity_size[0]}x{intensity_size[1]})
- Инвертированная яркостная компонента: ![Inverted Intensity](inverted_intensity.png) (Размер: {inverted_intensity_size[0]}x{inverted_intensity_size[1]})

## 3. Изменение размера изображения

### 3.1 Изначальные размеры изображения
Размер изображения: {img_main.shape[1]}x{img_main.shape[0]}

### 3.2 Растяжение (интерполяция)
Коэффициент растяжения: M = {M}  
Результат растяжения: ![Stretched Image](stretched_image.png) (Размер: {stretched_size[0]}x{stretched_size[1]})

### 3.3 Сжатие (децимация)
Коэффициент сжатия: N = {N}  
Результат сжатия: ![Decimated Image](decimated_image.png) (Размер: {decimated_size[0]}x{decimated_size[1]})

### 3.4 Передискретизация в два прохода (растяжение + сжатие)
Коэффициент: K = M / N = {K}  
Результат передискретизации в два прохода: ![Resampled Two Pass](resampled_two_pass.png) (Размер: {two_pass_size[0]}x{two_pass_size[1]})

### 3.5 Передискретизация в один проход
Коэффициент: K = {K}  
Результат передискретизации в один проход: ![Resampled One Pass](resampled_image_one_pass.png) (Размер: {one_pass_size[0]}x{one_pass_size[1]})
"""

    with open('report_lab1_исп__.md', 'w', encoding='utf-8') as report_file:
        report_file.write(report_text)