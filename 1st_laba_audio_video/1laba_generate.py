import cv2
import numpy as np

def generate_test_image(output_path):
    # Создаем изображение размером 256x256 с тремя каналами (RGB)
    height, width = 256, 256
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Добавляем градиент по красному каналу
    for i in range(height):
        image[i, :, 2] = i  # Красный канал (R)

    # Добавляем вертикальные полосы по зеленому и синему каналам
    for j in range(width):
        if j % 64 < 32:
            image[:, j, 1] = 255  # Зеленый канал (G)
        if j % 128 < 64:
            image[:, j, 0] = 255  # Синий канал (B)

    # Сохраняем изображение
    cv2.imwrite(output_path, image)
    print(f"Test image saved to {output_path}")

# Генерация тестового изображения
generate_test_image("test_image.png")