import numpy as np
from PIL import Image, ImageDraw, ImageFont

def create_test_image(width=400, height=400):

    image = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(image)

    draw.rectangle([50, 50, 150, 150], fill=0)

    draw.ellipse([200, 50, 300, 150], fill=128)

    draw.ellipse([100, 10, 150, 200], fill=64)

    for y in range(160, 200, 5):
        draw.line([(50, y), (350, y)], fill=0)

    for x in range(50, 350, 10):
        draw.line([(x, 210), (x, 310)], fill=0)

    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except IOError:
        font = ImageFont.load_default()
    draw.text((50, 320), "Barabulka", fill=0, font=font)
    draw.text((100, 370), "I love books", fill=50, font=font)
    noise = np.random.randint(0, 64, (height, width), dtype=np.uint8)
    noise_image = Image.fromarray(noise, mode="L")
    image = Image.blend(image, noise_image, alpha=0.3)

    return image

test_image = create_test_image()
test_image.save("test_image.png")
test_image.show(title="Тестовое изображение")