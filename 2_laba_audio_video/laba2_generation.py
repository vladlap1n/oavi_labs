import numpy as np
from PIL import Image, ImageDraw, ImageFont


def generate_test_image(width=400, height=300):
    image = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(image)
    for y in range(height):
        for x in range(width):
            r = int(255 * (x / width))
            g = int(255 * (y / height))
            b = int(255 * ((x + y) / (width + height)))
            draw.point((x, y), fill=(r, g, b))

    font = ImageFont.load_default()
    draw.text((50, 50), "Adaptive Binarization", fill="black", font=font)
    draw.text((50, 100), "Bradley-Roth Method", fill="black", font=font)
    noise = np.random.randint(0, 64, (height, width, 3), dtype=np.uint8)
    image = Image.fromarray(np.array(image) + noise)

    return image


test_image = generate_test_image()
test_image.save("test_image.png")