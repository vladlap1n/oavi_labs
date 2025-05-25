import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import matplotlib.pyplot as plt


def generate_characters():
    try:
        font_paths = [
            "timesi.ttf",
            "timesbi.ttf",
            "C:/Windows/Fonts/timesi.ttf",
            "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Italic.ttf"
        ]

        font = None
        for path in font_paths:
            try:
                font = ImageFont.truetype(path, 52)
                break
            except:
                continue

        if font is None:
            raise FileNotFoundError("Не найден курсивный шрифт Times New Roman")

        alphabet = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
        output_dir = "symbols_output"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("Генерация символов:")
        for char in alphabet:
            img = Image.new('L', (150, 150), 255)
            draw = ImageDraw.Draw(img)
            draw.text((20, 20), char, font=font, fill=0)

            bbox = img.getbbox()
            if bbox:
                img = img.crop(bbox)
                img.save(f"{output_dir}/{char}.png")
                print(f"  ✓ {char}", end=' ', flush=True)
            else:
                print(f"  ✗ {char} (пусто)", end=' ', flush=True)

        print("\nВсе символы сгенерированы!")
        return True

    except Exception as e:
        print(f"\nОшибка генерации символов: {str(e)}")
        return False


def calculate_features(image_path):
    try:
        img = Image.open(image_path)
        img_array = np.array(img)
        binary = (img_array < 128).astype(np.uint8)
        height, width = binary.shape

        if np.sum(binary) == 0:
            return None

        h_half = height // 2
        w_half = width // 2
        quarters = [
            binary[:h_half, :w_half],
            binary[:h_half, w_half:],
            binary[h_half:, :w_half],
            binary[h_half:, w_half:]
        ]
        weights = [np.sum(q) for q in quarters]

        quarter_areas = [q.size for q in quarters]
        relative_weights = [w / (a + 1e-6) for w, a in zip(weights, quarter_areas)]

        y, x = np.where(binary == 1)
        center_y = np.mean(y)
        center_x = np.mean(x)

        norm_center_y = center_y / (height + 1e-6)
        norm_center_x = center_x / (width + 1e-6)

        Iy = np.sum((x - center_x) ** 2)
        Ix = np.sum((y - center_y) ** 2)

        norm_Iy = Iy / ((width ** 2) * binary.sum() + 1e-6)
        norm_Ix = Ix / ((height ** 2) * binary.sum() + 1e-6)

        profile_x = binary.sum(axis=0)
        profile_y = binary.sum(axis=1)

        return {
            'Символ': os.path.basename(image_path)[0],
            'Вес Q1': weights[0],
            'Вес Q2': weights[1],
            'Вес Q3': weights[2],
            'Вес Q4': weights[3],
            'Уд.вес Q1': relative_weights[0],
            'Уд.вес Q2': relative_weights[1],
            'Уд.вес Q3': relative_weights[2],
            'Уд.вес Q4': relative_weights[3],
            'Центр X': center_x,
            'Центр Y': center_y,
            'Норм.центр X': norm_center_x,
            'Норм.центр Y': norm_center_y,
            'Момент X': Ix,
            'Момент Y': Iy,
            'Норм.момент X': norm_Ix,
            'Норм.момент Y': norm_Iy,
            'Ширина': width,
            'Высота': height,
            'profile_x': profile_x,
            'profile_y': profile_y
        }

    except Exception as e:
        print(f"Ошибка обработки {image_path}: {str(e)}")
        return None


def save_profiles(features, output_dir):
    try:
        char = features['Символ']

        plt.close('all')

        plt.figure(figsize=(8, 3))
        plt.bar(range(len(features['profile_x'])), features['profile_x'], color='blue')
        plt.title(f'X профиль символа {char}')
        plt.xlabel('Позиция по горизонтали')
        plt.ylabel('Интенсивность')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{char}_x_profile.png", dpi=100)
        plt.close()

        plt.figure(figsize=(8, 3))
        plt.bar(range(len(features['profile_y'])), features['profile_y'], color='green')
        plt.title(f'Y профиль символа {char}')
        plt.xlabel('Позиция по вертикали')
        plt.ylabel('Интенсивность')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{char}_y_profile.png", dpi=100)
        plt.close()

        print(f"  ✓ {char}", end=' ', flush=True)
    except Exception as e:
        print(f"  ✗ {char} (ошибка профилей)", end=' ', flush=True)


def generate_report(features_list, output_dir):
    try:
        valid_features = [f for f in features_list if f is not None]

        df = pd.DataFrame(valid_features)

        df_report = df.drop(columns=['profile_x', 'profile_y'])

        csv_path = f"{output_dir}/features.csv"
        df_report.to_csv(csv_path, index=False, sep=';', encoding='utf-8-sig')

        report = """# Лабораторная работа №5. Выделение признаков символов\n\n"""
        report += "## Вариант 25: Русские заглавные курсивные буквы\n\n"

        report += "## Все сгенерированные символы\n\n"
        report += "<div style='display: flex; flex-wrap: wrap; gap: 10px;'>\n"
        for features in valid_features:
            char = features['Символ']
            report += f"  <div style='text-align: center;'>\n"
            report += f"    <img src='{char}.png' alt='{char}' width='50'>\n"
            report += f"    <div>{char}</div>\n"
            report += f"  </div>\n"
        report += "</div>\n\n"

        report += "## Профили символов\n\n"
        for features in valid_features:
            char = features['Символ']
            report += f"### Символ {char}\n\n"
            report += "<div style='display: flex; gap: 20px; margin-bottom: 20px;'>\n"
            report += f"  <div>\n"
            report += f"    <img src='{char}_x_profile.png' alt='X профиль' width='300'>\n"
            report += f"  </div>\n"
            report += f"  <div>\n"
            report += f"    <img src='{char}_y_profile.png' alt='Y профиль' width='300'>\n"
            report += f"  </div>\n"
            report += "</div>\n\n"

        report += "## Таблица признаков\n\n"
        report += "Полные данные доступны в [файле features.csv](features.csv)\n\n"
        report += df_report.to_markdown(index=False)

        report_path = f"{output_dir}/report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"\nОтчёт успешно сгенерирован:")
        print(f"- Markdown отчёт: {report_path}")
        print(f"- Таблица признаков: {csv_path}")
        print(f"- Изображения символов: {output_dir}/*.png")
        print(f"- Графики профилей: {output_dir}/*_profile.png")

    except Exception as e:
        print(f"Ошибка генерации отчёта: {str(e)}")


def main():
    print("=== Лабораторная работа №5 ===")
    print("Выделение признаков русских заглавных курсивных букв\n")

    if not generate_characters():
        return

    print("\nРасчёт признаков и сохранение профилей:")
    features_list = []
    output_dir = "symbols_output"

    for char in "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ":
        img_path = f"{output_dir}/{char}.png"
        if os.path.exists(img_path):
            features = calculate_features(img_path)
            if features:
                features_list.append(features)
                save_profiles(features, output_dir)
            else:
                print(f"  ✗ {char} (ошибка расчёта)", end=' ', flush=True)
        else:
            print(f"  ✗ {char} (нет файла)", end=' ', flush=True)

    if features_list:
        generate_report(features_list, output_dir)
    else:
        print("\nОшибка: не удалось обработать ни одного символа")


if __name__ == "__main__":
    main()