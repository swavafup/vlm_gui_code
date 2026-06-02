from PIL import Image
from pathlib import Path

input_folder = Path("non_crack")      # folder with 512x512 images
output_folder = Path("non_crack_224")   # folder to save 224x224 images

output_folder.mkdir(parents=True, exist_ok=True)

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for img_path in input_folder.rglob("*"):
    if img_path.suffix.lower() in image_extensions:
        # Keep same subfolder structure
        relative_path = img_path.relative_to(input_folder)
        save_path = output_folder / relative_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img_resized = img.resize((224, 224), Image.Resampling.LANCZOS)
            img_resized.save(save_path)

        print(f"Resized: {img_path} -> {save_path}")

print("All images resized to 224x224.")