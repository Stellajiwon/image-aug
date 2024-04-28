import os
import cv2
from imgaug import augmenters as iaa

# fire.jpg 파일 경로
fire_path = "fire.jpg"

fire_image = cv2.imread(fire_path)

# 이미지 어그멘테이션 적용
flipper = iaa.Fliplr(1.0)
fire_image_flipped = flipper.augment_image(fire_image)

vflipper = iaa.Flipud(0.9)
fire_image_vflipped = vflipper.augment_image(fire_image)

blurer = iaa.GaussianBlur(3.0)
fire_image_blurred = blurer.augment_image(fire_image)

translater = iaa.Affine(translate_px={"x": -16})
fire_image_translated = translater.augment_image(fire_image)

scaler = iaa.Affine(scale={"y": (0.8, 1.2)})
fire_image_scaled = scaler.augment_image(fire_image)

# 아웃풋 폴더 생성
output_folder = "output_images_fire"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 이미지 저장
cv2.imwrite(os.path.join(output_folder, "flipped_fire.jpg"), fire_image_flipped)
cv2.imwrite(os.path.join(output_folder, "vflipped_fire.jpg"), fire_image_vflipped)
cv2.imwrite(os.path.join(output_folder, "blurred_fire.jpg"), fire_image_blurred)
cv2.imwrite(os.path.join(output_folder, "translated_fire.jpg"), fire_image_translated)
cv2.imwrite(os.path.join(output_folder, "scaled_fire.jpg"), fire_image_scaled)
