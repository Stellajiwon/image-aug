import os

import cv2
import numpy as np

from imgaug import augmenters as iaa

# quokka.jpg 파일 경로
quokka_path = "quokka.jpg"

# quokka 이미지 로드
quokka_image = cv2.imread(quokka_path)
quokka_image = cv2.cvtColor(quokka_image, cv2.COLOR_BGR2RGB)  # OpenCV는 BGR 채널 순서를 사용하므로 RGB로 변환

# 이미지 어그멘테이션 적용
flipper = iaa.Fliplr(1.0)
quokka_image_flipped = flipper.augment_image(quokka_image)

vflipper = iaa.Flipud(0.9)
quokka_image_vflipped = vflipper.augment_image(quokka_image)

blurer = iaa.GaussianBlur(3.0)
quokka_image_blurred = blurer.augment_image(quokka_image)

translater = iaa.Affine(translate_px={"x": -16})
quokka_image_translated = translater.augment_image(quokka_image)

scaler = iaa.Affine(scale={"y": (0.8, 1.2)})
quokka_image_scaled = scaler.augment_image(quokka_image)

# 아웃풋 폴더 생성
output_folder = "output_images"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 이미지 저장
cv2.imwrite(os.path.join(output_folder, "flipped_quokka.jpg"), cv2.cvtColor(quokka_image_flipped, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(output_folder, "vflipped_quokka.jpg"), cv2.cvtColor(quokka_image_vflipped, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(output_folder, "blurred_quokka.jpg"), cv2.cvtColor(quokka_image_blurred, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(output_folder, "translated_quokka.jpg"), cv2.cvtColor(quokka_image_translated, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(output_folder, "scaled_quokka.jpg"), cv2.cvtColor(quokka_image_scaled, cv2.COLOR_RGB2BGR))