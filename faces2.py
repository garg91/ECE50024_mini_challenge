from facenet_pytorch import MTCNN
from PIL import Image
import os
from pathlib import Path
import torch


def identify_and_save_largest_face_mtcnn(input_folder, output_folder):
    # Initialize MTCNN
    mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')

    # Iterate over all images in the input folder
    for img_name in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_name)
        image = Image.open(img_path).convert('RGB')

        # Detect faces in the image
        boxes, _ = mtcnn.detect(image)

        if boxes is None:
            face_path = os.path.join(output_folder, img_name)
            image.save(face_path)
            continue  # No faces found, move to the next image

        # Identify the largest face
        largest_face_area = 0
        largest_face_box = None
        for box in boxes:
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            if area > largest_face_area:
                largest_face_area = area
                largest_face_box = box

        if largest_face_box is not None:
            # Crop the image to the largest face
            cropped_face = image.crop(largest_face_box)
            # Save the cropped face with the same name as the original image
            face_path = os.path.join(output_folder, img_name)
            cropped_face.save(face_path)

input_folder = 'test'
output_folder = 'temp4'

identify_and_save_largest_face_mtcnn(input_folder, output_folder)
