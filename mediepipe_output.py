from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
import torch
import os
import cv2
from tqdm import tqdm

def mediapipe_detection(input_folder):
    # 获取输入文件夹中所有图像文件的文件名
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    for image_file in tqdm(image_files, desc="Processing Images"):
        input_path = os.path.join(input_folder, image_file)
        image = cv2.imread(input_path)

        if image is not None:
            base_options = python.BaseOptions(model_asset_path='../pose_landmarker_heavy.task')
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                output_segmentation_masks=True)
            detector = vision.PoseLandmarker.create_from_options(options)

            # STEP 3: Load the input image.
            loaded_image = mp.Image.create_from_file(input_path)

            # STEP 4: Detect pose landmarks from the input image.
            detection_result = detector.detect(loaded_image)
            detection_result.pose_landmarks[0]
            # STEP 5: Process the detection result. In this case, visualize it.
            # annotated_image = draw_landmarks_on_image(loaded_image.numpy_view(), detection_result)

            # segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
            # visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
            #
            # image1 = image.astype(np.uint16)
            # image2 = visualized_mask.astype(np.uint16)
            # result = np.clip(image1 + image2, 0, 255).astype(np.uint8)
            #
            # output_path = os.path.join(output_folder, 'masked_' + image_file)
            # cv2.imwrite(output_path, result)
        else:
            print(f"Failed to read image: {image_file}")

if __name__ == '__main__':
    sequence_idx = 13

    i = sequence_idx
    dataset = torch.load(os.path.join('data/dataset_work/3DPW/', 'test' + '.pt'))
    name_sequence = dataset['name'][i]
    image_folder = os.path.join('../imageFiles/imageFiles/', name_sequence)
    mediapipe_detection(image_folder)