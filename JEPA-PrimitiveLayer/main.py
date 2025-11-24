# from Utils.dataset_prep import download_dataset, prep_map_csv

# download_dataset(already_download=True)

# prep_map_csv("/Users/davestran/Desktop/CS-3331/World-Model-Project/5/exported_maps/local_maps")

# from JEPA_PrimitiveLayer.jepa_1 import PrimitiveLayer

# primtive_model = PrimitiveLayer()
# print(primtive_model)
# total_params = sum(p.numel() for p in primtive_model.parameters())
# print(f"Total parameters: {total_params:,}")

from ultralytics import YOLO
import torch

yolo = YOLO("yolov8s.pt")
backbone = yolo.model.model    # correct for YOLOv8 API
print(backbone)