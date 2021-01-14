import detectron2
import cv2
from matplotlib import pyplot as plt
import os
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
import random
from detectron2.utils.visualizer import Visualizer

if __name__ == "__main__":
    detectron2.data.datasets.register_coco_instances("wires", {}, "./mydata/coco.json", "./mydata/images")
    wires_metadata = detectron2.data.MetadataCatalog.get("wires")
    dataset_dicts = detectron2.data.DatasetCatalog.get("wires")

    for d in random.sample(dataset_dicts,1):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=wires_metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        a = vis.get_image()[:, :, ::-1]
        plt.imshow(a)

