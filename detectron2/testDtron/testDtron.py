from detectron2.data.datasets import register_coco_instances
import detectron2
import cv2
from matplotlib import pyplot as plt
import random
from detectron2.utils.visualizer import Visualizer

register_coco_instances("fruits_nuts", {}, "./mydata/trainval.json", "./mydata/images")

fruits_nuts_metadata = detectron2.data.MetadataCatalog.get("fruits_nuts")

dataset_dicts = detectron2.data.DatasetCatalog.get("fruits_nuts")


for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=fruits_nuts_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    a = vis.get_image()[:, :, ::-1]
    plt.imshow(a)


from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os

cfg = get_cfg()
cfg.merge_from_file(
    "./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
)

cfg.DATASETS.TRAIN = ("fruits_nuts",)
cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.02
cfg.SOLVER.MAX_ITER = (
    300
)  # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
    128
)  # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # 3 classes (data, fig, hazelnut)


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
cfg.DATASETS.TEST = ("fruits_nuts", )
predictor = detectron2.engine.defaults.DefaultPredictor(cfg)

from detectron2.utils.visualizer import ColorMode
import time

for d in random.sample(dataset_dicts, 1):    
    im = cv2.imread(d["file_name"])
    t0 = time.time()
    outputs = predictor(im)
    t1 = time.time()
    v = Visualizer(im[:, :, ::-1],
                   metadata=fruits_nuts_metadata, 
                   scale=0.8, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    t2 = time.time()
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    a = v.get_image()[:, :, ::-1]
    plt.imshow(a)
    print(f'{t2-t1},{t1-t0}')


