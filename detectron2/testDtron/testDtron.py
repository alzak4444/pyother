import detectron2
import cv2
from matplotlib import pyplot as plt
import os
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
import random
from detectron2.utils.visualizer import Visualizer

cfg = get_cfg()
cfg.merge_from_file(
    "../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
)

cfg.MODEL.WEIGHTS = '../output/model_final.pth'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75   # set the testing threshold for this model
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # 3 classes (data, fig, hazelnut)

predictor = detectron2.engine.defaults.DefaultPredictor(cfg)
import torch
import time
plt.figure(figsize=(10,10))
#for d in random.sample(dataset_dicts, 1):    
#    im = cv2.imread(d["file_name"])
im = cv2.imread('../mydata/images/0025.png')
t0 = time.time()
outputs, tempInput = predictor(im)
model1 = predictor.model
traced_script_module = torch.jit.trace(model1, tempInput)
traced_script_module.save("traced.pt")

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
