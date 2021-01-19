python demo/demo.py --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml 	--input mydata/images/0001.png --confidence-threshold 0.8 --opts MODEL.WEIGHTS ./output/model_final.pth MODEL.ROI_HEADS.NUM_CLASSES 3
rem pause
rem start output.jpg

