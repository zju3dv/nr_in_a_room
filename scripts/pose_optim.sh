#!/bin/bash

config=test/config/real_room_0_emb_a.yml
obj_prediction_json=data/object_prediction/real_data_room_0/ours_arr_1/arrangement1_wb/000077/pred.json
seg_path=data/object_prediction/real_data_room_0/ours_arr_1/arrangement1_wb/000077/seg.png
img_path=data/real_room_0/arrangement_panorama_select/arrangement1/000077.png
arrangement_name=arrangement1

python test/test_optim_pano.py  config=$config "img_wh=[320,180]" obj_prediction_json=$obj_prediction_json test_image_path=$img_path seg_mask_path=$seg_path prefix=real_room_0_arr_1_000077 arrangement_name=$arrangement_name
