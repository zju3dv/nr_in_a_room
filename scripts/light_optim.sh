#!/bin/bash

config=test/config/real_room_0_emb_a.yml
test_image_path=data/real_room_0/arrangement_panorama_select/arrangement1/000077.png
arrangement_name=arrangement1
scene_info_json=data/real_room_0/scene/full/data.json
state_file=debug/xxx/000480.state.ckpt

python test/test_light_adaptation.py  config=$config "img_wh=[320,180]" prefix=light_optim scene_info_json=$scene_info_json state_file=$state_file arrangement_name=$arrangement_name
