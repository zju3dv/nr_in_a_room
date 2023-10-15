#!/bin/bash

function train_obj() {
    local config_file=$1
    local exp_name=$2
    local training_obj_ids=$3
    local val_obj_ids=$4
    
    python train_obj.py config=$config_file exp_name=$exp_name "dataset.training_obj_ids=$training_obj_ids" "dataset.val_obj_ids=$val_obj_ids"
}

#---------real room-------
train_obj "config/real_room_0_objs.yml" "real_room_0_bg_train_batch_10000_bs_1024"

# 1: box, 2: shelf, 3: chair, 4: desk, 5: nightstand
# 30 series use rendered depth

train_obj "config/real_room_0_objs.yml" "real_room_0_objs_31_PE_10_0_good_mask" "[31]" "[31]"
train_obj "config/real_room_0_objs.yml" "real_room_0_objs_32_PE_10_0_good_mask" "[32]" "[32]"
train_obj "config/real_room_0_objs.yml" "real_room_0_objs_33_PE_10_0_good_mask" "[33]" "[33]"
train_obj "config/real_room_0_objs.yml" "real_room_0_objs_34_PE_10_0_good_mask" "[34]" "[34]"
train_obj "config/real_room_0_objs.yml" "real_room_0_objs_35_PE_10_0_good_mask" "[35]" "[35]"

# uncomment to train other scenes
exit 0

# ----------lobby----------------
# training_obj_ids: [4, 5, 28, 29, 30, 31, 32, 33, 34, 35, 95, 96]
train_obj "config/igibson_Beechwood_0_int_lobby_0_bg.yml" "ig_lobby_bg_shrink_1.0"
train_obj "config/igibson_Beechwood_0_int_lobby_0.yml" "ig_lobby_objs_4_5_28_29" "[4,5,28,29]" "[4,5]"
train_obj "config/igibson_Beechwood_0_int_lobby_0.yml" "ig_lobby_objs_30_31_32_33" "[30,31,32,33]" "[30,31]"
train_obj "config/igibson_Beechwood_0_int_lobby_0.yml" "ig_lobby_objs_34_35_95_96" "[34,35,95,96]" "[34,35]"


# ----------home_office----------------
#   training_obj_ids: [6, 7, 8, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
train_obj "config/igibson_Pomaria_0_int_home_office_0.yml" "ig_home_office_bg_shrink_1.0" 
train_obj "config/igibson_Pomaria_0_int_home_office_0.yml" "ig_home_office_6_7_8_22" "[6,7,8,22]" "[8, 22]"
train_obj "config/igibson_Pomaria_0_int_home_office_0.yml" "ig_home_office_23_24_25_26" "[23,24,25,26]" "[23,24]"
train_obj "config/igibson_Pomaria_0_int_home_office_0.yml" "ig_home_office_27_28_29_30_31" "[27,28,29,30,31]" "[27,28]"
train_obj "config/igibson_Pomaria_0_int_home_office_0.yml" "ig_home_office_27_28_retrain" "[27,28]" "[27,28]"

# ----------childs_room----------------
# training_obj_ids: [3, 6, 7, 8, 9, 10, 11, 47, 48, 49, 50]
train_obj "config/igibson_Merom_1_int_childs_room_0.yml" "ig_childs_room_bg_shrink_1.0"
train_obj "config/igibson_Merom_1_int_childs_room_0.yml" "ig_childs_room_objs_3_6_7_8" "[3,6,7,8]" "[3,6]"
train_obj "config/igibson_Merom_1_int_childs_room_0.yml" "ig_childs_room_objs_9_10_11_47" "[9,10,11,47]" "[9,11]"
train_obj "config/igibson_Merom_1_int_childs_room_0.yml" "ig_childs_room_objs_48_49_50" "[48,49,50]" "[48,49]"


# ----------bedroom----------------
# training_obj_ids: [3, 4, 5, 6, 7, 8, 9, 10, 45, 48, 49, 50, 51, 62, 64]
train_obj "config/igibson_Beechwood_1_int_bedroom_0.yml" "ig_bedroom_bg_shrink_1.0"
train_obj "config/igibson_Beechwood_1_int_bedroom_0.yml" "ig_bedroom_objs_3_4_5_6" "[3,4,5,6]" "[3,4]"
train_obj "config/igibson_Beechwood_1_int_bedroom_0.yml" "ig_bedroom_objs_7_8_9_10" "[7,8,9,10]" "[7,8]"
train_obj "config/igibson_Beechwood_1_int_bedroom_0.yml" "ig_bedroom_objs_45_48_49_50" "[45,48,49,50]" "[45,48]"
train_obj "config/igibson_Beechwood_1_int_bedroom_0.yml" "ig_bedroom_objs_51_62_64" "[51,62,64]" "[51,62]"
train_obj "config/igibson_Beechwood_1_int_bedroom_0.yml" "ig_bedroom_objs_48" "[48]" "[48]"





