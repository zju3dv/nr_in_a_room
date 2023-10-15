# Neural Rendering in a Room: Data Preparation

## iGibson-Synthetic Dataset

We provide the raw scripts to make iGibson synthetic dataset (`data_gen/igibson_render_scene.py`).
You can also directly download the processed data_from this [link](https://www.dropbox.com/scl/fi/3ph4xnlte68mlzh70cg0l/igibson_synthetic_dataset.zip?rlkey=nm1fy1uqk8xcyy7df151wbfj1&dl=0).


## Fresh-Room Dataset

You can download the pre-processed files through this [link](https://www.dropbox.com/scl/fi/bsusfefufrw1l7vviz2s2/fresh_room_dataset_v2.zip?rlkey=5ioljipo0vdng8h29qabdc9by&dl=0).

## Initial Prediction for Camera and Object Poses

We provide the initial prediction of panoramic camera poses from HLoc and object poses from the object prediction module.
You can download the pre-processed files through this [link](https://www.dropbox.com/scl/fi/idfs124cxzauafgj10dkp/object_prediction.zip?rlkey=0z2jy1vvtemc95nhutvl51ebe&dl=0).

## Diffused Light Maps

We use lighting augmentation to enable the network's ability to adapt to unseen lighting conditions (Sec. 3.1.1).
You can generate the diffused light maps by running the `tools/batch_hdr_to_diffuse_map.py` or directly download the pre-processed diffused lgiht maps through this [link](https://www.dropbox.com/scl/fi/apza4g86e5ocx14fqj2dm/hdr_galary_100_diffuse.zip?rlkey=4hgm6gitk9ya96rb2wta9cief&dl=0).
