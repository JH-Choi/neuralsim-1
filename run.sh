#!/bin/sh
# cd ./dataio/autonomous_driving/waymo
# INPUT_FILE=/mnt/hdd/data/R3Live/hku_campus_seq_00/image_poses.txt
# OUTPUT_FOLDER=/mnt/hdd/data/R3Live/hku_campus_seq_00/colmap_format
# python waymo_to_colmap_format.py $INPUT_FILE $OUTPUT_FOLDER

# In omnidata conda environment
# export PATH=/usr/local/cuda-11.4/bin:$PATH
# export LD_LIBRARY_PATH=/mnt/hdd/conda/envs/omnidata/lib:$LD_LIBRARY_PATH
# cd ./dataio/autonomous_driving/waymo
# DATA_ROOT=/mnt/hdd/data/R3Live
# OMNI_PATH=/mnt/hdd/code/omnidata/omnidata_tools/torch/
# SCENE_ID=hku_campus_seq_00
# python extract_mono_cue_single.py --task=depth --data_root=$DATA_ROOT \
#     --omnidata_path=$OMNI_PATH --scene_id=$SCENE_ID
# python extract_mono_cue_single.py --task=normal --data_root=$DATA_ROOT \
#     --omnidata_path=$OMNI_PATH --scene_id=$SCENE_ID


# In segformer conda environment, Extract masks
# cd ./dataio/autonomous_driving/waymo
# DATA_ROOT=/mnt/hdd/data/R3Live
# SEG_FOLDER=/mnt/hdd/code/SegFormer
# SCENE_ID=hku_campus_seq_00
# python extract_mask_single.py --data_root=$DATA_ROOT --scene_id $SCENE_ID \
#  --segformer_path=$SEG_FOLDER --checkpoint=$SEG_FOLDER/segformer.b5.1024x1024.city.160k.pth 


# Preprocess (colmap format -> waymo format)
# cd ./dataio/autonomous_driving/custom
# DATA_ROOT=/mnt/hdd/data/R3Live
# # SEG_FOLDER=/mnt/hdd/code/SegFormer
# SCENE_ID=hku_campus_seq_00
# mkdir -p $DATA_ROOT/$SCENE_ID/waymo_format
# python preprocess.py --scene_id $SCENE_ID \
#  --model_dir $DATA_ROOT/$SCENE_ID/colmap_format --root_dir $DATA_ROOT/$SCENE_ID \
#  --save_dir $DATA_ROOT/$SCENE_ID/waymo_format 
