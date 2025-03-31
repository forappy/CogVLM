#!/bin/bash
# python predict_bbox.py \
#     --data_json "/mnt/pfs-mc0p4k/nlu/team/yuhaofu/data/LLaVA-CoT-100k/json/unpre_data.json" \
#     --output_file "/mnt/pfs-mc0p4k/nlu/team/yuhaofu/data/LLaVA-CoT-100k/json/unpre_data_bbox.json" \
#     --save_intermediate \
#     --world_size 4 


python bbox_filter.py \
    --data_json "/mnt/pfs-mc0p4k/nlu/team/yuhaofu/data/LLaVA-CoT-100k/json/filter_entire_image_caption.json" \
    --output_file "/mnt/pfs-mc0p4k/nlu/team/yuhaofu/data/LLaVA-CoT-100k/json/agent_filter_bbox.json" \
    --world_size 2 