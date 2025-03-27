#!/bin/bash
python predict_bbox.py \
    --data_json "/mnt/pfs-mc0p4k/nlu/team/yuhaofu/data/LLaVA-CoT-100k/combined_results.json" \
    --output_file "/mnt/pfs-mc0p4k/nlu/team/yuhaofu/data/LLaVA-CoT-100k/fgeo_1_10_bbox.json" \
    --save_intermediate \
    --world_size 2 