RUN=hyp_aug

python -O main.py \
  --model HypImprovedENet \
  --loss compound \
  --clip-grad 1.0 \
  --lr 0.001 \
  --epochs 40 \
  --warmup-epochs 3 \
  --dest results/segthor/$RUN \
  --gpu \
  --augment

# python viewer/viewer.py --img_source data/SEGTHOR/val/img \
# data/SEGTHOR/val/gt results/segthor/$RUN/iter000/val results/segthor/$RUN/best_epoch/val \
# -n 2 -C 5 --remap "{63: 1, 126: 2, 189: 3, 252: 4}" \
# --legend --class_names background esophagus heart trachea aorta --no_contour

# python stitch.py --data_folder results/segthor/$RUN/best_epoch/val \
# --dest_folder volumes/segthor/$RUN \
# --num_classes 255 --grp_regex "(Patient_\d\d)_\d\d\d\d" \
# --source_scan_pattern "data/segthor_part1/train/{id_}/GT.nii.gz"

# python metrics.py \
#     --data_folder results/segthor/$RUN/best_epoch/val \
#     --volumes_folder volumes/segthor/$RUN \
#     --target_pattern "data/segthor_part1/train/{id_}/GT.nii.gz" \
#     --grp_regex "(Patient_\d\d)_\d\d\d\d" \
#     --num_classes 5 \
#     --backend distorch \
#     --device cuda