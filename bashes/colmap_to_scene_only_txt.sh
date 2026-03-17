python /nfs4/qhy/projects/threestudio/scripts/colmap_to_scene_only_txt.py \
  --car_ply /nfs4/qhy/projects/threestudio/dataset/test_cars/car_only_clean_edited.ply \
  --insert_report /nfs4/qhy/projects/threestudio/dataset/new_combined_scene/insert_report.json \
  --colmap_sparse_dir /nfs4/qhy/projects/threestudio/dataset/car_images/sparse/0 \
  --out_dir /nfs4/qhy/projects/threestudio/dataset/camera_transpose_result \
  --car_scale 1.0 \
  --write_summary