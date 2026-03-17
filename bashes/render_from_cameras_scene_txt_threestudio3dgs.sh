python /nfs4/qhy/projects/threestudio/scripts/render_from_cameras_scene_txt_threestudio3dgs_noshading_v29.py \
  --repo_root /nfs4/qhy/projects/threestudio \
  --combined_ply /nfs4/qhy/projects/threestudio/dataset/new_combined_scene/scene_plus_car_tagged.ply \
  --cameras_txt /nfs4/qhy/projects/threestudio/dataset/camera_transpose_result/cameras_scene.txt \
  --images_txt  /nfs4/qhy/projects/threestudio/dataset/camera_transpose_result/images_scene.txt \
  --out_dir     /nfs4/qhy/projects/threestudio/test_datas/test_render_4 \
  --device cuda \
  --bg 0,0,0
