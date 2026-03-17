python /nfs4/qhy/projects/threestudio/scripts/render_from_cameras_scene_txt.py \
  --combined_ply /nfs4/qhy/projects/threestudio/dataset/scene_resized_/scene_plus_car_tagged.ply \
  --cameras_txt /nfs4/qhy/projects/threestudio/dataset/camera_transpose_result/cameras_scene.txt \
  --images_txt  /nfs4/qhy/projects/threestudio/dataset/camera_transpose_result/images_scene.txt \
  --out_dir     /nfs4/qhy/projects/threestudio/test_datas/test_render_4 \
  --device cuda \
  --max_images 0 \
  --bg 0,0,0