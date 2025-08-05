import os

# === Config ===
prefix = "CUDA_VISIBLE_DEVICES=3"
dataset_root = "/workspace/data/TensoIR_Synthetic"
output_root = "/workspace/work/Outputs/TensoIR_Synthetic"
light_name_list = ["bridge", "city", "fireplace", "forest", "night"]

scenes = ['armadillo', 'ficus', 'hotdog', 'lego']

for scene in scenes:
    dataset_name = scene

    scene_path = os.path.join(dataset_root, dataset_name)
    output_path = os.path.join(output_root, 'gigs', dataset_name)
    checkpoint_30k = os.path.join(output_path, "chkpnt30000.pth")
    checkpoint_35k = os.path.join(output_path, "chkpnt35000.pth")

    # === Commands ===
    commands = [
        # f"{prefix} python train.py -m {output_path} -s {scene_path} --iterations 35000 --eval --gamma --radius 0.8 --bias 0.01 --thick 0.05 --delta 0.0625 --step 16 --start 64 --indirect",
        # f"{prefix} python render.py -m {output_path} -s {scene_path} --checkpoint {checkpoint_35k} --eval --skip_train --pbr --gamma --indirect",
        # f"{prefix} python normal_eval.py --gt_dir {scene_path} --output_dir {output_path}/test/ours_None",
        # f"{prefix} python render.py -m {output_path} -s {scene_path} --checkpoint {checkpoint_35k} --eval --skip_train --brdf_eval",
    ]
    for light_name in light_name_list:
        envmap_path = os.path.join(dataset_root, f"Environment_Maps/high_res_envmaps_2k/{light_name}.hdr")
        commands.append(f"{prefix} python relight.py -m {output_path} -s {scene_path} --checkpoint {checkpoint_35k} --hdri {envmap_path} --eval --gamma")
    commands.append(f"{prefix} python relight_eval.py --output_dir {output_path}/test/ours_None/relight/ --gt_dir {scene_path}")

    # === Execution ===
    for cmd in commands:
        print(f"\n=== Running: {cmd} ===")
        os.system(cmd)
