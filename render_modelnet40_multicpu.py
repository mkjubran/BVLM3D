import trimesh
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

def render_views_matplotlib(mesh_path, output_dir, views=8, image_size=224):
    mesh = trimesh.load(mesh_path)
    os.makedirs(output_dir, exist_ok=True)

    angles = np.linspace(0, 360, views, endpoint=False)
    for i, angle in enumerate(angles):
        # Rotate mesh
        mesh.apply_transform(trimesh.transformations.rotation_matrix(
            np.radians(angle), [0, 1, 0], mesh.centroid
        ))

        fig = plt.figure(figsize=(image_size/100, image_size/100), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_axis_off()

        # Plot
        ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
                        triangles=mesh.faces, color='gray', edgecolor='none', linewidth=0)

        # Set limits
        scale = mesh.extents.max()
        mid = mesh.centroid
        ax.set_xlim(mid[0]-scale, mid[0]+scale)
        ax.set_ylim(mid[1]-scale, mid[1]+scale)
        ax.set_zlim(mid[2]-scale, mid[2]+scale)

        ax.view_init(elev=30, azim=30)

        # Save
        plt.savefig(f"{output_dir}/view_{i:02d}.png", bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        # Undo rotation
        mesh.apply_transform(trimesh.transformations.inverse_matrix(
            trimesh.transformations.rotation_matrix(np.radians(angle), [0, 1, 0], mesh.centroid)
        ))

def process_single_file(off_file, modelnet_dir, output_root, views_per_model, image_size):
    relative = off_file.relative_to(modelnet_dir)
    target_dir = output_root / relative.parent / off_file.stem

    if target_dir.exists() and len(list(target_dir.glob("*.png"))) >= views_per_model:
        return f"Skipped (exists): {off_file}"

    try:
        render_views_matplotlib(str(off_file), str(target_dir), views=views_per_model, image_size=image_size)
        return f"Rendered: {off_file}"
    except Exception as e:
        return f"Failed to render {off_file}: {e}"

def process_modelnet40_matplotlib(modelnet_dir, output_root, views_per_model=8, image_size=224, max_workers=8):
    modelnet_dir = Path(modelnet_dir)
    output_root = Path(output_root)
    tasks = []

    for class_dir in sorted([d for d in modelnet_dir.iterdir() if d.is_dir()]):
        for split in ['train', 'test']:
            input_dir = class_dir / split
            if not input_dir.exists():
                continue
            for off_file in input_dir.glob("*.off"):
                tasks.append((off_file, modelnet_dir, output_root, views_per_model, image_size))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_file, *t) for t in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            print(f"[{i}/{len(futures)}] {future.result()}")

if __name__ == "__main__":
    modelnet_path = "../ModelNet40/ModelNet40"
    output_path = "../ModelNet40/ModelNet40_2D"
    process_modelnet40_matplotlib(modelnet_path, output_path, views_per_model=8, image_size=224, max_workers=16)
