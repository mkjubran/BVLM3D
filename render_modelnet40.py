import trimesh
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

def render_views_matplotlib(mesh_path, output_dir, views=8, image_size=224):
    mesh = trimesh.load(mesh_path)
    os.makedirs(output_dir, exist_ok=True)

    angles = np.linspace(0, 360, views, endpoint=False)
    for i, angle in enumerate(angles):
        # Rotate mesh around y-axis
        mesh.apply_transform(trimesh.transformations.rotation_matrix(
            np.radians(angle), [0, 1, 0], mesh.centroid
        ))

        fig = plt.figure(figsize=(image_size/100, image_size/100), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_axis_off()

        # Plot the mesh
        ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2], triangles=mesh.faces, color='gray', edgecolor='none', linewidth=0, antialiased=False)

        # Set limits
        scale = mesh.extents.max()
        mid = mesh.centroid
        ax.set_xlim(mid[0]-scale, mid[0]+scale)
        ax.set_ylim(mid[1]-scale, mid[1]+scale)
        ax.set_zlim(mid[2]-scale, mid[2]+scale)

        # Set viewing angle
        ax.view_init(elev=30, azim=30)

        # Save and close
        plt.savefig(f"{output_dir}/view_{i:02d}.png", bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        # Undo rotation for next angle
        mesh.apply_transform(trimesh.transformations.inverse_matrix(
            trimesh.transformations.rotation_matrix(np.radians(angle), [0, 1, 0], mesh.centroid)
        ))

def process_modelnet40_matplotlib(modelnet_dir, output_root, views_per_model=8, image_size=224):
    modelnet_dir = Path(modelnet_dir)
    output_root = Path(output_root)
    classes = sorted([d for d in modelnet_dir.iterdir() if d.is_dir()])
    
    for class_dir in classes:
        for split in ['train', 'test']:
            input_dir = class_dir / split
            if not input_dir.exists():
                continue

            for off_file in input_dir.glob("*.off"):
                relative = off_file.relative_to(modelnet_dir)
                target_dir = output_root / relative.parent / off_file.stem
                try:
                    render_views_matplotlib(str(off_file), str(target_dir), views=views_per_model, image_size=image_size)
                    print(f"Rendered: {off_file}")
                except Exception as e:
                    print(f"Failed to render {off_file}: {e}")

if __name__ == "__main__":
    modelnet_path = "./ModelNet40/ModelNet40"
    output_path = "./ModelNet40/ModelNet40_2D"

    process_modelnet40_matplotlib(modelnet_path, output_path, views_per_model=8, image_size=224)
