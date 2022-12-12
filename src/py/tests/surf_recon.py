import open3d as o3d
import numpy as np


def reconstruct(file):
    assert file.endswith(".txt")
    points = np.loadtxt(file)
    print(points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals()
    pcd = o3d.geometry.PointCloud(pcd)
    print(pcd)
    # ply_name = f"{file[:-4]}.ply"
    # o3d.io.write_point_cloud(ply_name, pcd)
    # pcd = o3d.io.read_point_cloud(ply_name)
    # print(pcd)
    o3d.visualization.draw_geometries([pcd])
    mesh = o3d.geometry.TriangleMesh().create_from_point_cloud_poisson(pcd)[0]
    o3d.visualization.draw_geometries([mesh])
    print(mesh)
    o3d.io.write_triangle_mesh(f"{file[:-4]}.obj", mesh)


if __name__ == "__main__":
    reconstruct("mpm3d_outputs/00127.txt")
