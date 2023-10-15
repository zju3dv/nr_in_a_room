# examples/python/visualization/interactive_visualization.py

import numpy as np
import copy
import open3d as o3d
import argparse


def demo_crop_geometry():
    print("Demo for manual geometry cropping")
    print("1) Press 'Y' twice to align geometry with negative direction of y-axis")
    print("2) Press 'K' to lock screen and to switch to selection mode")
    print("3) Drag for rectangle selection,")
    print("   or use ctrl + left click for polygon selection")
    print("4) Press 'C' to get a selected geometry and to save it")
    print("5) Press 'F' to switch to freeview mode")
    pcd = o3d.io.read_point_cloud("../../test_data/ICP/cloud_bin_0.pcd")
    o3d.visualization.draw_geometries_with_editing([pcd])


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def pick_points(pcd):
    print("")
    print("1) Please pick at least three correspondences using [shift + left click]")
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()


def demo_manual_registration(
    source_pcd_file: str,  # source pcd file
    source_mesh_file: str,  # source mesh file
    target_pcd_file: str,  # target pcd file
    target_mesh_file: str,  # target pcd file
    output_mesh_file: str,  # output mesh file
):
    print("Demo for manual ICP")
    if source_pcd_file is None:
        # sampling from tgt mesh
        source = o3d.io.read_triangle_mesh(source_mesh_file)
        source = source.sample_points_poisson_disk(
            number_of_points=10000, init_factor=5
        )
    else:
        source = o3d.io.read_point_cloud(source_pcd_file)

        cl, ind = source.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.1)
        source = source.select_by_index(ind)

    if target_pcd_file is None:
        # sampling from tgt mesh
        target = o3d.io.read_triangle_mesh(target_mesh_file)
        target = target.sample_points_poisson_disk(number_of_points=5000, init_factor=5)
    else:
        target = o3d.io.read_point_cloud(target_pcd_file)

    print("Visualization of two point clouds before manual alignment")
    draw_registration_result(source, target, np.identity(4))

    # pick points from two point clouds and builds correspondences
    picked_id_source = pick_points(source)
    picked_id_target = pick_points(target)
    assert len(picked_id_source) >= 3 and len(picked_id_target) >= 3
    assert len(picked_id_source) == len(picked_id_target)
    corr = np.zeros((len(picked_id_source), 2))
    corr[:, 0] = picked_id_source
    corr[:, 1] = picked_id_target

    # estimate rough transformation using correspondences
    print("Compute a rough transform using the correspondences given by user")
    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    p2p.with_scaling = True
    trans_init = p2p.compute_transformation(
        source, target, o3d.utility.Vector2iVector(corr)
    )

    # point-to-point ICP for refinement
    print("Perform point-to-point ICP refinement")
    threshold = 0.03  # 3cm distance threshold
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source,
        target,
        threshold,
        trans_init,
        # o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        p2p,
    )

    transformation = reg_p2p.transformation
    # transformation = trans_init
    draw_registration_result(source, target, transformation)

    print("transformation = \n", transformation)

    target_mesh = o3d.io.read_triangle_mesh(target_mesh_file)

    target_mesh.transform(np.linalg.inv(transformation))
    o3d.io.write_triangle_mesh(output_mesh_file, target_mesh)

    print("")


if __name__ == "__main__":
    """
    align tgt mesh to src pcd
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_pcd", default=None)
    parser.add_argument("--src_mesh", default=None)
    parser.add_argument("--tgt_pcd", default=None)
    parser.add_argument("--tgt_mesh", default=None)
    parser.add_argument("--output_mesh", default=None)
    args = parser.parse_args()
    demo_manual_registration(
        args.src_pcd, args.src_mesh, args.tgt_pcd, args.tgt_mesh, args.output_mesh
    )
