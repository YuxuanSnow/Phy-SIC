import trimesh
import torch
import smplx
import math
import io
import imageio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from .constants import body17_connMat, body17_colors
from scene import points_to_mesh
from scipy.spatial import cKDTree
import pickle
from pathlib import Path

smplx_layer = smplx.SMPLXLayer(model_path='data/body_models/smplx/SMPLX_NEUTRAL.npz', num_betas=10, use_face_contour = True).cuda()

from .geometry import rot6d_to_rotmat

def get_scene_with_normals(
    pts3d, normals, pts_colors=None, scale=1, percentage=0.01, show_normal_arrows=True
):
    """
    Create a trimesh scene with point cloud and arrows representing normals.

    Parameters:
        pts3d     : torch.Tensor or numpy.ndarray of shape (H, W, 3)
                    3D points.
        normals   : torch.Tensor or numpy.ndarray of shape (H, W, 3)
                    Normal vectors corresponding to the points.
        scale     : float
                    A scale factor to adjust the arrow lengths.
        percentage: float, optional
                    Fraction of points to sample along each dimension (default 0.1 for 10%).

    Returns:
        scene_with_normals: trimesh.Scene
                    The scene containing the colored point cloud, arrows and an axis helper.
    """

    # Convert tensors to numpy arrays, if needed.
    # Convert tensors to numpy arrays, if needed.
    pts = pts3d.cpu().numpy() if hasattr(pts3d, "cpu") else pts3d
    normals_np = normals.cpu().numpy() if hasattr(normals, "cpu") else normals

    # Reshape to (N, 3)
    pts = pts.reshape(-1, 3)
    normals_np = normals_np.reshape(-1, 3)

    # Define normal colors by mapping normals from [-1, 1] to [0, 255]
    normal_colors = ((normals_np + 1) / 2 * 255).astype(np.uint8)

    arrow_length = 0.1 * scale  # length proportional to the scale
    arrows = []

    # Sample points uniformly from the flattened array.
    N = pts.shape[0]
    sample_idx = np.linspace(0, N - 1, max(1, int(N * percentage))).astype(int)
    for i in sample_idx:
        start = pts[i]
        direction = normals_np[i]
        end = start + arrow_length * direction
        line = np.stack([start, end], axis=0)
        arrows.append(line)

    pts_flat = pts
    colors_flat = normal_colors

    if pts_colors is not None:
        colors_flat = pts_colors.reshape(-1, 3)

    # Create a list of trimesh paths for each arrow with red color.
    arrow_paths = [
        trimesh.path.Path3D(
            entities=[trimesh.path.entities.Line(np.array([0, 1]))],
            vertices=line,
            colors=np.array([[255, 0, 0, 255]]),
        )
        for line in arrows
    ]

    print(len(arrow_paths), "arrows created.")
    print("Number of points:", len(pts_flat))

    # Create the colored point cloud.
    pc = trimesh.PointCloud(pts_flat, colors_flat)

    if show_normal_arrows:
        # Combine the point cloud, arrows, and an axis helper into a single scene.
        scene_with_normals = trimesh.Scene([pc] + arrow_paths)
    else:
        # Create a scene with just the point cloud and an axis helper.
        scene_with_normals = trimesh.Scene([pc])

    return scene_with_normals


def visualize_human_pose(joints, connectivity, edge_colors=None):
    """
    Creates a trimesh scene from joint positions, a connectivity matrix, and optional edge colors.

    Args:
        joints (np.ndarray): Array of shape (n_joints, 3) representing joint locations.
        connectivity (np.ndarray): Array of shape (n_connections, 2) representing joint pairs.
        edge_colors (np.ndarray, optional): Array of shape (n_connections, 3) or (n_connections, 4)
            containing colors for each edge. If shape is (n_connections, 3), an alpha channel (255) is added.
            If not provided, random colors are used.

    Returns:
        trimesh.Scene: A scene containing spheres for joints and cylinders for bones.
    """

    geometry_list = []

    # Add a sphere (dot) for each joint.
    for j in joints:
        sphere = trimesh.creation.icosphere(radius=0.01)
        sphere.apply_translation(j)
        geometry_list.append(sphere)

    # Helper function to create a cylinder between two 3D points.
    def cylinder_between(p1, p2, radius=0.005):
        vec = p2 - p1
        height = np.linalg.norm(vec)
        if height < 1e-6:
            return None
        direction = vec / height
        T = trimesh.geometry.align_vectors([0, 0, 1], direction)
        cylinder = trimesh.creation.cylinder(radius=radius, height=height, sections=16)
        cylinder.apply_transform(T)
        cylinder.apply_translation(p1 + vec * 0.5)
        return cylinder

    # Add bones with specified or random colors using the connectivity matrix.
    for idx, connection in enumerate(connectivity):
        i, j = connection
        start = joints[i]
        end = joints[j]
        bone = cylinder_between(start, end)
        if bone is None:
            continue

        if edge_colors is not None:
            # Use provided edge color.
            color = np.array(edge_colors[idx], dtype=np.uint8)
            if color.shape[0] == 3:
                color = np.append(color, 255)
        else:
            # Fallback to a random color with full opacity.
            color = np.array(
                [np.random.randint(0, 255) for _ in range(3)] + [255], dtype=np.uint8
            )

        bone.visual.face_colors = np.tile(color, (bone.faces.shape[0], 1))
        geometry_list.append(bone)

    # Create scene with geometry and add an axis for reference.
    scene = trimesh.Scene(geometry_list + [trimesh.creation.axis()])
    return scene


def overlay_pose_on_image(
    image,
    keypoints,
    connectivity=body17_connMat,
    edge_colors=body17_colors,
    mask=None,
    joint_labels=False,
):
    """
    Overlay pose keypoints on an image.
    This function creates a matplotlib figure sized to the input image dimensions,
    plots the skeleton defined by the connectivity pairs between keypoints with the
    corresponding line colors, and overlays the keypoints as yellow markers on the image.
    It then converts the plotted figure into a PIL Image with the pose overlay and returns it.
    Parameters:
        image (PIL.Image.Image): The background image on which the pose will be overlaid.
        keypoints (np.ndarray): An array of keypoint coordinates. Each row should contain at least
                                the x and y coordinates of a keypoint.
        connectivity (iterable of tuple): A sequence of tuples, where each tuple (j1, j2) indicates a
                                           connection between keypoints at index j1 and j2.
        edge_colors (iterable): A sequence of RGB color values (in 0-255 scale) for each connection line,
                                corresponding to the order of the connectivity list.
        joint_labels (bool): If True, the keypoint indices will be displayed next to the markers.
        mask (np.ndarray, optional): A binary mask of shape (n_keypoints,) to filter out keypoints to be displayed.
    Returns:
        PIL.Image.Image: A new image with the pose overlay, where the pose is drawn on top of the original image.
    """
    # create a figure with size matching the image dimensions

    if isinstance(connectivity, dict):
        ctx = connectivity
        connectivity = [ctx[x]["link"] for x in ctx]
        edge_colors = [ctx[x]["color"] for x in ctx]

    dpi = 100
    figsize = (image.width / dpi, image.height / dpi)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(image)
    for i, (j1, j2) in enumerate(connectivity):
        pt1 = keypoints[int(j1)][:2]
        pt2 = keypoints[int(j2)][:2]
        if mask is not None:
            if mask[int(j1)] == 0 or mask[int(j2)] == 0:
                continue
        color = np.array(edge_colors[i]) / 255.0
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color=color, linewidth=2)
    ax.scatter(keypoints[:, 0][mask], keypoints[:, 1][mask], c="yellow", s=20)

    if joint_labels:
        for idx, kp in enumerate(keypoints):
            ax.text(
                kp[0],
                kp[1],
                str(idx),
                color="white",
                fontsize=8,
                bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=1),
            )

    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    fig.canvas.draw()
    overlayed = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    overlayed = overlayed.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    overlayed = overlayed[..., :3]
    plt.close(fig)
    return Image.fromarray(overlayed)


def get_colored_vertices(
    conf,
    body_model,
    color_channel=1,
    mesh_path=None,
    mesh=None,
    binary=False,
    base_color=None,
):
    """
    Generate a colored mesh by modulating the specified color channel of each vertex based on the given confidence values.
    Parameters:
        conf (Tensor or numpy.ndarray): Confidence values for each vertex (expected to be normalized between 0 and 1).
        body_model (str): The body model to use (either 'SMPLX' or 'SMPL') for selecting the appropriate default mesh.
        color_channel (int, optional): Index of the mesh vertex color channel to modulate based on confidence. Default is 1.
        mesh_path (str, optional): Path to the mesh file. Used if no pre-loaded mesh is provided.
        mesh (trimesh.Trimesh, optional): Pre-loaded mesh object. If provided, mesh_path is ignored.
        binary (bool, optional): If True, the color channel is set to 255 for confidence >= 0.5, otherwise it is set to 0. If false, the color channel is set to the confidence value (scaled to 0-255).

    Returns:
        trimesh.Trimesh: The mesh with updated vertex colors where the specified channel is modulated by the confidence values.
    """

    if mesh is not None:
        human_mesh = mesh
    else:
        if mesh_path is None:
            if body_model == "SMPLX":
                mesh_path = "data/body_models/smplx/smplx_neutral_tpose.ply"
            elif body_model == "SMPL":
                mesh_path = "data/body_models/smpl/smpl_neutral_tpose.ply"

        human_mesh = trimesh.load(mesh_path, process=False)

    if base_color is not None:
        base_color = np.array(base_color, dtype=np.uint8)
        if base_color.shape[0] == 3:
            base_color = np.concatenate([base_color, np.array([255], dtype=np.uint8)])
        human_mesh.visual.vertex_colors = np.tile(
            base_color, (human_mesh.vertices.shape[0], 1)
        )

    # Convert confidences to a numpy array (values between 0 and 255)
    conf_np = conf.cpu().numpy() if hasattr(conf, "cpu") else conf
    conf_np = (conf_np * 255).astype(np.uint8)
    # Create new vertex colors: modulate the green channel based on the confidence.
    # colors = np.zeros((human_mesh.vertices.shape[0], 4), dtype=np.uint8)

    if not binary:
        colors = human_mesh.visual.vertex_colors
        colors[:, color_channel] = conf_np[: human_mesh.vertices.shape[0]]
        colors[:, 3] = 255
    else:
        colors = human_mesh.visual.vertex_colors
        colors[:, 3] = 127
        mask = conf_np[: human_mesh.vertices.shape[0]] >= 0.5
        update = np.zeros((mask.sum(), 4), dtype=np.uint8)
        update[:, color_channel] = 255
        update[:, 3] = 255
        colors[mask] = update

    human_mesh.visual.vertex_colors = colors

    return human_mesh


def frames_to_gif(all_frames, fps=5, loop=0):
    """
    Converts a sequence of image frames into an animated GIF stored in an in-memory bytes buffer.
    Parameters:
        all_frames (iterable): A collection of image frames represented as NumPy arrays.
        fps (int, optional): Frames per second for the resulting GIF. Defaults to 5.
        loop (int, optional): Number of times the GIF should loop (0 means infinite looping). Defaults to 0.
    Returns:
        io.BytesIO: A BytesIO object containing the GIF image data.
    Notes:
        - If the frame pixel values are normalized (i.e., maximum value <= 1.0), they are scaled to 0-255.
        - Each frame is duplicated to slow down the animation.
    """

    frames = []
    for frame in all_frames:
        if frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)
        frames.extend([frame] * 2)

    gif_buffer = io.BytesIO()
    imageio.mimsave(gif_buffer, frames, fps=fps, format="GIF", loop=loop)
    gif_buffer.seek(0)

    return gif_buffer


def get_scene_mesh(frame_path, load_floor_points: bool = True, separate_human_scene: bool = False, max_faces: int = 100000):
    """
    Generate a textured 3D mesh from depth and floor point predictions for a given scene.

    This function reads prediction data from a pickle file and an image from the specified frame_path.
    It computes a point cloud from the predicted depth and scales it, then processes floor points from the plane.
    The function combines these points and their corresponding colours (from the scene image and a fixed floor colour)
    to generate mesh vertices, faces, and vertex colours using the points_to_mesh utility.

    Parameters:
        frame_path (Path): Path to the folder containing "scene_data_final.pkl" and "scene_image.png".

    Returns:
        tuple: A tuple (vertices, faces, colours) where:
            vertices (np.ndarray): Combined 3D coordinates of the mesh vertices.
            faces (np.ndarray): Indices forming triangular faces of the mesh.
            colours (np.ndarray): Corresponding colours for each vertex.
    """

    with open(frame_path / "scene_data_final.pkl", "rb") as f:
        final_predictions = pickle.load(f)

    scene_image = np.array(Image.open(frame_path / "scene_image.png"))
    scale = final_predictions['scale']

    H, W = np.array(scene_image).shape[:2]
    points = np.full((H, W, 3), np.nan, dtype=np.float32)
    points[final_predictions['inlier_mask']] = final_predictions['pts3d'] * scale

    points_list = [points]
    colours_list = [scene_image]
    masks_list = [final_predictions['inlier_mask']]

    if load_floor_points:
        points_floor = final_predictions['plane_points'] * scale
        num_points_floor_sqrt = int(round(math.sqrt(points_floor.shape[0])))
        colours_floor = np.tile(np.array([210, 255, 210], dtype=np.uint8), (num_points_floor_sqrt**2, 1)).reshape(num_points_floor_sqrt, num_points_floor_sqrt, 3)
        points_floor.resize((num_points_floor_sqrt, num_points_floor_sqrt, 3))

        points_list.append(points_floor)
        colours_list.append(colours_floor)
        masks_list.append(None)

    vertices, faces, colours = points_to_mesh(points_list, colours_list, masks_list=masks_list)
    
    vertices = np.concatenate(vertices, axis=0)
    faces = np.concatenate(faces, axis=0)
    colours = np.concatenate(colours, axis=0)

    if faces.shape[0] > max_faces:
        print(f"Decimating mesh from {faces.shape[0]} to {max_faces} faces")
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=(colours * 255).astype(np.uint8))
        # Store the original vertex positions and color information before decimation.
        original_vertices = mesh.vertices.copy()
        original_colors = mesh.visual.vertex_colors.copy() if hasattr(mesh.visual, "vertex_colors") else None
        # Simplify the mesh.
        mesh = mesh.simplify_quadric_decimation(face_count=min(max_faces, faces.shape[0]), aggression=9)
        vertices = mesh.vertices
        faces = mesh.faces
        # Reapply the original colors (adjusting for any lost vertices using nearest neighbor lookup)
        if original_colors is not None:
            tree = cKDTree(original_vertices)
            dists, indices = tree.query(vertices, k=1)
            colours = original_colors[indices, :3].astype(np.float32) / 255.0

    body_params_final = {
        k: (rot6d_to_rotmat(torch.tensor(v).float())
            if 'betas' not in k else torch.tensor(v).float())
        for k, v in final_predictions['body_params'].items()
    }
    output_final = smplx_layer(
        **{k: torch.tensor(v).float().cuda() for k, v in body_params_final.items()},
        transl=torch.tensor(final_predictions['cam_trans']).float().cuda()  # Added .cuda() here
    )

    faces_human = []
    vertices_human = []
    colours_human = []

    for i in range(len(output_final.vertices)):
        untransformed_final_verts = output_final.vertices[i].detach().cpu().numpy()

        faces_human.append(smplx_layer.faces + i * untransformed_final_verts.shape[0])
        vertices_human.append(untransformed_final_verts)
        colours_human.append(np.tile(np.array([(188/255), (188/255), (188/255)], dtype=np.float32), (untransformed_final_verts.shape[0], 1)))

    faces_human = np.concatenate(faces_human, axis=0)
    vertices_human = np.concatenate(vertices_human, axis=0)
    colours_human = np.concatenate(colours_human, axis=0)

    if separate_human_scene:
        return vertices, faces, colours, vertices_human, faces_human, colours_human
    else:
        # Combine human and scene meshes
        faces = np.concatenate([faces, faces_human + vertices.shape[0]], axis=0)
        vertices = np.concatenate([vertices, vertices_human], axis=0)
        colours = np.concatenate([colours, colours_human], axis=0)

    return vertices, faces, colours


def get_scene(frame_path, separate_human_scene=False, max_faces=100000):
    # Ensure the frame_path is a pathlib.Path object
    if not isinstance(frame_path, Path):
        frame_path = Path(frame_path)
    
    if separate_human_scene:
        vertices, faces, colours, vertices_human, faces_human, colours_human = get_scene_mesh(frame_path, load_floor_points=False, separate_human_scene=True, max_faces=max_faces)

        vertex_colors = (colours * 255).astype(np.uint8)
        vertex_colors_human = (colours_human * 255).astype(np.uint8)

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors)
        mesh_human = trimesh.Trimesh(vertices=vertices_human, faces=faces_human, vertex_colors=vertex_colors_human)
        return mesh.scene(), mesh_human.scene()
    
    # Get vertices, faces, and colours from the scene mesh
    else:
        vertices, faces, colours = get_scene_mesh(frame_path, load_floor_points=False, max_faces=max_faces)
    
        # Scale colours from [0,1] to [0,255] and convert to uint8
        vertex_colors = (colours * 255).astype(np.uint8)
        
        # Create a trimesh mesh and return its scene for visualization
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors)
        return mesh.scene()
