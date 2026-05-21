import numpy as np
import mujoco


def get_projection_matrix(cam_id, model, width, height):
    """Calculates the intrinsic camera matrix K from MuJoCo fovy."""
    fovy = model.cam_fovy[cam_id]
    f = 0.5 * height / np.tan(fovy * np.pi / 360)
    K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]])
    return K


def project_point(p_3d, K, R_world_cam, t_world_cam):
    """Projects a 3D world point to 2D image coordinates and camera-frame depth."""
    p_cam = R_world_cam.T @ (p_3d - t_world_cam)
    p_img = K @ p_cam
    if p_img[2] <= 0:
        return None, -1.0
    return p_img[:2] / p_img[2], float(p_cam[2])


def is_allowed_action_chain(b_id, model):
    """
    Checks if a body belongs to the active manipulation chain (Right Arm + Fingers + Spine Anchor).
    """
    if b_id <= 0:
        return False
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b_id)
    name_l = name.lower()

    if "right" in name_l or name.startswith("R_"):
        if any(k in name_l for k in ["thigh", "calf", "knee", "foot", "toe"]):
            return False
        return True

    allowed_spine = ["torso_link", "waist_roll_link", "waist_pitch_link"]
    if name in allowed_spine:
        return True

    return False


def get_cube_edges(model, data, obj_name="box"):
    """
    Returns the 3D world coordinates of the 12 edges of the target cube.
    """
    try:
        obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, obj_name)
    except:
        return []

    # Get object pose
    pos = data.geom_xpos[obj_id]
    mat = data.geom_xmat[obj_id].reshape(3, 3)
    size = model.geom_size[obj_id]  # [half_x, half_y, half_z]

    # Define the 8 corners of the cube in local coordinates
    corners_local = (
        np.array(
            [
                [-1, -1, -1],
                [1, -1, -1],
                [1, 1, -1],
                [-1, 1, -1],
                [-1, -1, 1],
                [1, -1, 1],
                [1, 1, 1],
                [-1, 1, 1],
            ]
        )
        * size
    )

    # Transform corners to world coordinates
    corners_world = [pos + mat @ c for c in corners_local]

    # Define the 12 edges (indices of the corners)
    edge_indices = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),  # Bottom
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),  # Top
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),  # Verticals
    ]

    edges = [(corners_world[start], corners_world[end]) for start, end in edge_indices]
    return edges
