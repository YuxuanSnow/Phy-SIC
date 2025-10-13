import dill
import torch
import trimesh
import pickle
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
from tqdm import tqdm
from pytorch3d.loss.chamfer import chamfer_distance
from pathlib import Path
from robust_loss_pytorch.adaptive import AdaptiveLossFunction
from torch import nn

from image_helpers import (
    get_mask_and_bbox,
    fill_masks_with_background_color,
    get_inpainted_image_omni,
    load_gsam,
    load_omni,
)
from scene import (
    extend_floor_plane,
    render_mesh_points,
    compute_outlier_mask,
    get_normal_outlier_mask,
    scale_points_to_human_mesh,
    compute_point_normals,
)
from utils.vis import (
    get_scene_with_normals,
    overlay_pose_on_image,
    get_colored_vertices,
    frames_to_gif,
)
from utils.depth import run_metric_moge_with_human_from_depthpro, load_dpro, load_moge
from utils.geometry import (
    project_points_torch,
    kNNField,
    rotmat_to_rot6d,
    rot6d_to_rotmat,
    pointmap_to_intrinsics,
)
from utils.utils import pack_points_into_batch

from models.hsfm.vitpose_inference import get_human_pose_2d_vitpose, load_vitpose

from human import (
    process_human,
    get_contact_probs,
    get_self_occlusion_masks,
    get_body_model,
    convert_smplx_to_coco_wholebody,
    get_point_visibility_mask,
    compute_camera_facing_mask,
    get_static_contacts,
    COCO_WHOLEBODY_SKELETON_INFO,
    COCO_WHOLEBODY_KEYPOINTS,
    coco_wholebody_left_hand_joint_indices,
    coco_wholebody_right_hand_joint_indices,
    load_chmr,
    load_deco,
    load_wilor,
)


class HumanScene(nn.Module):
    def __init__(
        self,
        cfg,
        image_path=None,
        output_path=None,
    ):
        super(HumanScene, self).__init__()
        self.image_path = image_path
        self.outpath = (
            Path("outputs") / Path(image_path).stem
            if output_path is None
            else Path(output_path)
        )
        self.outpath.mkdir(parents=True, exist_ok=True)
        self.body_model = get_body_model(cfg.body_model)
        self.body_model_type = cfg.body_model
        self.keypoint_threshold = cfg.keypoint_threshold
        self.max_distance_interpenetration = cfg.max_distance_interpenetration
        self.max_distance_contact = cfg.max_distance_contact
        self.self_occlusion_threshold = cfg.self_occlusion_threshold
        self.max_img_size = cfg.max_img_size
        self.cfg = cfg

        #########################################################################
        # initialize the images, and masks, bounding boxes, and the scene image.
        #########################################################################
        self.image = Image.open(image_path)

        if cfg.flip_image:
            self.image = self.image.transpose(Image.FLIP_LEFT_RIGHT)

        width, height = self.image.size
        if max(width, height) > self.max_img_size:
            scale = self.max_img_size / max(width, height)
            new_size = (int(width * scale), int(height * scale))
            print(f"Resizing the image from {self.image.size} to {new_size}.")
            self.image = self.image.resize(new_size, Image.LANCZOS)

        num_neighbours = int(min(self.image.size) * 0.05)
        print(f"Using {num_neighbours} neighbours for the outlier removal.")

        self.masks, self.boxes, self.confs, self.masks_org = get_mask_and_bbox(
            self.image
        )
        self.mask, self.mask_org = (
            np.stack(self.masks, axis=0).max(axis=0),
            np.stack(self.masks_org, axis=0).max(axis=0),
        )
        self.masks_floor = get_mask_and_bbox(
            self.image, text_prompt="ground, floor, carpet."
        )[-1]
        self.mask_floor = np.stack(self.masks_floor, axis=0).max(axis=0)

        if cfg.inpaint_model == "omni":
            self.scene_image = get_inpainted_image_omni(
                self.image, Image.fromarray(self.mask_org)
            )
            self.scene_image.save(self.outpath / "scene_image.png")
        else:
            raise ValueError("Only omni inpainting is supported currently.")

        self.image_np, self.scene_image_np = (
            np.array(self.image),
            np.array(self.scene_image),
        )
        self.width, self.height = self.image_np.shape[1], self.image_np.shape[0]

        if cfg.single_human:
            # if single human is requested, we only keep the most confident human.
            max_idx = np.argmax(self.confs)
            self.boxes = self.boxes[max_idx][None]
            self.confs = self.confs[max_idx][None]
            self.masks = [self.masks[max_idx]]
            self.masks_org = [self.masks_org[max_idx]]
        else:
            # otherwise, we keep all humans with confidence greater than 0.45.
            box_masks = self.confs > 0.45
            self.boxes = self.boxes[box_masks]
            self.confs = self.confs[box_masks]
            self.masks = [
                self.masks[i] for i in range(len(self.masks)) if box_masks[i]
            ]
            self.masks_org = [
                self.masks_org[i]
                for i in range(len(self.masks_org))
                if box_masks[i]
            ]
            assert len(self.boxes) > 0, "No humans detected in the image."

        ###########################################################
        # Initialize the human mesh, contact vertices, and 2D pose.
        ###########################################################
        self.n_humans = self.boxes.shape[0]
        keypoints_2d, scores_2d = get_human_pose_2d_vitpose(
            self.image_np, self.boxes
        )
        conf_2d = torch.tensor(scores_2d).float()
        conf_2d_thresh = conf_2d.clone()
        # if the confidence is less than the threshold, we set it to 0.
        conf_2d_thresh[conf_2d_thresh < self.keypoint_threshold] = 0
        self.register_buffer(
            "pose_2d", torch.tensor(keypoints_2d).float()
        )  # (n_humans, n_joints, 2)
        self.register_buffer("conf_2d", conf_2d)  # (n_humans, n_joints)
        self.register_buffer(
            "conf_2d_thresh", conf_2d_thresh
        )  # (n_humans, n_joints)

        for idxs in [
            coco_wholebody_left_hand_joint_indices,
            coco_wholebody_right_hand_joint_indices,
        ]:
            for human_idx in range(self.n_humans):
                if self.conf_2d_thresh[human_idx, idxs].mean() < 0.7:
                    self.conf_2d_thresh[human_idx, idxs] = 0
        # weight the body joints higher than rest.
        self.conf_2d_thresh[
            :, : COCO_WHOLEBODY_KEYPOINTS.index("right_heel") + 1
        ] *= 10
        if cfg.only_body_joints:
            self.conf_2d_thresh[
                :, COCO_WHOLEBODY_KEYPOINTS.index("right_heel") + 1 :
            ] = 0

        # run CHMR to get the human mesh, camera intrinsics, SMPL(X) parameters, and the output camera transformation.
        mesh, cam_intr, out_body_params, output_cam_trans, _, _ = process_human(
            self.image_np,
            body_model=cfg.body_model,
            boxes=self.boxes,
            confs=self.confs,
            keypoints=keypoints_2d,
            keypoint_scores=scores_2d,
            smpl_model=cfg.smpl_model,
        )

        assert (
            "betas" in out_body_params
        ), "Betas not found in the output body parameters."
        # create learnable parameters for the body parameters. For all the rotations, we use rot6d. (except for betas)
        self.body_params = nn.ParameterDict(
            {
                k: nn.Parameter(rotmat_to_rot6d(v) if "betas" not in k else v)
                for k, v in out_body_params.items()
            }
        )
        self.cam_trans = nn.Parameter(output_cam_trans)
        self.body_params_init = {
            k: torch.tensor(v.detach().clone().cuda())
            for k, v in self.body_params.items()
        }
        self.cam_trans_init = output_cam_trans.detach().clone().cuda()

        # initialize the contact vertices for each human.
        if cfg.contact == "deco":
            self.cont_vertices_conf = get_contact_probs(
                [self.image_np[:, :, ::-1]], [self.boxes], cfg.body_model
            )[0]
        else:
            self.cont_vertices_conf = get_static_contacts(
                "SMPLX", self.boxes.shape[0]
            )
        self.register_buffer(
            "cont_vertices", torch.tensor(self.cont_vertices_conf >= 0.5)
        )

        ##############################################################
        # initialize the scene depth map and the camera intrinsics.
        ##############################################################
        # depth, K = run_metric_moge_from_depthpro(self.scene_image_np, f_px = cam_intr[0, 0])
        if cfg.f_px == "chmr" or cfg.f_px == "moge_c":
            f_px_gt = np.array(cam_intr[0, 0])
        elif cfg.f_px == "dpro" or cfg.f_px == "moge":
            f_px_gt = None  # use depthpro to get the focal length.
        elif isinstance(cfg.f_px, (int, float)):
            f_px_gt = np.array(cfg.f_px)
        else:
            raise ValueError("f_px should be 'chmr', 'dpro', or a float/int value.")

        # run depthpro to get the depth map and the camera intrinsics.
        depth, depth_human, pts3d, pts3d_human, K = (
            run_metric_moge_with_human_from_depthpro(
                self.scene_image_np, self.image_np, self.mask_org, f_px=f_px_gt
            )
        )  # cam_intr[0, 0])
        self.register_buffer("depth", torch.from_numpy(depth).float())
        self.register_buffer("K", torch.from_numpy(K).float())
        if cfg.f_px == "moge" or cfg.f_px == "moge_c":
            self.register_buffer(
                "K", pointmap_to_intrinsics(torch.tensor(pts3d).cuda())
            )

        ######################################################################
        # initialize the scene points and scale, normals, and the floor plane.
        ######################################################################

        # unproject scene points to 3D, and scale it appropriately, to ensure occluded scene points are behind the human mesh.
        pts3d = pts3d  # depth_to_points(depth[None], K)
        scale = scale_points_to_human_mesh(mesh, pts3d, self.mask_org, K)

        self.scale = nn.Parameter(torch.tensor(1).float())
        self.scale_initial = scale
        self.register_buffer(
            "pts3d_all", torch.tensor(pts3d * scale).float().cuda()
        )
        # compute normals and floor plane for the scene points.
        all_normals = compute_point_normals(self.pts3d_all)

        # remove outliers in the scene point cloud
        self.inlier_mask_pts = ~(
            compute_outlier_mask(
                self.pts3d_all.cuda(), k=num_neighbours, std=1.45
            ).reshape(self.pts3d_all.shape[:-1])
        )
        pts3d = torch.tensor(pts3d * scale).float().cuda()[self.inlier_mask_pts]
        normals = all_normals[self.inlier_mask_pts]

        # remove outliers in the normals, and corresponding points.
        self.inlier_mask_normals = ~(
            get_normal_outlier_mask(
                pts3d, normals, k=num_neighbours, threshold_degrees=50
            )
        )
        self.register_buffer("pts3d", pts3d[self.inlier_mask_normals])
        self.to("cuda")
        self.register_buffer("normals", normals[self.inlier_mask_normals])

        # convert the normals to a boolean mask for the image.
        inlier_mask_normal_image = torch.zeros_like(
            self.inlier_mask_pts, dtype=torch.bool
        )
        inlier_mask_normal_image[self.inlier_mask_pts] = self.inlier_mask_normals
        self.inlier_mask = self.inlier_mask_pts & inlier_mask_normal_image

        print(
            f"Removed outliers from the normals: {self.normals.shape[0]} points left after filtering {(~self.inlier_mask_normals).sum()} outliers."
        )

        # For each human, compute the point cloud, and the inlier mask.
        pts_human = torch.tensor(pts3d_human * scale).float().cuda()
        pts_humans = [pts_human[m > 0] for m in self.masks_org]
        # Compute an inliers mask for each human's point cloud.
        pts_humans_inliers = [
            (
                ~compute_outlier_mask(p, k=num_neighbours, std=1.45).reshape(
                    p.shape[:-1]
                )
            )
            for p in pts_humans
        ]
        pts_humans = [
            p[pts_inliers] for p, pts_inliers in zip(pts_humans, pts_humans_inliers)
        ]
        pts_colors_humans = [
            torch.tensor(self.image_np[m > 0]).cuda()[pts_humans_inliers[idx]]
            for idx, m in enumerate(self.masks_org)
        ]

        pts_humans, pts_humans_lengths = pack_points_into_batch(pts_humans)
        pts_colors_humans, _ = pack_points_into_batch(pts_colors_humans)
        self.register_buffer("pts_humans", pts_humans)
        self.register_buffer("pts_human_colors", pts_colors_humans)
        self.register_buffer("pts_human_lengths", pts_humans_lengths)

        if cfg.compute_floor_points:
            # compute the floor plane for the scene points.
            self.plane_points, self.plane_normal, self.plane_d = extend_floor_plane(
                self.pts3d_all,
                all_normals,
                self.mask_floor & self.inlier_mask.cpu().numpy(),
            )
        else:
            print("Floor plane not computed, skipping.")
            self.plane_points, self.plane_normal, self.plane_d = (
                torch.zeros(0, 3, device=self.pts3d_all.device),
                torch.zeros(3, device=self.pts3d_all.device),
                0.0,
            )

        assert (
            self.normals.shape[0] == self.pts3d.shape[0]
        ), "The number of normals and points do not match."

        ############################################################################
        # Initialize the visibility mask for the human mesh vertices.
        ############################################################################
        # initialize the visibility mask for the mesh vertices.
        self.register_buffer(
            "mesh_visibility",
            torch.ones(self.n_humans, mesh.vertices.shape[0], dtype=torch.bool).cuda(),
        )
        self.register_buffer(
            "self_occlusion",
            torch.zeros(self.n_humans, mesh.vertices.shape[0], dtype=torch.bool).cuda(),
        )
        # initialize the camera facing mask for the mesh vertices.
        self.register_buffer(
            "camera_facing",
            torch.ones(self.n_humans, mesh.vertices.shape[0], dtype=torch.bool).cuda(),
        )

        #################################################################################
        # Initialize the knn field for the scene points for fast nearest neighbor search.
        #################################################################################
        _, _, pred_vertices, scene_pts, scene_floor = self()
        all_scene_points = torch.concatenate(
            [scene_pts.reshape(-1, 3), scene_floor.reshape(-1, 3)], dim=0
        )

        self.knn = kNNField(all_scene_points, 128)

        ######################################################################
        # Initialize the optimizer
        ######################################################################
        self.optimizer = None

        ##############################################################################
        # Handle occluded or truncated joints the mesh before the coarse optimization.
        ##############################################################################
        self.conf_2d_thresh = self.conf_2d_thresh * get_point_visibility_mask(
            self.mask, self.pose_2d
        )

        ######################################################################
        # Coarse optimization: optimize the camera translation and body betas.
        ######################################################################
        self.optimize(**cfg["opt_1"])

        ###########################################################################
        # Handle occluded or truncated mesh vertices after the coarse optimization.
        ###########################################################################
        if cfg.ray_casting:
            _, _, pred_vertices, _, _ = self()
            vertices_2d = project_points_torch(
                pred_vertices.reshape(-1, 3), self.K
            ).reshape(self.n_humans, -1, 2)
            # we use the dilated mask for visibility to avoid false negatives.
            self.register_buffer(
                "self_occlusion",
                torch.tensor(
                    get_self_occlusion_masks(
                        pred_vertices,
                        self.body_model.faces,
                        self.self_occlusion_threshold,
                    )
                ).to(self.mesh_visibility.device),
            )
            self.register_buffer(
                "mesh_visibility", get_point_visibility_mask(self.mask_org, vertices_2d)
            )
            self.register_buffer(
                "camera_facing",
                torch.tensor(
                    compute_camera_facing_mask(
                        pred_vertices.detach().cpu().numpy(),
                        self.body_model.faces,
                        angle_threshold=80,
                    )[
                        0
                    ]  # returns mask, and normals.
                ).to(self.mesh_visibility.device),
            )

            # set the camera facing mask to false for the occluded/truncated vertices.
            self.camera_facing[~self.mesh_visibility] = False

        print("Completed the initialization of the HumanScene module.")

        self.optimize(**cfg["opt_2"])
        self.optimize(**cfg["opt_3"])

    def forward(self):
        body_params_rotmat = {
            k: (rot6d_to_rotmat(v) if "betas" not in k else v)
            for k, v in self.body_params.items()
        }

        smpl_output = self.body_model(
            **{k: v.float() for k, v in body_params_rotmat.items()}
        )
        pred_keypoints_3d = smpl_output.joints + self.cam_trans.unsqueeze(1)
        pred_vertices = smpl_output.vertices + self.cam_trans.unsqueeze(1)

        scene_pts = self.pts3d * self.scale
        scene_floor = self.plane_points * self.scale

        pred_pose_2d = project_points_torch(
            pred_keypoints_3d.reshape(-1, 3), self.K
        )  # (n_humans * n_joints, 2)
        pred_pose_2d = pred_pose_2d.reshape(
            self.n_humans, -1, 2
        )  # (n_humans, n_joints, 2)

        return pred_keypoints_3d, pred_pose_2d, pred_vertices, scene_pts, scene_floor

    def compute_losses(
        self,
        iloss,
        closs,
        invisible_vertex_weight,
        orient_reg_weight,
        scale_reg_weight,
    ):
        # make a forward pass through the SMPL model with current paramters.
        pred_keypoints_3d, pred_pose_2d, pred_vertices, scene_pts, scene_floor = self()

        all_scene_points = torch.concatenate(
            [scene_pts.reshape(-1, 3), scene_floor.reshape(-1, 3)], dim=0
        )
        all_human_points = pred_vertices.reshape(-1, 3)
        human_depth_points = self.pts_humans * self.scale.detach()

        loss_dict = {}

        # compute the nearest neighbors for the human vertices.
        _, nn_idxs = self.knn(all_human_points, scale=self.scale)
        scene_normals = torch.cat(
            [
                self.normals.reshape(-1, 3),
                torch.tile(self.plane_normal, (self.plane_points.shape[0], 1)),
            ],
            dim=0,
        )
        nn_points, nn_normals = (
            all_scene_points[nn_idxs.squeeze()],
            scene_normals[nn_idxs.squeeze()],
        )

        nn_to_vertex = torch.nn.functional.normalize(
            all_human_points - nn_points, dim=-1
        )
        nn_normals = torch.nn.functional.normalize(nn_normals, dim=-1)
        dots = (nn_to_vertex * nn_normals).sum(dim=-1)
        distances = torch.linalg.norm(all_human_points - nn_points, dim=-1)

        ########################################################
        # Interpenetration loss.
        ########################################################

        # the vertex needs to be a) interpenetrated b) not occluded c) within threshold distance d) not a contact vertex.
        condition_i = (
            (dots < 0)
            & (self.mesh_visibility.reshape(-1))
            & (distances < self.max_distance_interpenetration)
            * (~(self.cont_vertices.reshape(-1)))
        )
        interpenetration_terms = (nn_points - all_human_points)[condition_i]
        loss_dict["interpenetration_loss"] = iloss.lossfun(interpenetration_terms).sum()

        ########################################################
        # Human-scene contact loss.
        ########################################################

        # the vertex needs to be a) not interpenetrated b) a contact vertex c) within threshold distance.
        condition_c = (
            (dots > 0)
            & (self.cont_vertices.reshape(-1))
            & (distances < self.max_distance_contact)
        )
        contact_terms = (nn_points - all_human_points)[condition_c]
        loss_dict["contact_loss"] = closs.lossfun(contact_terms).sum()

        #########################################################
        # Human depth loss.
        #########################################################
        # Each human is optimized separately, with the respective human depth points.
        camera_facing_vertices = []
        for human, human_vis in enumerate(self.camera_facing):
            camera_facing_vertices.append(pred_vertices[human][human_vis])
        camera_facing_vertices, vertex_lengths = pack_points_into_batch(
            camera_facing_vertices
        )

        loss_dict["human_depth_loss"] = chamfer_distance(
            camera_facing_vertices,
            human_depth_points,
            x_lengths=vertex_lengths,
            y_lengths=self.pts_human_lengths,
            batch_reduction="sum",
            point_reduction="sum",
            single_directional=False,
        )[0]

        ########################################################
        # 3D-2D projection loss.
        ########################################################
        pred_human_joints_coco = convert_smplx_to_coco_wholebody(pred_pose_2d)

        dists = pred_human_joints_coco - self.pose_2d
        dists[..., 0] = dists[..., 0] / self.width
        dists[..., 1] = dists[..., 1] / self.height
        proj_loss = torch.abs(dists) * self.conf_2d_thresh[..., None]

        loss_dict["proj_loss"] = proj_loss.mean()

        ########################################################
        # Regularization terms: against initial predictions:
        ########################################################
        reg_loss = 0.0
        body_params_rotmat = {
            k: (rot6d_to_rotmat(v) if "betas" not in k else v)
            for k, v in self.body_params.items()
        }
        body_params_init_rotmat = {
            k: (rot6d_to_rotmat(v) if "betas" not in k else v)
            for k, v in self.body_params_init.items()
        }

        assert (
            "global_orient" in body_params_rotmat
        ), "Global orientation not found in body_params_rotmat."

        # Apply regularization with respect to the initial predictions in root-relative space.
        combined_params = {
            k: torch.cat(
                (body_params_rotmat[k].float(), body_params_init_rotmat[k].float()),
                dim=0,
            )
            for k in body_params_rotmat
            if k != "global_orient"
        }
        combined_output = self.body_model(**combined_params)
        joints_3d = combined_output.vertices[: self.n_humans]
        joints_3d_init = combined_output.vertices[self.n_humans :]

        # use a higher weight for the vertices with object- or self-occlusion.
        reg_loss = (
            (joints_3d - joints_3d_init) ** 2
            * (
                ((~self.mesh_visibility[..., None]) | self.self_occlusion[..., None])
                * invisible_vertex_weight
            )
        ).sum()

        loss_dict["reg_loss"] = (
            reg_loss
            + ((self.cam_trans - self.cam_trans_init) ** 2).sum()
            + (
                (
                    self.body_params["global_orient"]
                    - self.body_params_init["global_orient"]
                )
                ** 2
            ).sum()
            * orient_reg_weight
            + ((self.scale - 1) ** 2 * scale_reg_weight)
        )

        return loss_dict

    def get_loss_closure(
        self,
        iloss,
        closs,
        invisible_vertex_weight,
        orient_reg_weight,
        scale_reg_weight,
        loss_weights,
        optimizer,
    ):
        def loss_closure():
            optimizer.zero_grad()
            loss_dict = self.compute_losses(
                iloss,
                closs,
                invisible_vertex_weight,
                orient_reg_weight,
                scale_reg_weight,
            )
            assert set(loss_dict.keys()) == set(
                loss_weights.keys()
            ), "The keys in loss_dict and loss_weights do not match."
            loss = sum([loss_dict[k] * loss_weights[k] for k in loss_weights.keys()])
            loss.backward()
            return loss

        return loss_closure

    def optimize(
        self,
        n_iter=100,
        check_self_occlusion_every=30,
        train_params=[
            "scale",
            "cam_trans",
            "body_params.betas",
            "body_params.body_pose",
            "body_params.global_orient",
            "body_params.left_hand_pose",
            "body_params.right_hand_pose",
        ],
        loss_weights={
            "contact_loss": 0.7,
            "interpenetration_loss": 0.3,
            "human_depth_loss": 0.005,
            "reg_loss": 45,
            "proj_loss": 10000,
        },
        invisible_vertex_weight=12,
        scale_reg_weight=1,
        orient_reg_weight=50,
        lr=1e-2,
        optimizer="adam",
    ):
        ########################################################
        # Initialize the optimizer and the loss functions.
        ########################################################

        iloss = AdaptiveLossFunction(
            num_dims=3, float_dtype=np.float32, device=torch.device("cuda:0")
        )
        closs = AdaptiveLossFunction(
            num_dims=3, float_dtype=np.float32, device=torch.device("cuda:0")
        )

        if train_params is None:
            train_params = [
                name for name, param in self.named_parameters() if param.requires_grad
            ]
        else:
            all_param_names = [name for name, _ in self.named_parameters()]
            assert all(
                name in all_param_names for name in train_params
            ), "Some of the train_params are not found in the module."

        print("Optimizing the HumanScene module with the following parameters:")
        for name, param in self.named_parameters():
            if name in train_params:
                print(f"{name}: {param.shape}")

        params_to_optimize = [
            param for name, param in self.named_parameters() if name in train_params
        ]
        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(params_to_optimize, lr=lr)
        elif optimizer == "lbfgs":
            self.optimizer = torch.optim.LBFGS(
                params_to_optimize,
                lr=1,
                max_iter=15,
                # line_search_fn='strong_wolfe'
            )
            closure = self.get_loss_closure(
                iloss,
                closs,
                invisible_vertex_weight,
                orient_reg_weight,
                scale_reg_weight,
                loss_weights,
                self.optimizer,
            )

        pbar = tqdm(range(n_iter))
        for idx in pbar:
            # check for self occlusion every few iterations.
            if (
                self.cfg.ray_casting
                and idx != 0
                and idx % check_self_occlusion_every == 0
            ):
                _, _, pred_vertices, _, _ = self()
                self.register_buffer(
                    "self_occlusion",
                    torch.tensor(
                        get_self_occlusion_masks(
                            pred_vertices,
                            self.body_model.faces,
                            self.self_occlusion_threshold,
                        )
                    ).to(self.mesh_visibility.device),
                )

            if optimizer == "adam":
                loss_dict = self.compute_losses(
                    iloss,
                    closs,
                    invisible_vertex_weight,
                    orient_reg_weight,
                    scale_reg_weight,
                )
                assert set(loss_dict.keys()) == set(
                    loss_weights.keys()
                ), "The keys in loss_dict and loss_weights do not match."
                # loss = contact_loss * c_w + interpenetration_loss * i_w + reg_loss * r_w + proj_loss * p_w
                loss = sum(
                    [loss_dict[k] * loss_weights[k] for k in loss_weights.keys()]
                )

                #########################################################
                # Backward pass and optimization step.
                #########################################################
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            elif optimizer == "lbfgs":
                loss = self.optimizer.step(closure)
                current_loss = loss.item()
                try:
                    prev_loss
                except NameError:
                    prev_loss = current_loss
                    no_improve = 0
                else:
                    no_improve = no_improve + (1 if current_loss >= prev_loss else 0)
                    prev_loss = current_loss
                if no_improve >= 5:
                    print(
                        "Loss did not improve in {} iterations. Stopping optimization.".format(
                            no_improve
                        )
                    )
                    break
                loss_dict = self.compute_losses(
                    iloss,
                    closs,
                    invisible_vertex_weight,
                    orient_reg_weight,
                    scale_reg_weight,
                )

            pbar.set_description(
                f"Loss: {loss.item():.2f};"
            )

    def coarse_optimize(
        self, n_iter=100, optimize_human_depth=False, optimizer="adam"
    ):
        loss_weights = {
            "contact_loss": 0,
            "interpenetration_loss": 0,
            "human_depth_loss": 0,
            "reg_loss": 0,
            "proj_loss": 1,
        }
        train_params = ["cam_trans"]  # , 'body_params.global_orient']

        if optimize_human_depth:
            # when optimizing human depth, disable the global orientation optimization.
            loss_weights["human_depth_loss"] = 0.6
            loss_weights["proj_loss"] = 5000
            train_params = ["cam_trans"]
            optimizer = "lbfgs"
            n_iter = 2

        return self.optimize(
            n_iter=n_iter,
            train_params=train_params,
            loss_weights=loss_weights,
            lr=1e-2,
            optimizer=optimizer,
        )
