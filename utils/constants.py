# Reference:
# https://github.com/Arthur151/ROMP/blob/00c43337855c41fffae7ea552dda9be05cacd282/src/constants.py#L12

import json
import torch
import numpy as np


#####################################################################
# SMPLX Constants
#####################################################################
# load the body segmentation mask
with open("data/body_models/smplx/smplx_vert_segmentation.json", "r") as f:
    vert_segmentation = json.load(f)

# Define the segments for the upper and lower limbs
upper_arm = ["Shoulder", "Arm"]
lower_arm = ["ForeArm"]
hand = ["Hand", "HandIndex1"]
upper_leg = ["UpLeg"]
lower_leg = ["Leg"]
foot = ["Foot", "ToeBase"]

parts = [upper_arm, lower_arm, hand, upper_leg, lower_leg, foot]
part_names = ["upper_arm", "lower_arm", "hand", "upper_leg", "lower_leg", "foot"]

part_segments = []
part_segment_names = []

for side in ["left", "right"]:
    for part in parts:
        verts = []
        for seg in part:
            verts.extend(vert_segmentation[f"{side}{seg}"])
        part_segments.append(verts)
        part_segment_names.append(f"{side}_{part}")

assert len(part_segments) == 12

# Define unique colors for each of the six part segments as RGBA
part_segment_colors = [
    np.array([255, 0, 0, 255], dtype=np.uint8),  # red
    np.array([0, 255, 0, 255], dtype=np.uint8),  # green
    np.array([0, 0, 255, 255], dtype=np.uint8),  # blue
    np.array([255, 255, 0, 255], dtype=np.uint8),  # yellow
    np.array([255, 0, 255, 255], dtype=np.uint8),  # magenta
    np.array([0, 255, 255, 255], dtype=np.uint8),  # cyan
    np.array([255, 165, 0, 255], dtype=np.uint8),  # orange
    np.array([128, 0, 128, 255], dtype=np.uint8),  # purple
    np.array([0, 128, 128, 255], dtype=np.uint8),  # teal
    np.array([128, 128, 0, 255], dtype=np.uint8),  # olive
    np.array([255, 192, 203, 255], dtype=np.uint8),  # pink
    np.array([128, 128, 128, 255], dtype=np.uint8),  # gray
]


#####################################################################
# 2D Joint Indices
#####################################################################


body25_connMat = np.array(
    [
        0,
        1,
        1,
        2,
        2,
        3,
        3,
        4,
        1,
        5,
        5,
        6,
        6,
        7,
        1,
        8,
        8,
        9,
        9,
        10,
        10,
        11,
        8,
        12,
        12,
        13,
        13,
        14,
        0,
        15,
        15,
        17,
        0,
        16,
        16,
        18,
        14,
        19,
        19,
        20,
        14,
        21,
        11,
        22,
        22,
        23,
        11,
        24,
    ]
).reshape(-1, 2)
body17_connMat = (
    np.array(
        [
            [16, 14],
            [14, 12],
            [17, 15],
            [15, 13],
            [12, 13],
            [6, 12],
            [7, 13],
            [6, 7],
            [6, 8],
            [7, 9],
            [8, 10],
            [9, 11],
            [2, 3],
            [1, 2],
            [1, 3],
            [2, 4],
            [3, 5],
            [4, 6],
            [5, 7],
        ]
    )
    - 1
)


def joint_mapping(source_format, target_format):
    mapping = np.ones(len(target_format), dtype=np.int32) * -1
    for joint_name in target_format:
        if joint_name in source_format:
            mapping[target_format[joint_name]] = source_format[joint_name]
    return np.array(mapping)


joint_weights_17 = torch.ones(17).float()
joint_weights_17[[11, 12]] = (
    0  # the weights for the hip joints are set to 0 since they are ambiguous
)

# COCO
COCO_17 = {
    "Nose": 0,
    "L_Eye": 1,
    "R_Eye": 2,
    "L_Ear": 3,
    "R_Ear": 4,
    "L_Shoulder": 5,
    "R_Shoulder": 6,
    "L_Elbow": 7,
    "R_Elbow": 8,
    "L_Wrist": 9,
    "R_Wrist": 10,
    "L_Hip": 11,
    "R_Hip": 12,
    "L_Knee": 13,
    "R_Knee": 14,
    "L_Ankle": 15,
    "R_Ankle": 16,
}

OpenPose_25 = {
    "Nose": 0,
    "Neck": 1,
    "R_Shoulder": 2,
    "R_Elbow": 3,
    "R_Wrist": 4,
    "L_Shoulder": 5,
    "L_Elbow": 6,
    "L_Wrist": 7,
    "Pelvis": 8,
    "R_Hip": 9,
    "R_Knee": 10,
    "R_Ankle": 11,
    "L_Hip": 12,
    "L_Knee": 13,
    "L_Ankle": 14,
    "R_Eye": 15,
    "L_Eye": 16,
    "R_Ear": 17,
    "L_Ear": 18,
    "L_Toe": 19,
    "L_Foot": 20,
    "L_Heel": 21,
    "R_Toe": 22,
    "R_Foot": 23,
    "R_Heel": 24,
}

# Index mappings
smpl_2_openpose = np.array(
    [
        24,
        12,
        17,
        19,
        21,
        16,
        18,
        20,
        0,
        2,
        5,
        8,
        1,
        4,
        7,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
    ],
    dtype=np.int32,
)
openpose_2_coco17 = joint_mapping(OpenPose_25, COCO_17)

# Predefined colours for body25 edges (24 colours) and body17 edges (19 colours)
body25_colors = np.array(
    [
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
        [128, 0, 0],
        [0, 128, 0],
        [0, 0, 128],
        [128, 128, 0],
        [128, 0, 128],
        [0, 128, 128],
        [64, 0, 0],
        [0, 64, 0],
        [0, 0, 64],
        [64, 64, 0],
        [64, 0, 64],
        [0, 64, 64],
        [192, 0, 0],
        [0, 192, 0],
        [0, 0, 192],
        [192, 192, 0],
        [192, 0, 192],
        [0, 192, 192],
    ]
)

body17_colors = np.array(
    [
        [255, 128, 0],
        [128, 255, 0],
        [0, 255, 128],
        [0, 128, 255],
        [128, 0, 255],
        [255, 0, 128],
        [64, 128, 192],
        [192, 64, 128],
        [128, 192, 64],
        [64, 192, 128],
        [192, 128, 64],
        [128, 64, 192],
        [255, 64, 0],
        [0, 64, 255],
        [64, 0, 255],
        [255, 0, 64],
        [0, 255, 64],
        [64, 255, 0],
        [192, 192, 192],
    ]
)
