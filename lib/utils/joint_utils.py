COCO_JOINT_NAMES = {
'nose': 0,
'left_eye': 1,
'right_eye': 2,
'left_ear': 3,
'right_ear': 4,
'left_shoulder': 5,
'right_shoulder': 6,
'left_elbow': 7,
'right_elbow': 8,
'left_wrist': 9,
'right_wrist': 10,
'left_hip': 11,
'right_hip': 12,
'left_knee': 13,
'right_knee': 14,
'left_ankle': 15,
'right_ankle': 16}

COCO_SKELETON = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11],
                [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10],
                [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]

CROWDPOSE_JOINT_NAMES = {
'left_shoulder': 0,
'right_shoulder': 1,
'left_elbow': 2,
'right_elbow': 3,
'left_wrist': 4,
'right_wrist': 5,
'left_hip': 6,
'right_hip': 7,
'left_knee': 8,
'right_knee': 9,
'left_ankle': 10,
'right_ankle': 11,
'head': 12,
'neck': 13
}

CROWDPOSE_SKELETON = [[12, 13], [13, 0], [13, 1], [0, 2], [2, 4], [1, 3],
                        [3, 5], [13, 7], [13, 6], [7, 9], [9, 11], [6, 8], [8, 10]]

joint_names_dict = {
'coco': COCO_JOINT_NAMES,
'crowdpose': CROWDPOSE_JOINT_NAMES
}

skeleton_dict = {
'coco': COCO_SKELETON,
'crowdpose': CROWDPOSE_SKELETON
}

def get_joint_dict(dataset='coco'):
    if dataset == 'ochuman':
        dataset = 'coco'
    return joint_names_dict[dataset]

def get_skeleton(dataset='coco'):
    if dataset == 'ochuman':
        dataset = 'coco'
    return skeleton_dict[dataset]
