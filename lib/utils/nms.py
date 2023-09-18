import numpy as np

def oks_nms(poses, scores, thresh, sigmas=None, in_vis_thre=None):
    if len(poses) == 0: return []
    areas = (np.max(poses[:, :, 0], axis=1) - np.min(poses[:, :, 0], axis=1)) * \
            (np.max(poses[:, :, 1], axis=1) - np.min(poses[:, :, 1], axis=1))
    poses = poses.reshape(poses.shape[0], -1)

    order = scores.argsort()[::-1]

    keep = []
    keep_ind = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        oks_ovr = oks_iou(poses[i], poses[order[1:]], areas[i], areas[order[1:]], sigmas, in_vis_thre)
        inds = np.where(oks_ovr <= thresh)[0]
        nms_inds = np.where(oks_ovr > thresh)[0]
        nms_inds = order[nms_inds + 1]
        keep_ind.append(nms_inds.tolist())
        order = order[inds + 1]

    return keep, keep_ind

def oks_iou(g, d, a_g, a_d, sigmas=None, in_vis_thre=None):
    vars = (sigmas * 2) ** 2
    xg = g[0::3]
    yg = g[1::3]
    vg = g[2::3]
    ious = np.zeros((d.shape[0]))
    for n_d in range(0, d.shape[0]):
        xd = d[n_d, 0::3]
        yd = d[n_d, 1::3]
        vd = d[n_d, 2::3]
        dx = xd - xg
        dy = yd - yg
        e = (dx ** 2 + dy ** 2) / vars / ((a_g + a_d[n_d]) / 2 + np.spacing(1)) / 2
        if in_vis_thre is not None:
            ind = list(vg >= in_vis_thre) and list(vd >= in_vis_thre)
            e = e[ind]
        ious[n_d] = np.sum(np.exp(-e)) / e.shape[0] if e.shape[0] != 0 else 0.0
    return ious