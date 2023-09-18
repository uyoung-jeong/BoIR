import numpy as np
import cv2
import copy

# follow matplotlib tab color
COLORS = {
'blue': [31, 119, 180],
'orange': [255, 127, 14],
'red': [214, 39, 40],
'purple': [148, 103, 189],
'brown': [140, 86, 75],
'pink': [227, 119, 194],
'gray': [127, 127, 127],
'olive': [188, 189, 34],
'cyan': [23, 190, 207]
}

FONT_TYPE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.4

# kpts: n_instances x n_kpts x (x,y,visibility)
# text: n_instance x str
def draw_skeleton(img, kpts, format='coco', skeleton=None, text=None, colors=None):
    out_img = copy.deepcopy(img)
    h,w = out_img.shape[:2]
    c_i = 0
    colors = list(COLORS.values()) if colors is None else colors
    for k_i, kpt in enumerate(kpts):
        kpt[:,2] = (kpt[:,2]>0)*1 # set 1 if nonzero
        if sum(kpt[:,2])==0:
            continue
        kpt = kpt.astype(int)
        pts,v = kpt[:,:2], kpt[:,2]
        color = colors[c_i]
        c_i += 1
        if c_i == len(colors):
            c_i = 0

        # draw skeleton
        for sk in skeleton:
            if np.all(v[sk]>0):
                out_img = cv2.line(out_img, pts[sk[0]], pts[sk[1]], color=color, thickness=2, lineType=cv2.LINE_AA)

        # draw keypoint
        valid_mask = (v > 0)
        pts_valid = pts[valid_mask]
        for pt in pts_valid:
            out_img = cv2.circle(out_img, pt, radius=3, color=color, thickness=3, lineType=cv2.LINE_AA)

        # draw text at topleft corner
        if text is not None:
            left, right = max(np.min(kpt[:,0])-5,0), min(np.max(kpt[:,0])+5, w-1)
            top, down = max(np.min(kpt[:,1])-5,0), min(np.max(kpt[:,1])+5, h-1)
            out_img = cv2.rectangle(out_img, (left, top), (right, down), color, 2) # draw bbox

            text_size, _ = cv2.getTextSize(text[k_i], FONT_TYPE, FONT_SCALE, 1)
            text_w, text_h = text_size
            out_img = cv2.rectangle(out_img, (left, top), (left + text_w, top + text_h), color, -1)
            out_img = cv2.putText(out_img, text[k_i], (left, top+10), FONT_TYPE, FONT_SCALE, (255,255,255), 1, cv2.LINE_AA)
    return out_img

# bbox is xywh format
def draw_bbox(img, bboxs, text=None, colors=None):
    out_img = copy.deepcopy(img)
    h,w = out_img.shape[:2]
    c_i = 0
    colors = list(COLORS.values()) if colors is None else colors
    for b_i, bbox in enumerate(bboxs):
        color = colors[c_i]
        c_i += 1
        if c_i == len(colors):
            c_i = 0

        # draw bbox    
        left = int(bbox[0])
        top = int(bbox[1])
        right = int(bbox[0]+bbox[2])
        down = int(bbox[1]+bbox[3])
        out_img = cv2.rectangle(out_img, (left, top), (right, down), color, 2)

        if text is not None:
            text_size, _ = cv2.getTextSize(text[b_i], FONT_TYPE, FONT_SCALE, 1)
            text_w, text_h = text_size
            out_img = cv2.rectangle(out_img, (left, top), (left + text_w, top + text_h), color, -1)
            out_img = cv2.putText(out_img, text[b_i], (left, top+10), FONT_TYPE, FONT_SCALE, (255,255,255), 1, cv2.LINE_AA)
    return out_img

def tabular_str(lst, space=4, sep=' | '):
    str = ''
    #print(*lst, sep=sep)
    for i, e in enumerate(lst):
        str += f"{e:<{space}}"
        if i+1 < len(lst):
            str += sep
    return str
