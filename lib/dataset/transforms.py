import random
import cv2
import numpy as np

from torchvision.transforms import functional as F

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask, joints, area, bbox):
        for t in self.transforms:
            image, mask, joints, area, bbox = t(image, mask, joints, area, bbox)
        return image, mask, joints, area, bbox

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

class ToTensor(object):
    def __call__(self, image, mask, joints, area, bbox):
        return F.to_tensor(image), mask, joints, area, bbox

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask, joints, area, bbox):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, mask, joints, area, bbox

class RandomHorizontalFlip(object):
    def __init__(self, flip_index, output_size, prob=0.5):
        self.flip_index = flip_index
        self.prob = prob
        self.output_size = output_size
        self.bbox_flip_index = [1, 0, 3, 2]

    def __call__(self, image, mask, joints, area, bbox):
        if random.random() < self.prob:
            image = image[:, ::-1] - np.zeros_like(image)
            mask = mask[:, ::-1] - np.zeros_like(mask)
            joints = joints[:, self.flip_index]
            joints[:, :, 0] = self.output_size - joints[:, :, 0] - 1
            bbox = bbox[:, self.bbox_flip_index]
            bbox[:, :, 0] = self.output_size - bbox[:, :, 0] - 1
        return image, mask, joints, area, bbox

class RandomAffineTransform(object):
    def __init__(self,input_size, output_size, max_rotation,
                 min_scale, max_scale, scale_type, max_translate):
        self.input_size = input_size
        self.output_size = output_size
        self.max_rotation = max_rotation
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scale_type = scale_type
        self.max_translate = max_translate

    def _get_affine_matrix(self, center, scale, res, rot=0):
        # Generate transformation matrix
        h = 200 * scale
        t = np.zeros((3, 3))
        t[0, 0] = float(res[1]) / h
        t[1, 1] = float(res[0]) / h
        t[0, 2] = res[1] * (-float(center[0]) / h + .5)
        t[1, 2] = res[0] * (-float(center[1]) / h + .5)
        t[2, 2] = 1
        scale = t[0,0]*t[1,1]
        if not rot == 0:
            rot = -rot  # To match direction of rotation from cropping
            rot_mat = np.zeros((3, 3))
            rot_rad = rot * np.pi / 180
            sn, cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0, :2] = [cs, -sn]
            rot_mat[1, :2] = [sn, cs]
            rot_mat[2, 2] = 1
            # Need to rotate around center
            t_mat = np.eye(3)
            t_mat[0, 2] = -res[1]/2
            t_mat[1, 2] = -res[0]/2
            t_inv = t_mat.copy()
            t_inv[:2, 2] *= -1
            t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
        return t, scale

    def _affine_joints(self, joints, mat):
        joints = np.array(joints)
        shape = joints.shape
        joints = joints.reshape(-1, 2)
        return np.dot(np.concatenate(
            (joints, joints[:, 0:1]*0+1), axis=1), mat.T).reshape(shape)

    def __call__(self, image, mask, joints, area, bbox):
        height, width = image.shape[:2]

        center = np.array((width/2, height/2))
        if self.scale_type == 'long':
            scale = max(height, width)/200
            print("###################please modify range")
        elif self.scale_type == 'short':
            scale = min(height, width)/200
        else:
            raise ValueError('Unkonw scale type: {}'.format(self.scale_type))
        aug_scale = np.random.random() * (self.max_scale - self.min_scale) + self.min_scale
        scale *= aug_scale
        aug_rot = (np.random.random() * 2 - 1) * self.max_rotation

        if self.max_translate > 0:
            dx = np.random.randint(
                -self.max_translate*scale, self.max_translate*scale)
            dy = np.random.randint(
                -self.max_translate*scale, self.max_translate*scale)
            center[0] += dx
            center[1] += dy

        mat_output, _ = self._get_affine_matrix(
            center, scale, (self.output_size, self.output_size), aug_rot
        )
        mat_output = mat_output[:2]
        mask = cv2.warpAffine((mask*255).astype(np.uint8), mat_output, (self.output_size, self.output_size)) / 255
        mask = (mask > 0.5).astype(np.float32)

        joints[:, :, 0:2] = self._affine_joints(
            joints[:, :, 0:2], mat_output
        )
        bbox = self._affine_joints(bbox, mat_output)

        mat_input, final_scale = self._get_affine_matrix(
            center, scale, (self.input_size, self.input_size), aug_rot
        )
        mat_input = mat_input[:2]
        area = area*final_scale
        image = cv2.warpAffine(image, mat_input, (self.input_size, self.input_size))

        return image, mask, joints, area, bbox