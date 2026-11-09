"""SI units; explicit optical frames. No nominal camera intrinsics."""
import numpy as np


def finite(a, shape=None):
    a = np.asarray(a, dtype=float)
    if (shape is not None and a.shape != shape) or not np.isfinite(a).all():
        raise ValueError('shape or nonfinite value')
    return a


def rotation(axis, angle):
    a = finite(axis, (3,)); norm = np.linalg.norm(a)
    if norm < 1e-12 or not np.isfinite(angle):
        raise ValueError('invalid rotation')
    x, y, z = a / norm
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K@K)


def pose(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    T = np.eye(4); T[:3, 3] = finite(xyz, (3,))
    r, p, y = finite(rpy, (3,))
    T[:3, :3] = rotation([0, 0, 1], y) @ rotation([0, 1, 0], p) @ rotation([1, 0, 0], r)
    return T


def check_transform(T):
    T = finite(T, (4, 4))
    if not np.allclose(T[3], [0, 0, 0, 1]) or not np.allclose(T[:3, :3].T@T[:3, :3], np.eye(3), atol=1e-6) or np.linalg.det(T[:3, :3]) < 0.999:
        raise ValueError('not rigid transform')
    return T


def quaternion_transform(xyz, xyzw):
    q = finite(xyzw, (4,))
    if abs(np.linalg.norm(q)-1) > 1e-3:
        raise ValueError('quaternion not unit')
    x, y, z, w = q / np.linalg.norm(q)
    T = pose(xyz)
    T[:3, :3] = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    return check_transform(T)


def fresh(now, stamp, age=0.3):
    if not np.isfinite([now, stamp, age]).all() or age <= 0 or not 0 <= now-stamp <= age:
        raise ValueError('stale/future/unclocked data')


def project(depth_mm, K, T, now, stamp, max_age=0.3):
    fresh(now, stamp, max_age)
    depth = np.asarray(depth_mm)
    if depth.ndim != 2 or depth.dtype.kind != 'u' or depth.dtype.itemsize != 2:
        raise ValueError('expected 16UC1 aligned depth in mm')
    K = finite(K, (3, 3)); T = check_transform(T)
    if K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1]) or abs(K[0, 1]) > 1e-8 or abs(K[1, 0]) > 1e-8:
        raise ValueError('unsupported/invalid K')
    v, u = np.indices(depth.shape); z = depth.astype(float)*0.001
    valid = (z >= .15) & (z <= 6) & (depth != 0)
    p = np.stack(((u-K[0, 2])*z/K[0, 0], (v-K[1, 2])*z/K[1, 1], z), axis=-1)
    p = p @ T[:3, :3].T + T[:3, 3]
    p[~valid] = np.nan
    return p, valid


def check_rotation(matrix):
    matrix=finite(matrix,(3,3))
    if not np.allclose(matrix.T@matrix,np.eye(3),atol=1e-6) or abs(np.linalg.det(matrix)-1)>1e-6: raise ValueError("invalid rotation matrix")
    return matrix


def rotation_error(target, current):
    """SO(3) logarithm, including pi rotations (not a sine-only residual)."""
    target=check_rotation(target); current=check_rotation(current)
    R = target @ current.T
    theta = np.arccos(np.clip((np.trace(R)-1)/2, -1, 1))
    v = np.array([R[2, 1]-R[1, 2], R[0, 2]-R[2, 0], R[1, 0]-R[0, 1]])
    if theta < 1e-7:
        return v*.5
    if np.pi-theta < 1e-5:
        vals, vecs = np.linalg.eigh((R+R.T)/2)
        axis = vecs[:, np.argmax(vals)]
        if np.dot(axis,v)<0: axis=-axis
        return axis*theta
    return v*theta/(2*np.sin(theta))
