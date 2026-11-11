"""SI units; explicit optical frames. No nominal camera intrinsics."""
import numpy as np


def finite(a, shape=None):
    """float array of the given shape with finite values, else ValueError."""
    a = np.asarray(a, dtype=float)
    if (shape is not None and a.shape != shape) or not np.isfinite(a).all():
        raise ValueError('shape or nonfinite value')
    return a


def rotation(axis, angle):
    """3x3 rotation about `axis` by `angle` (rad), Rodrigues formula."""
    a = finite(axis, (3,)); norm = np.linalg.norm(a)
    if norm < 1e-12 or not np.isfinite(angle):
        raise ValueError('invalid rotation')
    x, y, z = a / norm
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K@K)


def pose(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    """4x4 transform from translation xyz (m) and URDF roll-pitch-yaw (rad), R = Rz(y) Ry(p) Rx(r)."""
    T = np.eye(4); T[:3, 3] = finite(xyz, (3,))
    r, p, y = finite(rpy, (3,))
    T[:3, :3] = rotation([0, 0, 1], y) @ rotation([0, 1, 0], p) @ rotation([1, 0, 0], r)
    return T


_E4 = np.array([0., 0., 0., 1.]); _I3 = np.eye(3)


def _close(a, b, atol=1e-8, rtol=1e-5):
    # == np.allclose(a, b, rtol, atol) for finite inputs (np.isclose: |a-b| <= atol + rtol*|b|), without its overhead
    return bool(np.all(np.abs(a-b) <= atol+rtol*np.abs(b)))


def check_transform(T):
    """T if it is a finite rigid 4x4 transform (orthonormal, det > 0, last row 0 0 0 1), else ValueError."""
    T = finite(T, (4, 4))
    if not _close(T[3], _E4) or not _close(T[:3, :3].T@T[:3, :3], _I3, atol=1e-6) or np.linalg.det(T[:3, :3]) < 0.999:
        raise ValueError('not rigid transform')
    return T


def quaternion_transform(xyz, xyzw):
    """4x4 transform from translation xyz (m) and unit quaternion (x, y, z, w)."""
    q = finite(xyzw, (4,))
    if abs(np.linalg.norm(q)-1) > 1e-3:
        raise ValueError('quaternion not unit')
    x, y, z, w = q / np.linalg.norm(q)
    T = pose(xyz)
    T[:3, :3] = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    return check_transform(T)


def check_rotation(matrix):
    """The matrix if it is a proper 3x3 rotation, else ValueError."""
    matrix=finite(matrix,(3,3))
    if not _close(matrix.T@matrix,_I3,atol=1e-6) or abs(np.linalg.det(matrix)-1)>1e-6: raise ValueError("invalid rotation matrix")
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
