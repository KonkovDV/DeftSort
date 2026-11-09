"""OFFLINE (never at run time): fit conservative collision spheres to robot link meshes.

Input : URDF (data/tiangong_pro_with_hands.urdf) + a directory with the link STL meshes it references
        (package://tiangong2pro_urdf/meshes/*.STL; public source: github.com/UBTECH-Robot/tiangong2pro_tf, Apache-2.0).
Output: JSON {"validated": false, "source": ..., "spheres": [{link, center, radius}]} in LINK frames.
        validated stays false until the FK of our URDF is checked against the simulator /tf for every arm link
        (see arm.USE_MESH_SPHERES). Robot geometry only, no scene/objects.
Method: per link, vertices -> visual-origin transform -> greedy covering: k-means++-style centres, radius = max
        distance of assigned vertices + pad, split the worst sphere until radius <= r_max (every vertex covered).
"""
import argparse, json, os, struct, sys
import xml.etree.ElementTree as ET
import numpy as np


def read_stl(path):
    data = open(path, 'rb').read()
    if len(data) >= 84:
        n = struct.unpack('<I', data[80:84])[0]
        if 84+50*n == len(data):
            rec = np.frombuffer(data, dtype=np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]), count=n, offset=84)
            return rec['v'].reshape(-1, 3).astype(float)
    pts = [list(map(float, l.split()[1:4])) for l in data.decode('ascii', 'ignore').splitlines() if l.strip().startswith('vertex')]
    if not pts:
        raise ValueError('not an STL: '+path)
    return np.array(pts)


def rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr], [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]])


def cover(points, r_max=.05, pad=.005, max_spheres=24):
    """Spheres (centre, radius) covering ALL points, each radius <= r_max+pad when achievable within max_spheres."""
    P = np.unique(np.round(np.asarray(points, float), 5), axis=0)
    centres = [P.mean(axis=0)]
    while True:
        C = np.array(centres)
        d = np.linalg.norm(P[:, None, :]-C[None], axis=2); a = d.argmin(axis=1)
        for _ in range(10):   # Lloyd refinement of the centres on their assigned points
            C = np.array([P[a == i].mean(axis=0) if np.any(a == i) else C[i] for i in range(len(C))])
            d = np.linalg.norm(P[:, None, :]-C[None], axis=2); a = d.argmin(axis=1)
        radii = np.array([d[a == i, i].max() if np.any(a == i) else 0. for i in range(len(C))])
        worst = int(radii.argmax())
        if radii[worst] <= r_max or len(C) >= max_spheres:
            return [(C[i], float(radii[i]+pad)) for i in range(len(C)) if radii[i] > 0 or np.any(a == i)]
        far = P[a == worst][d[a == worst, worst].argmax()]
        centres = list(C)+[far]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--urdf', required=True); ap.add_argument('--meshes', required=True)
    ap.add_argument('--links', nargs='+', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--r-max', type=float, default=.05); ap.add_argument('--pad', type=float, default=.005)
    ap.add_argument("--max-spheres", type=int, default=24)
    a = ap.parse_args(argv)
    root = ET.parse(a.urdf).getroot(); out = []
    for link in a.links:
        L = root.find(f"link[@name='{link}']")
        vis = None if L is None else L.find('visual')
        if vis is None:
            raise SystemExit(f'no visual for {link}')
        m = vis.find('geometry/mesh'); fn = os.path.basename(m.get('filename'))
        o = vis.find('origin'); xyz = [0, 0, 0]; rpy = [0, 0, 0]
        if o is not None:
            xyz = [float(x) for x in (o.get('xyz') or '0 0 0').split()]; rpy = [float(x) for x in (o.get('rpy') or '0 0 0').split()]
        scale = [float(x) for x in (m.get('scale') or '1 1 1').split()]
        V = read_stl(os.path.join(a.meshes, fn))*np.array(scale)
        V = V@rpy_matrix(*rpy).T+np.array(xyz)
        for c, r in cover(V, a.r_max, a.pad, a.max_spheres):
            out.append(dict(link=link, center=[round(float(x), 5) for x in c], radius=round(r, 5)))
    json.dump(dict(validated=False, source='tools/fit_spheres.py on tiangong2pro_urdf STL meshes (Apache-2.0, UBTECH)',
                   r_max=a.r_max, pad=a.pad, spheres=out), open(a.out, 'w'), indent=1)
    print(len(out), 'spheres ->', a.out)


if __name__ == '__main__':
    sys.exit(main())
