"""SYNTHETIC TEST SCENE ONLY -- never imported by the solution package.
The organiser's default simulator layout (hackathon/config/scene.yaml) is used here to check reachability and planning
offline. The solution itself measures the table and the trays with its cameras (#645, see COMPLIANCE.md)."""
TABLE_TOP = .70-.88                       # scene.yaml table.height - standing_pelvis_height (pelvis frame)
TRAYS = {'tray_left': (.34, .69), 'tray_right': (.34, -.69), 'tray_center': (.53, 0.)}
PLANE = None
def plane():
    import numpy as np
    return (np.array([0, 0, 1.]), -TABLE_TOP)
