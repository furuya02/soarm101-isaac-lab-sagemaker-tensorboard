"""Replace the Reach goal marker with a magenta sphere."""

from pathlib import Path

CFG_PY = Path("/opt/isaac_so_arm101/src/isaac_so_arm101/tasks/reach/reach_env_cfg.py")

INSERT = """import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg

GOAL_SPHERE_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/Command/goal_sphere",
    markers={
        "goal": sim_utils.SphereCfg(
            radius=0.025,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 1.0)),
        ),
    },
)


"""

src = CFG_PY.read_text()
if "GOAL_SPHERE_MARKER_CFG" not in src:
    src = src.replace("@configclass\nclass CommandsCfg:", INSERT + "@configclass\nclass CommandsCfg:", 1)
    src = src.replace(
        "        debug_vis=True,\n",
        "        debug_vis=True,\n        goal_pose_visualizer_cfg=GOAL_SPHERE_MARKER_CFG,\n",
        1,
    )
    assert "GOAL_SPHERE_MARKER_CFG" in src, "patch failed"
    CFG_PY.write_text(src)
