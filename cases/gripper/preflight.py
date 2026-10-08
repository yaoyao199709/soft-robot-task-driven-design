"""Dependency-free, read-only Gripper structure and syntax check."""
from pathlib import Path
import ast
import sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parent
SCRIPTS=["Sim2Real/run_grasp_simulation.py","Sim2Real/run_grasp_trials.py","RL/train_surrogate_meta_model.py","RL/train_ppo_codesign.py","RL/evaluate_ppo_policy.py","RL/gripper_rl_env.py","RL/gripper_simulation_rl.py","urdf/modify_cylinder_urdf.py","urdf/modify_egg_urdf.py"]
ASSETS=["RL/poly_surrogate_coefficient.csv","RL/poly_meta-model.h5","RL/poly_meta-model_x_scaler.joblib","RL/poly_meta-model_y_scaler.joblib","RL/models/ywk5xiqs/final_model.zip","urdf/cylinder.urdf","urdf/egg.urdf","urdf/holder.urdf","link/top.STL","link/module.STL","link/end.STL"]
def check():
    issues=[]
    for n in SCRIPTS:
        p=ROOT/n
        if not p.is_file(): issues.append("missing script: "+n);continue
        try: ast.parse(p.read_text(encoding="utf-8-sig"),filename=n)
        except SyntaxError as ex: issues.append("syntax: "+n+": "+str(ex))
    for n in ASSETS:
        if not (ROOT/n).is_file():issues.append("missing asset: "+n)
    for urdf in (ROOT/"urdf").glob("*.urdf"):
        try:
            for mesh in ET.parse(urdf).findall(".//mesh"):
                name=mesh.attrib.get("filename","")
                if name and not ((urdf.parent/name).is_file() or (ROOT/name).is_file()):issues.append("unresolved URDF mesh: "+urdf.name+": "+name)
        except ET.ParseError as ex:issues.append("invalid URDF "+str(ex))
    print("Checked",len(SCRIPTS),"scripts,",len(ASSETS),"required assets, URDF references")
    for issue in issues:print("FAIL:",issue)
    if not issues: print("PASS static checks; simulation/training NOT run")
    return int(bool(issues))
if __name__=="__main__":sys.exit(check())
