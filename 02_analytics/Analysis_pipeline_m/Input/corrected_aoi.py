"""Corrected AOI geometry, ground-truthed by pixel-measuring real stimulus images."""
import numpy as np
from matplotlib.path import Path

SCREEN_W, SCREEN_H = 2560, 1440

CENTERS_PX = {
    "pos1": (2112, 650),
    "pos2": (1628, 1140),
    "pos3": (916, 1140),
    "pos4": (441, 645),
}
OBJ_BOX = (420, 420)
SUBJECT_BOX_PX = (1277-235, 451-340, 1277+235, 451+340)

AOI_PX = {}
for name, (cx, cy) in CENTERS_PX.items():
    w, h = OBJ_BOX
    AOI_PX[name] = (cx-w/2, cy-h/2, cx+w/2, cy+h/2)
AOI_PX["subject"] = SUBJECT_BOX_PX

AOI_NORM = {name: (x1/SCREEN_W, y1/SCREEN_H, x2/SCREEN_W, y2/SCREEN_H) for name,(x1,y1,x2,y2) in AOI_PX.items()}

def classify_aoi_tight(x, y):
    for name, (x1,y1,x2,y2) in AOI_NORM.items():
        if x1<=x<=x2 and y1<=y<=y2:
            return name
    return "elsewhere"

def classify_aoi_tight_vec(xs, ys):
    pts = np.column_stack([xs,ys])
    labels = np.full(len(pts), "elsewhere", dtype=object)
    for name, (x1,y1,x2,y2) in AOI_NORM.items():
        mask = (labels=="elsewhere") & (pts[:,0]>=x1)&(pts[:,0]<=x2)&(pts[:,1]>=y1)&(pts[:,1]<=y2)
        labels[mask] = name
    return labels

# ---- pie-slice equal-area version, built on the CORRECTED centers ----
subject_center = ((SUBJECT_BOX_PX[0]+SUBJECT_BOX_PX[2])/2, (SUBJECT_BOX_PX[1]+SUBJECT_BOX_PX[3])/2)
POS_NAMES = ["pos1","pos2","pos3","pos4"]
obj_angle, obj_radius = {}, {}
for name in POS_NAMES:
    cx, cy = CENTERS_PX[name]
    dx, dy = cx-subject_center[0], cy-subject_center[1]
    obj_angle[name] = np.degrees(np.arctan2(dy,dx))
    obj_radius[name] = np.hypot(dx,dy)

order = sorted(POS_NAMES, key=lambda n: obj_angle[n])
angles_sorted = [obj_angle[n] for n in order]
bisectors = [(angles_sorted[i]+angles_sorted[i+1])/2 for i in range(3)]
lower = angles_sorted[0] - (bisectors[0]-angles_sorted[0])
upper = angles_sorted[3] + (angles_sorted[3]-bisectors[2])
bounds_by_order = [lower]+bisectors+[upper]
bounds_deg = {order[i]: (bounds_by_order[i], bounds_by_order[i+1]) for i in range(4)}

def ray_box_distance(cx,cy,x1,y1,x2,y2,angle_deg):
    t = np.radians(angle_deg)
    dx,dy = np.cos(t), np.sin(t)
    cands=[]
    if dx>0: cands.append((x2-cx)/dx)
    elif dx<0: cands.append((x1-cx)/dx)
    if dy>0: cands.append((y2-cy)/dy)
    elif dy<0: cands.append((y1-cy)/dy)
    return min(c for c in cands if c>0)

def sector_area(name, t1, t2, r_out, n=200):
    ts = np.linspace(t1,t2,n)
    r_in = np.array([ray_box_distance(*subject_center,*SUBJECT_BOX_PX,t) for t in ts])
    return np.trapezoid(0.5*(r_out**2-r_in**2), np.radians(ts))

from scipy.optimize import brentq
min_areas = {name: sector_area(name,*bounds_deg[name], obj_radius[name]) for name in POS_NAMES}
target_area = max(min_areas.values())*1.3
r_out_sol = {}
for name in POS_NAMES:
    t1,t2 = bounds_deg[name]
    f = lambda r: sector_area(name,t1,t2,r)-target_area
    r_out_sol[name] = brentq(f, obj_radius[name], obj_radius[name]*6)

def sector_polygon_norm(name, n=60):
    t1,t2 = bounds_deg[name]
    r_out = r_out_sol[name]
    pts=[]
    for t in np.linspace(t1,t2,n):
        pts.append((subject_center[0]+r_out*np.cos(np.radians(t)), subject_center[1]+r_out*np.sin(np.radians(t))))
    for t in np.linspace(t2,t1,n):
        r_in = ray_box_distance(*subject_center,*SUBJECT_BOX_PX,t)
        pts.append((subject_center[0]+r_in*np.cos(np.radians(t)), subject_center[1]+r_in*np.sin(np.radians(t))))
    pts = np.array(pts)
    return pts/np.array([SCREEN_W,SCREEN_H])

SECTOR_PATHS = {name: Path(sector_polygon_norm(name)) for name in POS_NAMES}

def classify_aoi_pieslice(x,y):
    x1,y1,x2,y2 = AOI_NORM["subject"]
    if x1<=x<=x2 and y1<=y<=y2: return "subject"
    for name, path in SECTOR_PATHS.items():
        if path.contains_point((x,y)): return name
    return "elsewhere"

def classify_aoi_pieslice_vec(xs, ys):
    pts = np.column_stack([xs,ys])
    labels = np.full(len(pts), "elsewhere", dtype=object)
    x1,y1,x2,y2 = AOI_NORM["subject"]
    in_subj = (pts[:,0]>=x1)&(pts[:,0]<=x2)&(pts[:,1]>=y1)&(pts[:,1]<=y2)
    labels[in_subj] = "subject"
    for name, path in SECTOR_PATHS.items():
        mask = labels=="elsewhere"
        if mask.sum()==0: continue
        inside = path.contains_points(pts[mask])
        idx = np.where(mask)[0]
        labels[idx[inside]] = name
    return labels

if __name__ == "__main__":
    print("bounds_deg:", bounds_deg)
    print("r_out_sol:", r_out_sol)
    print("target_area:", target_area)
    for name in POS_NAMES:
        cx,cy = CENTERS_PX[name]
        print(name, "tight->", classify_aoi_tight(cx/SCREEN_W, cy/SCREEN_H), " pieslice->", classify_aoi_pieslice(cx/SCREEN_W, cy/SCREEN_H))
