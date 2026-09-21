import numpy as np
import math
import torch

# ── OBB helpers ──────────────────────────────────────────────────────────────

def _obb_corners_xz(cx, cz, l, w, angle_deg):
    """4 corners of an OBB footprint in the xz plane."""
    theta = np.radians(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    ax = np.array([ cos_t, sin_t])   # local x-axis → world xz
    az = np.array([-sin_t, cos_t])   # local z-axis → world xz
    c  = np.array([cx, cz])
    hx, hz = l / 2.0, w / 2.0
    return np.stack([
        c + hx*ax + hz*az,
        c + hx*ax - hz*az,
        c - hx*ax + hz*az,
        c - hx*ax - hz*az,
    ])                                # (4, 2)  columns = [x, z]

def _sat_overlap(corners_a, corners_b, axis):
    """Signed overlap of two corner sets projected onto axis (positive = overlapping)."""
    pa = corners_a @ axis
    pb = corners_b @ axis
    return min(pa.max(), pb.max()) - max(pa.min(), pb.min())

def _obb_sat_xz(box_i, ang_i, box_j, ang_j, room_bounds=None, alignment_bias=2.0):
    """
    Separating Axis Theorem for two OBBs in the xz floor plane.
    Returns (colliding, translation_depth, push_axis_xz).
    push_axis_xz points FROM i TOWARD j (unit vector in xz).

    Instead of always choosing the axis with minimum overlap (MTV), we
    prefer the axis that is best aligned with the center-to-center
    direction so that objects are pushed *apart* rather than sideways
    or behind each other.

    Each candidate axis is scored as:
        score = depth / (alignment ** alignment_bias + eps)
    where alignment = |dot(signed_axis, c2c_unit)|.  Lower score wins.
    This favours axes that are both shallow *and* point away from each
    other.  ``alignment_bias`` controls how strongly alignment is
    weighted (higher = stronger preference for center-aligned axes).
    When no SAT axis is reasonably aligned (all < 0.15), we fall back
    to pushing along the raw center-to-center direction using the
    minimum overlap depth.
    """
    li, wi = float(box_i[0]), float(box_i[2])
    cxi, czi = float(box_i[3]), float(box_i[5])

    lj, wj = float(box_j[0]), float(box_j[2])
    cxj, czj = float(box_j[3]), float(box_j[5])

    corners_i = _obb_corners_xz(cxi, czi, li, wi, ang_i)
    corners_j = _obb_corners_xz(cxj, czj, lj, wj, ang_j)

    theta_i = np.radians(ang_i)
    theta_j = np.radians(ang_j)

    # 4 candidate separating axes (2 per OBB)
    axes = [
        np.array([ np.cos(theta_i),  np.sin(theta_i)]),  # i local-x
        np.array([-np.sin(theta_i),  np.cos(theta_i)]),  # i local-z
        np.array([ np.cos(theta_j),  np.sin(theta_j)]),  # j local-x
        np.array([-np.sin(theta_j),  np.cos(theta_j)]),  # j local-z
    ]

    # Center-to-center direction (from i toward j)
    c2c = np.array([cxj - cxi, czj - czi])
    c2c_len = np.linalg.norm(c2c)
    if c2c_len > 1e-8:
        c2c_unit = c2c / c2c_len
    else:
        c2c_unit = None                      # nearly coincident → fallback later

    # First pass: check for separation & collect per-axis info
    axis_info = []                           # (overlap, signed_axis)
    min_depth = float('inf')

    for axis in axes:
        n = np.linalg.norm(axis)
        if n < 1e-8:
            continue
        axis = axis / n
        overlap = _sat_overlap(corners_i, corners_j, axis)
        if overlap <= 0:
            return False, 0.0, None          # separating axis found
        # orient axis so it points from i toward j
        if c2c_unit is not None:
            sign = 1.0 if np.dot(c2c, axis) >= 0 else -1.0
        else:
            sign = 1.0
        signed_axis = axis * sign
        
        penalty = 1.0
        if room_bounds is not None:
            cxL, czL, hxL, hzL, axL, azL = room_bounds
            shift = overlap / 2.0
            
            def get_out(corners):
                c_rel = corners - np.array([cxL, czL])
                proj_x = c_rel @ axL
                proj_z = c_rel @ azL
                return max(0, proj_x.max() - hxL) + max(0, -hxL - proj_x.min()) + max(0, proj_z.max() - hzL) + max(0, -hzL - proj_z.min())
            
            out_i_before = get_out(corners_i)
            out_j_before = get_out(corners_j)
            
            ci_new = corners_i - shift * signed_axis
            cj_new = corners_j + shift * signed_axis
            
            out_i_after = get_out(ci_new)
            out_j_after = get_out(cj_new)
            
            diff_i = out_i_after - out_i_before
            diff_j = out_j_after - out_j_before
            
            if diff_i > 1e-3 or diff_j > 1e-3:
                penalty = 1000.0
                
        axis_info.append((overlap * penalty, signed_axis, overlap))
        if overlap < min_depth:
            min_depth = overlap

    if c2c_unit is None:
        best_axis = min(axis_info, key=lambda x: x[0])
        return True, best_axis[2], best_axis[1]

    ALIGN_THRESHOLD = 0.15
    best_score = float('inf')
    best_axis = None
    best_true_overlap = 0.0

    for penalized_overlap, signed_axis, true_overlap in axis_info:
        alignment = abs(np.dot(signed_axis, c2c_unit))
        if alignment < ALIGN_THRESHOLD:
            continue
        score = penalized_overlap / (alignment ** alignment_bias + 1e-12)
        if score < best_score:
            best_score = score
            best_axis = signed_axis
            best_true_overlap = true_overlap

    if best_axis is not None:
        return True, best_true_overlap, best_axis

    return True, min_depth, c2c_unit

# ── public API ────────────────────────────────────────────────────────────────

def resolve_bbox_collisions_obb(
    boxes,                    # (N, 6) tensor  [l, h, w, x, y, z]
    angles_pred,              # (N,) or (N,1) tensor — yaw in DEGREES
    objectness_mask=None,     # (N,) bool tensor — False → skip (floor, _scene_)
    class_labels=None,        # (N, C) class probabilities or None
    max_iter=500,             # Increased iterations for better convergence
    push_eps=0.02,            # extra clearance added after resolving each overlap (m)
    verbose=True,
):
    boxes   = boxes.clone()
    N       = boxes.shape[0]
    device  = boxes.device

    if isinstance(angles_pred, torch.Tensor):
        angles_np = angles_pred.detach().cpu().numpy().flatten().astype(float)
    else:
        angles_np = np.array(angles_pred, dtype=float).flatten()

    boxes_np = boxes.detach().cpu().numpy().astype(float)
    
    labels_idx = None
    if class_labels is not None:
        labels_idx = np.argmax(class_labels, axis=-1)

    layout_indices = []
    if objectness_mask is not None:
        for i in range(N):
            # Identify layout or floor instances
            if not bool(objectness_mask[i]) or (labels_idx is not None and labels_idx[i] in [0]):
                layout_indices.append(i)
                
    main_layout_idx = -1
    if layout_indices:
        areas = [float(boxes_np[i, 0]) * float(boxes_np[i, 2]) for i in layout_indices]
        main_layout_idx = layout_indices[np.argmax(areas)]

    for iteration in range(max_iter):
        moved = False
        
        room_bounds = None
        if main_layout_idx != -1:
            L_i = main_layout_idx
            l_L, w_L = float(boxes_np[L_i, 0]), float(boxes_np[L_i, 2])
            cxL, czL = float(boxes_np[L_i, 3]), float(boxes_np[L_i, 5])
            ang_L = angles_np[L_i]
            theta_L = np.radians(ang_L)
            axL = np.array([ np.cos(theta_L),  np.sin(theta_L)])
            azL = np.array([-np.sin(theta_L),  np.cos(theta_L)])
            hxL, hzL = l_L / 2.0, w_L / 2.0
            room_bounds = (cxL, czL, hxL, hzL, axL, azL)

        # 1. Enforce layout containment for all objects (including lamps)
        if room_bounds is not None:
            cxL, czL, hxL, hzL, axL, azL = room_bounds
            for i in range(N):
                if i in layout_indices: continue
                
                # Check corners of i
                corners_i = _obb_corners_xz(float(boxes_np[i, 3]), float(boxes_np[i, 5]), 
                                           float(boxes_np[i, 0]), float(boxes_np[i, 2]), angles_np[i])
                
                # Shift relative to layout center
                c_rel = corners_i - np.array([cxL, czL])
                
                # Project onto layout axes
                proj_x = c_rel @ axL
                proj_z = c_rel @ azL
                
                min_x, max_x = proj_x.min(), proj_x.max()
                min_z, max_z = proj_z.min(), proj_z.max()
                
                shift_x = 0.0
                if (max_x - min_x) > 2 * hxL:
                    shift_x = - (max_x + min_x) / 2.0
                else:
                    if max_x > hxL: shift_x = hxL - max_x
                    elif min_x < -hxL: shift_x = -hxL - min_x
                
                shift_z = 0.0
                if (max_z - min_z) > 2 * hzL:
                    shift_z = - (max_z + min_z) / 2.0
                else:
                    if max_z > hzL: shift_z = hzL - max_z
                    elif min_z < -hzL: shift_z = -hzL - min_z
                
                if abs(shift_x) > 1e-4 or abs(shift_z) > 1e-4:
                    boxes_np[i, 3] += shift_x * axL[0] + shift_z * azL[0]
                    boxes_np[i, 5] += shift_x * axL[1] + shift_z * azL[1]
                    moved = True

        # 2. Resolve object-object collisions
        for i in range(N):
            if i in layout_indices:
                continue
            if labels_idx is not None and labels_idx[i] in []:
                continue # Ignore lamps for collisions

            for j in range(i + 1, N):
                if j in layout_indices:
                    continue
                if labels_idx is not None and labels_idx[j] in []:
                    continue # Ignore lamps for collisions

                # Vertical check
                hy_i = float(boxes_np[i, 1]) / 2.0
                cy_i = float(boxes_np[i, 4])
                hy_j = float(boxes_np[j, 1]) / 2.0
                cy_j = float(boxes_np[j, 4])
                
                if (cy_i + hy_i) <= (cy_j - hy_j) or (cy_j + hy_j) <= (cy_i - hy_i):
                    continue # No vertical intersection

                colliding, depth, push_axis = _obb_sat_xz(
                    boxes_np[i], angles_np[i],
                    boxes_np[j], angles_np[j],
                    room_bounds=room_bounds
                )

                if colliding and push_axis is not None:
                    moved = True
                    shift = (depth + push_eps) / 2.0
                    boxes_np[i, 3] -= shift * push_axis[0]  # x
                    boxes_np[i, 5] -= shift * push_axis[1]  # z
                    boxes_np[j, 3] += shift * push_axis[0]  # x
                    boxes_np[j, 5] += shift * push_axis[1]  # z

        if not moved:
            if verbose:
                print(f"  [OBB resolve] converged at iteration {iteration+1}")
            break
    else:
        if verbose:
            print(f"  [OBB resolve] hit max_iter={max_iter}, residual overlaps may remain")

    boxes[:, 3] = torch.tensor(boxes_np[:, 3], dtype=boxes.dtype, device=device)
    boxes[:, 5] = torch.tensor(boxes_np[:, 5], dtype=boxes.dtype, device=device)
    return boxes
