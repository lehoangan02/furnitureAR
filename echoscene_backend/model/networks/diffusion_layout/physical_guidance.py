import torch
import numpy as np
import cv2
import heapq
from .loss import axis_aligned_bbox_overlaps_3d
from .oriented_iou_loss import cal_iou_3d

def draw_2d_gaussian(center, size, angle, image_size = 256):
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    covariance_matrix = np.array([
        [size[0]**2, 0],
        [0, size[1]**2]
    ])
    rotation_convariance_matrix = rotation_matrix @ covariance_matrix @ rotation_matrix.T

    x = np.arange(0,image_size)
    y = np.arange(0,image_size)
    xx, yy = np.meshgrid(x, y)
    xy = np.stack([xx.ravel(), yy.ravel()]).T -center
    try:
        z = np.sum((xy @ np.linalg.inv(rotation_convariance_matrix)) * xy, axis=1)
    except:
        z = np.zeros(xx.shape[0]*xx.shape[1])
    gaussian = np.exp(-0.5 * z)
    gaussian = gaussian.reshape(xx.shape)
    return gaussian

def heuristic_distance(node1, node2):
    return np.sqrt((node1[0] - node2[0])**2 + (node1[1] - node2[1])**2)

def find_shortest_path(matrix, start, end):
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    open_set = [(0, start)]
    parent_map = {}
    g_cost = {node: float('inf') for node in np.ndindex(matrix.shape)}
    g_cost[start] = 0
    count = 0
    while open_set and count<5000:
        count+=1
        _, current = heapq.heappop(open_set)

        if current == end:
            path = []
            while current in parent_map:
                path.append(current)
                current = parent_map[current]
            path.append(start)
            return path[::-1]

        for direction in directions:
            new_node = (current[0] + direction[0], current[1] + direction[1])
            if 0 <= new_node[0] < matrix.shape[0] and 0 <= new_node[1] < matrix.shape[1]:
                tentative_g_cost = g_cost[current] + matrix[new_node]
                if tentative_g_cost < g_cost[new_node]:
                    parent_map[new_node] = current
                    g_cost[new_node] = tentative_g_cost
                    f_cost = tentative_g_cost + heuristic_distance(new_node, end) * 0.01
                    heapq.heappush(open_set, (f_cost, new_node))

    return None

def compute_room_outer_loss(bbox, room_outer_box=None, scene_ids=None, objectness=None):
    """
    Computes Room-Layout Guidance loss by penalizing overlap with infinite walls or boundaries.
    If room_outer_box is None, we attempt to use the floor bounding box (found via ~objectness)
    as the boundary. If that fails, we fallback to a hardcoded 6m x 6m bounds: [-3, 3] in X/Z.
    """
    # Note: bbox format is [N, 7] (sizes, translations, angle) because we are passing it in directly.
    # Actually _denormalize_box_params returns [sizes, translations, angle]. 
    # Let's extract the sizes and centers.
    if len(bbox.shape) == 3:
        # If batched, flatten it or just take the first batch if B=1
        bbox = bbox.squeeze(0)
        
    half_sizes_obj = bbox[:, :3].clamp(min=1e-4) * 0.5
    centers_obj = bbox[:, 3:6]
    
    max_corners = centers_obj + half_sizes_obj
    min_corners = centers_obj - half_sizes_obj
    
    total_loss = 0.0
    
    unique_scenes = torch.unique(scene_ids) if scene_ids is not None else [0]
    
    for scene_id in unique_scenes:
        if scene_ids is not None:
            scene_mask = (scene_ids == scene_id)
        else:
            scene_mask = torch.ones(bbox.shape[0], dtype=torch.bool, device=bbox.device)
            
        if objectness is not None:
            obj_mask = scene_mask & objectness.to(dtype=torch.bool, device=bbox.device)
            non_obj_mask = scene_mask & ~objectness.to(dtype=torch.bool, device=bbox.device)
        else:
            obj_mask = scene_mask
            non_obj_mask = torch.zeros_like(scene_mask)
            
        if not obj_mask.any():
            continue
            
        # Default fallback boundaries
        max_bound_x = 3.0
        min_bound_x = -3.0
        max_bound_z = 3.0
        min_bound_z = -3.0
        
        # Extract the floor boundary if available
        if non_obj_mask.any():
            # non_obj_mask contains background objects like `_scene_` and `floor`.
            # Find the largest object (by X * Z area) which is typically the floor.
            non_obj_idx = torch.where(non_obj_mask)[0]
            sizes_x = half_sizes_obj[non_obj_idx, 0]
            sizes_z = half_sizes_obj[non_obj_idx, 2]
            areas = sizes_x * sizes_z
            best_idx = non_obj_idx[torch.argmax(areas)]
            
            best_center = centers_obj[best_idx].detach()
            best_half_size = half_sizes_obj[best_idx].detach()
            
            max_bound_x = best_center[0] + best_half_size[0]
            min_bound_x = best_center[0] - best_half_size[0]
            max_bound_z = best_center[2] + best_half_size[2]
            min_bound_z = best_center[2] - best_half_size[2]
        else:
            print("Warning: No floor object found for scene. Falling back to default [-3.0, 3.0] boundaries for room outer loss.")
            
        # Compute L1 penalty for objects exceeding these boundaries
        cur_max_corners = max_corners[obj_mask]
        cur_min_corners = min_corners[obj_mask]
        
        loss_x_max = torch.relu(cur_max_corners[:, 0] - max_bound_x).sum()
        loss_x_min = torch.relu(min_bound_x - cur_min_corners[:, 0]).sum()
        loss_z_max = torch.relu(cur_max_corners[:, 2] - max_bound_z).sum()
        loss_z_min = torch.relu(min_bound_z - cur_min_corners[:, 2]).sum()
        
        total_loss = total_loss + loss_x_max + loss_x_min + loss_z_max + loss_z_min
        
    return total_loss

def calc_loss_on_path(image, shortest_path, robot_width, robot_width_real, robot_hight_real, map_to_image_coordinate, image_to_map_coordinate, scale, image_size, bbox_floor):
    loss_walkable = 0.0
    box_mask = image[:,:,1] == 255
    bbox_path = []
    path_count = 0
    for i in range(len(shortest_path)):
        if box_mask[shortest_path[i]]:
            if path_count % robot_width == 0:
                center_map = image_to_map_coordinate((shortest_path[i][1], shortest_path[i][0]))
                angle = 0.
                # cal_iou_3d expects (X, Z, Y_up, X_size, Z_size, Y_size_up, alpha)
                box = np.array([center_map[0], center_map[1], 0.0, robot_width_real, robot_width_real, robot_hight_real, angle])
                bbox_path.append(box)
            path_count += 1
            
    if not bbox_path:
        return loss_walkable
        
    for box in bbox_path:
        center = map_to_image_coordinate((box[0], box[2]))
        size = (int(box[5] / scale * image_size / 2), int(box[3] / scale * image_size / 2)) # l and w
        angle = box[-1]
        box_points = cv2.boxPoints(((center[0], center[1]), size, -angle/np.pi*180))
        box_points = np.intp(box_points)
        cv2.drawContours(image, [box_points], 0, (0, 255, 255), robot_width)
        
    bbox_path = np.expand_dims(np.stack(bbox_path, 0), 0)
    bbox_cnt_path = bbox_path.shape[1]
    bbox_floor_exp = bbox_floor.unsqueeze(0) if bbox_floor.dim() == 2 else bbox_floor
    bbox_floor_cur_cnt = bbox_floor_exp.shape[1]
    
    for bbox_cnt_idx in range(bbox_floor_cur_cnt):    
        bbox_target = bbox_floor_exp[:, bbox_cnt_idx, :]
        bbox_target = torch.tile(bbox_target.unsqueeze(1), [1, bbox_cnt_path, 1])
        loss_walkable = loss_walkable + cal_iou_3d(
            torch.tensor(bbox_path, device=bbox_target.device, dtype=bbox_target.dtype), 
            bbox_target
        ).sum() / max(1, len(bbox_floor)) / bbox_floor_cur_cnt
        
    return loss_walkable

def compute_center_penalty_loss(bbox, objectness=None, sigma=0.5):
    """
    Computes Center Penalty Walkable Loss (Radial Gaussian penalty centered at room origin).
    """
    if len(bbox.shape) == 2:
        bbox = bbox.unsqueeze(0)
    centers_obj = bbox[:, :, 3:6]
    dist_sq = centers_obj[:, :, 0]**2 + centers_obj[:, :, 2]**2
    walk_penalty = torch.exp(-dist_sq / sigma).sum()
    return walk_penalty

def compute_pathfinding_walkable_loss(bbox, floor_plan, objectness=None, robot_width_real=0.35, robot_hight_real=1.5):
    """
    Computes Reachability Guidance by verifying an agent can traverse the room (Dijkstra pathfinding).
    """
    if floor_plan is None:
        return torch.tensor(0.0, device=bbox.device, dtype=bbox.dtype)

    if len(bbox.shape) == 2:
        bbox = bbox.unsqueeze(0)
        if objectness is not None and len(objectness.shape) == 1:
            objectness = objectness.unsqueeze(0)
        
    loss_walkable = torch.tensor(0.0, device=bbox.device, dtype=bbox.dtype)
    for i in range(len(bbox)):
        bbox_cur = bbox[i:i+1, :, :]
        if objectness is not None:
            obj_mask = objectness[i]
            if obj_mask.dim() > 1:
                obj_mask = obj_mask[:, 0]
            bbox_cur = bbox_cur[:, obj_mask.bool(), :]
        
        bbox_cur_cnt = bbox_cur.shape[1]
        
        if isinstance(floor_plan, list) and len(floor_plan) > i:
            fp = floor_plan[i]
        else:
            fp = floor_plan
            
        if fp is None or len(fp) != 2:
            continue
            
        vertices, faces = fp
        
        if isinstance(vertices, torch.Tensor):
            vertices = vertices.cpu().numpy()
        if isinstance(faces, torch.Tensor):
            faces = faces.cpu().numpy()
            
        floor_centroid = np.mean(vertices, axis=0)
        vertices_centered = vertices - floor_centroid
        vertices_2d = vertices_centered[:, 0::2]
        scale = np.abs(vertices_2d).max() + 0.2
        
        bbox_floor = bbox_cur[0, bbox_cur[0, :, 4] < robot_hight_real]
        
        image_size = 256
        image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        robot_width = int(robot_width_real / scale * image_size/2)

        def map_to_image_coordinate(point):
            x, y = point
            x_image = int(x / scale * image_size/2)+image_size/2
            y_image = int(y / scale * image_size/2)+image_size/2
            return x_image, y_image
        
        def image_to_map_coordinate(point):
            x, y = point
            x_map = (x - image_size/2) * 2 / image_size *scale
            y_map = (y - image_size/2) * 2 / image_size *scale
            return x_map, y_map

        for face in faces:
            face_vertices = vertices_2d[face]
            face_vertices_image = [map_to_image_coordinate(v) for v in face_vertices]
            pts = np.array(face_vertices_image, np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(image, [pts], (255, 0, 0))

        kernel = np.ones((robot_width, robot_width))
        image[:, :, 0] = cv2.erode(image[:, :, 0], kernel, iterations=1)
        floor_plan_mask = image[:, :, 0] == 255
        box_heat_map = np.zeros((image_size, image_size), dtype=np.float32)

        for box in bbox_floor:
            box = box.cpu().detach().numpy()
            rel_x = box[3] - floor_centroid[0]
            rel_z = box[5] - floor_centroid[2]
            center = map_to_image_coordinate((rel_x, rel_z))
            size = (int(box[0] / scale * image_size / 2), int(box[2] / scale * image_size / 2))
            angle = box[-1]

            box_points = cv2.boxPoints(((center[0], center[1]), size, -angle/np.pi*180))
            box_points = np.intp(box_points)
            
            box_mask = np.zeros((image_size, image_size, 3), dtype=np.uint8)
            cv2.drawContours(image, [box_points], 0, (0, 255, 0), robot_width)
            cv2.fillPoly(image, [box_points], (0, 255, 0))
            cv2.drawContours(box_mask, [box_points], 0, (0, 255, 0), robot_width)
            cv2.fillPoly(box_mask, [box_points], (0, 255, 0))
            
            box_mask_bool = box_mask[:,:,1] == 255
            if min(size) != 0:
                gaussian = draw_2d_gaussian((int(center[0]), int(center[1])), size, -angle, image_size)
                box_heat_map = box_heat_map + gaussian * box_mask_bool
            
        box_heat_map = floor_plan_mask * box_heat_map
        box_wall_heat_map = box_heat_map + (1 - floor_plan_mask) * box_heat_map.max()
        
        walkable_map = image[:, :, 0].copy()
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(walkable_map, connectivity=8)
        
        if num_labels > 2:
            area_1 = np.zeros_like(walkable_map)
            area_2 = np.zeros_like(walkable_map)
            for label in range(1, num_labels):
                mask = np.zeros_like(walkable_map)
                mask[labels == label] = 1
                if mask.sum() > area_2.sum():
                    area_2 = mask.copy()
                if area_2.sum() > area_1.sum():
                    area_2, area_1 = area_1.copy(), area_2.copy()
                    
            if area_2.sum() > 100:
                dist_tf_1 = cv2.distanceTransform(area_1.astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=5)
                minimum_area_1 = np.argmax(dist_tf_1)
                minimum_area_1_position = np.unravel_index(minimum_area_1, area_1.shape)
                
                dist_tf_2 = cv2.distanceTransform(area_2.astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=5)
                minimum_area_2 = np.argmax(dist_tf_2)
                minimum_area_2_position = np.unravel_index(minimum_area_2, area_2.shape)
                
                shortest_path = find_shortest_path(
                    box_wall_heat_map, 
                    (minimum_area_1_position[0], minimum_area_1_position[1]),
                    (minimum_area_2_position[0], minimum_area_2_position[1])
                )
                if shortest_path is not None:
                    mapped_bbox_floor = bbox_floor[:, [3, 5, 4, 0, 2, 1, 6]].clone()
                    mapped_bbox_floor[:, 0] -= torch.tensor(floor_centroid[0], device=mapped_bbox_floor.device, dtype=mapped_bbox_floor.dtype)
                    mapped_bbox_floor[:, 1] -= torch.tensor(floor_centroid[2], device=mapped_bbox_floor.device, dtype=mapped_bbox_floor.dtype)
                    loss_walkable = loss_walkable + calc_loss_on_path(
                        image, shortest_path, robot_width, robot_width_real, robot_hight_real,
                        map_to_image_coordinate, image_to_map_coordinate,
                        scale, image_size, mapped_bbox_floor
                    )

    return loss_walkable

# Backward compatibility alias
compute_walkable_loss = compute_pathfinding_walkable_loss

def compute_edge_gaussian_walkable_loss(
    bbox, floor_plan, objectness=None, 
    robot_width_real=0.35, robot_hight_real=1.5, 
    sigma_scale=0.5, heatmap_weight=1.0, repulsion_weight=1.0, 
    return_components=False, verbose=True
):
    """
    Exact OBB Edge-Gaussian Walkability Guidance Loss (Option 3).
    
    For each ground-level furniture object with OBB parameters (l, h, w, x, y, z, angle):
      1. Transform query coordinates (x, z) into the object's local rotated frame (x', z'):
           x' =  (x - x_c) * cos(angle) + (z - z_c) * sin(angle)
           z' = -(x - x_c) * sin(angle) + (z - z_c) * cos(angle)
      2. Compute exact orthogonal distance from query point to the 4 box edges:
           d_x = max(|x'| - l/2, 0)
           d_z = max(|z'| - w/2, 0)
           d_edge = sqrt(d_x^2 + d_z^2)
      3. Compute Edge Gaussian field radiating from object boundaries:
           G(x, z) = exp(-d_edge^2 / (2 * sigma^2))
         where sigma = (l + w) / 2.0 * sigma_scale. Inside the box (d_edge = 0), G(x, z) = 1.0.
    
    Component 1 - Floor Edge Heatmap (weighted by heatmap_weight):
      Evaluates the sum of per-object edge Gaussians across a PyTorch floor grid,
      penalizing floor area covered by object edge influence zones.
      
    Component 2 - Pairwise OBB Edge Repulsion (weighted by repulsion_weight):
      Evaluates object j's center in object i's exact OBB edge-Gaussian field G_i(x_j, z_j).
      Directly penalizes furniture pairs whose OBB edge zones overlap.
    """
    if len(bbox.shape) == 2:
        bbox = bbox.unsqueeze(0)
        if objectness is not None and len(objectness.shape) == 1:
            objectness = objectness.unsqueeze(0)
            
    B, N, _ = bbox.shape
    device = bbox.device
    dtype = bbox.dtype
    
    # Filter for ground-level furniture objects (exclude lamps/ceilings & padded objects)
    if objectness is not None:
        furniture_mask = objectness.to(dtype=torch.bool, device=device)
        if furniture_mask.dim() > 2:
            furniture_mask = furniture_mask[:, :, 0]
    else:
        furniture_mask = torch.ones((B, N), dtype=torch.bool, device=device)
        
    height_mask = bbox[:, :, 4] < robot_hight_real
    furniture_mask = furniture_mask & height_mask
    
    total_repulsion = torch.tensor(0.0, device=device, dtype=dtype)
    total_heatmap = torch.tensor(0.0, device=device, dtype=dtype)
    
    for b in range(B):
        furn_idx = torch.where(furniture_mask[b])[0]
        if len(furn_idx) == 0:
            continue
            
        furn_boxes = bbox[b, furn_idx]  # [M, 7] (l, h, w, x, y, z, angle)
        M = furn_boxes.shape[0]
        
        lengths = furn_boxes[:, 0]  # [M]
        widths  = furn_boxes[:, 2]  # [M]
        cx      = furn_boxes[:, 3]  # [M]
        cz      = furn_boxes[:, 5]  # [M]
        angles  = furn_boxes[:, 6]  # [M] rad
        
        half_l = lengths / 2.0
        half_w = widths / 2.0
        sigmas = (lengths + widths) / 2.0 * sigma_scale  # [M]
        
        # -------------------------------------------------------------
        # Component 2: Pairwise OBB Edge Repulsion (differentiable)
        # -------------------------------------------------------------
        if M >= 2:
            dx = cx.unsqueeze(0) - cx.unsqueeze(1)  # [M, M] (j - i)
            dz = cz.unsqueeze(0) - cz.unsqueeze(1)  # [M, M]
            
            cos_a = torch.cos(angles).unsqueeze(1)  # [M, 1] (angle of box i)
            sin_a = torch.sin(angles).unsqueeze(1)  # [M, 1]
            
            x_local =  dx * cos_a + dz * sin_a       # [M, M]
            z_local = -dx * sin_a + dz * cos_a       # [M, M]
            
            dist_x = torch.relu(torch.abs(x_local) - half_l.unsqueeze(1))  # [M, M]
            dist_z = torch.relu(torch.abs(z_local) - half_w.unsqueeze(1))  # [M, M]
            edge_dist = torch.sqrt(dist_x**2 + dist_z**2 + 1e-8)          # [M, M]
            
            avg_size_j = (lengths + widths) / 2.0  # [M]
            edge_to_edge_dist = torch.relu(edge_dist - (avg_size_j.unsqueeze(0) / 2.0))
            
            sigma_i = sigmas.unsqueeze(1)  # [M, 1]
            pairwise_g = torch.exp(-edge_to_edge_dist**2 / (2.0 * sigma_i**2 + 1e-8))
            
            mask_diag = 1.0 - torch.eye(M, device=device, dtype=dtype)
            repulsion_loss = (pairwise_g * mask_diag).sum() / 2.0
            total_repulsion = total_repulsion + repulsion_loss
            
        # -------------------------------------------------------------
        # Component 1: Floor Grid Edge Heatmap
        # -------------------------------------------------------------
        grid_res = 64
        gx = torch.linspace(-3.0, 3.0, grid_res, device=device, dtype=dtype)
        gz = torch.linspace(-3.0, 3.0, grid_res, device=device, dtype=dtype)
        grid_x, grid_z = torch.meshgrid(gx, gz, indexing='ij')  # [GR, GR]
        
        rel_x = grid_x.unsqueeze(-1) - cx.view(1, 1, M)
        rel_z = grid_z.unsqueeze(-1) - cz.view(1, 1, M)
        
        cos_arr = torch.cos(angles).view(1, 1, M)
        sin_arr = torch.sin(angles).view(1, 1, M)
        
        local_x =  rel_x * cos_arr + rel_z * sin_arr  # [GR, GR, M]
        local_z = -rel_x * sin_arr + rel_z * cos_arr  # [GR, GR, M]
        
        d_x = torch.relu(torch.abs(local_x) - half_l.view(1, 1, M))
        d_z = torch.relu(torch.abs(local_z) - half_w.view(1, 1, M))
        d_edge_grid = torch.sqrt(d_x**2 + d_z**2 + 1e-8)
        
        sigmas_grid = sigmas.view(1, 1, M)
        grid_gaussians = torch.exp(-d_edge_grid**2 / (2.0 * sigmas_grid**2 + 1e-8))  # [GR, GR, M]
        
        heatmap_sum = grid_gaussians.sum(dim=-1)  # [GR, GR]
        heatmap_loss = heatmap_sum.mean()
        
        total_heatmap = total_heatmap + heatmap_loss

    total_loss = total_heatmap * heatmap_weight + total_repulsion * repulsion_weight

    if verbose:
        print(f"[Edge-Gaussian Walkable Loss] Component 1 (Floor Heatmap, w={heatmap_weight}): {total_heatmap.item():.4f} | Component 2 (Pairwise Repulsion, w={repulsion_weight}): {total_repulsion.item():.4f} | Total Weighted Walkable: {total_loss.item():.4f}")

    if return_components:
        return total_loss, {"c1_floor_heatmap": total_heatmap, "c2_pairwise_repulsion": total_repulsion}
    return total_loss


def compute_relational_guidance_loss(
    bbox, 
    triples, 
    predicate_names=None, 
    objectness=None, 
    margin=0.05, 
    close_threshold=0.45, 
    stand_threshold=0.04
):
    """
    Computes Differentiable Spatial, Directional, Proximity, Support, Symmetry & Relative Size Relational Guidance Loss.
    Fully vectorized parallel GPU implementation with ZERO host-device CPU synchronizations.
    """
    if triples is None or len(triples) == 0:
        return torch.tensor(0.0, device=bbox.device, dtype=bbox.dtype)

    if len(bbox.shape) == 2:
        bbox = bbox.unsqueeze(0)  # [1, N, 7]

    B, N, _ = bbox.shape
    device = bbox.device
    dtype = bbox.dtype

    if triples.dim() == 2:
        triples_batch = [triples]
    else:
        triples_batch = triples

    total_loss = torch.tensor(0.0, device=device, dtype=dtype)
    num_valid_relations = 0

    for b in range(min(B, len(triples_batch))):
        cur = triples_batch[b]
        if cur is None or len(cur) == 0:
            continue

        s = cur[:, 0].long()
        p = cur[:, 1].long()
        o = cur[:, 2].long()

        valid = (s >= 0) & (s < N) & (o >= 0) & (o < N) & (s != o)
        if not valid.any():
            continue

        s, p, o = s[valid], p[valid], o[valid]
        num_valid_relations += len(s)

        # Parallel gather of box centers and sizes
        xs, ys, zs = bbox[b, s, 3], bbox[b, s, 4], bbox[b, s, 5]
        xo, yo, zo = bbox[b, o, 3], bbox[b, o, 4], bbox[b, o, 5]
        ls, hs, ws = bbox[b, s, 0], bbox[b, s, 1], bbox[b, s, 2]
        lo, ho, wo = bbox[b, o, 0], bbox[b, o, 1], bbox[b, o, 2]

        loss_vec = torch.zeros(len(s), device=device, dtype=dtype)

        # 1. left: Subject Z must be < Object Z - margin (along Z axis)
        m = (p == 1)
        if m.any():
            loss_vec[m] = torch.relu(zs[m] - zo[m] + margin)

        # 2. right: Subject Z must be > Object Z + margin (along Z axis)
        m = (p == 2)
        if m.any():
            loss_vec[m] = torch.relu(zo[m] - zs[m] + margin)

        # 3. front: Subject X must be > Object X + margin (along X axis)
        m = (p == 3)
        if m.any():
            loss_vec[m] = torch.relu(xo[m] - xs[m] + margin)

        # 4. behind: Subject X must be < Object X - margin (along X axis)
        m = (p == 4)
        if m.any():
            loss_vec[m] = torch.relu(xs[m] - xo[m] + margin)

        # 5. close by: 2D distance between Subject and Object <= close_threshold (0.45m)
        m = (p == 5)
        if m.any():
            dist_xz = torch.sqrt((xs[m] - xo[m])**2 + (zs[m] - zo[m])**2 + 1e-8)
            loss_vec[m] = torch.relu(dist_xz - close_threshold)

        # 6. standing on: Ground-level support alignment (Center Y difference < 0.04m)
        m = (p == 7)
        if m.any():
            loss_vec[m] = torch.relu(torch.abs(ys[m] - yo[m]) - stand_threshold)

        # 7. above: Subject Y must be above Object Y (+margin)
        m = (p == 6)
        if m.any():
            loss_vec[m] = torch.relu(yo[m] - ys[m] + margin)

        # 8. symmetrical to: Subject and Object are mirrored across X, Z, or XZ plane (within 0.45m)
        m = (p == 12)
        if m.any():
            d_flip_x = torch.sqrt((-xs[m] - xo[m])**2 + (zs[m] - zo[m])**2 + 1e-8)
            d_flip_z = torch.sqrt((xs[m] - xo[m])**2 + (-zs[m] - zo[m])**2 + 1e-8)
            d_flip_xz = torch.sqrt((-xs[m] - xo[m])**2 + (-zs[m] - zo[m])**2 + 1e-8)
            min_symm_dist = torch.minimum(torch.minimum(d_flip_x, d_flip_z), d_flip_xz)
            loss_vec[m] = torch.relu(min_symm_dist - close_threshold)

        # 9. bigger than: (vol_s - vol_o) / vol_s >= 0.15 <=> vol_o - 0.85 * vol_s <= 0
        m = (p == 8)
        if m.any():
            vol_s = ls[m].clamp(min=1e-3) * hs[m].clamp(min=1e-3) * ws[m].clamp(min=1e-3)
            vol_o = lo[m].clamp(min=1e-3) * ho[m].clamp(min=1e-3) * wo[m].clamp(min=1e-3)
            loss_vec[m] = torch.relu(vol_o - 0.85 * vol_s)

        # 10. smaller than: (vol_s - vol_o) / vol_s <= -0.15 <=> 1.15 * vol_s - vol_o <= 0
        m = (p == 9)
        if m.any():
            vol_s = ls[m].clamp(min=1e-3) * hs[m].clamp(min=1e-3) * ws[m].clamp(min=1e-3)
            vol_o = lo[m].clamp(min=1e-3) * ho[m].clamp(min=1e-3) * wo[m].clamp(min=1e-3)
            loss_vec[m] = torch.relu(1.15 * vol_s - vol_o)

        # 11. taller than: (top_s - top_o) / top_s >= 0.10 <=> top_o - 0.90 * top_s <= 0
        m = (p == 10)
        if m.any():
            top_s = ys[m] + hs[m].clamp(min=1e-3)
            top_o = yo[m] + ho[m].clamp(min=1e-3)
            loss_vec[m] = torch.relu(top_o - 0.90 * top_s)

        # 12. shorter than: (top_s - top_o) / top_s <= -0.10 <=> 1.10 * top_s - top_o <= 0
        m = (p == 11)
        if m.any():
            top_s = ys[m] + hs[m].clamp(min=1e-3)
            top_o = yo[m] + ho[m].clamp(min=1e-3)
            loss_vec[m] = torch.relu(1.10 * top_s - top_o)

        total_loss = total_loss + loss_vec.sum()

    if num_valid_relations > 0:
        total_loss = total_loss / num_valid_relations

    return total_loss





