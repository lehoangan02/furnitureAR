import json
import math
import os


SUPPORT_SURFACE_CATEGORIES = {
    "bed",
    "bookshelf",
    "cabinet",
    "desk",
    "nightstand",
    "shelf",
    "table",
    "tv_stand",
    "wardrobe",
}

SEATING_CATEGORIES = {
    "chair",
    "sofa",
}

ARCHITECTURAL_CATEGORIES = {
    "_scene_",
    "floor",
}


def _to_builtin(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    return value


def _to_list(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return _to_builtin(value)


def _to_float_list(value):
    return [float(v) for v in _to_list(value)]


def _to_int_list(value):
    return [int(v) for v in _to_list(value)]


def _clean_label(label):
    return str(label).strip()


def _abs_path(path):
    if path is None:
        return None
    return os.path.abspath(path)


def _expected_object_mesh_path(mesh_dir, category, category_id, mesh_export_index):
    if mesh_dir is None or mesh_export_index is None:
        return None
    filename = f"{category}_{category_id}_{mesh_export_index}.obj"
    return os.path.join(mesh_dir, filename)


def _infer_affordances(category):
    if category in ARCHITECTURAL_CATEGORIES:
        return ["architectural_surface"]

    affordances = ["scene_obstacle"]
    if category in SUPPORT_SURFACE_CATEGORIES:
        affordances.append("support_surface")
    if category in SEATING_CATEGORIES:
        affordances.append("sittable")
    if category == "lamp":
        affordances.append("illumination")
    return affordances


def _suggested_body_type(category):
    if category in ARCHITECTURAL_CATEGORIES:
        return "static"
    return "static"


def _bbox_corners(param7):
    length, height, width, px, py, pz, yaw_degrees = param7
    theta = math.radians(yaw_degrees)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    corners = []
    for ix in (-1.0, 1.0):
        for iy in (0.0, 1.0):
            for iz in (-1.0, 1.0):
                x = length * 0.5 * ix
                y = height * iy
                z = width * 0.5 * iz
                rot_x = x * cos_t + z * sin_t
                rot_z = -x * sin_t + z * cos_t
                corners.append([rot_x + px, y + py, rot_z + pz])
    return corners


def _aabb_from_corners(corners):
    mins = [min(point[axis] for point in corners) for axis in range(3)]
    maxs = [max(point[axis] for point in corners) for axis in range(3)]
    return {
        "min": mins,
        "max": maxs,
        "size": [maxs[axis] - mins[axis] for axis in range(3)],
        "center": [(mins[axis] + maxs[axis]) * 0.5 for axis in range(3)],
    }


def _room_bounds(objects):
    object_aabbs = [
        obj["aabb"]
        for obj in objects
        if obj["category"] not in ARCHITECTURAL_CATEGORIES and obj.get("aabb") is not None
    ]
    if not object_aabbs:
        return None

    mins = [min(aabb["min"][axis] for aabb in object_aabbs) for axis in range(3)]
    maxs = [max(aabb["max"][axis] for aabb in object_aabbs) for axis in range(3)]
    return {
        "min": mins,
        "max": maxs,
        "size": [maxs[axis] - mins[axis] for axis in range(3)],
        "center": [(mins[axis] + maxs[axis]) * 0.5 for axis in range(3)],
        "source": "generated_object_aabbs",
        "up_axis": "Y",
    }


def _architectural_surfaces(bounds):
    if bounds is None:
        return []

    min_x, min_y, min_z = bounds["min"]
    max_x, max_y, max_z = bounds["max"]
    return [
        {
            "id": "floor",
            "type": "floor",
            "aabb": {"min": [min_x, min_y, min_z], "max": [max_x, min_y, max_z]},
            "affordances": ["architectural_surface", "support_surface"],
            "suggested_body_type": "static",
        },
        {
            "id": "ceiling",
            "type": "ceiling",
            "aabb": {"min": [min_x, max_y, min_z], "max": [max_x, max_y, max_z]},
            "affordances": ["architectural_surface"],
            "suggested_body_type": "static",
        },
        {
            "id": "walls",
            "type": "walls",
            "aabb": {"min": [min_x, min_y, min_z], "max": [max_x, max_y, max_z]},
            "affordances": ["architectural_surface"],
            "suggested_body_type": "static",
        },
    ]


def _source_metadata_for_instance(source_object_metadata, instance_id):
    if source_object_metadata is None or instance_id is None:
        return {}
    info = source_object_metadata.get(instance_id)
    if info is None:
        info = source_object_metadata.get(str(instance_id))
    if not isinstance(info, dict):
        return {}
    return {
        "source_model_path": _abs_path(info.get("model_path")) if info.get("model_path") else None,
        "source_scale": _to_builtin(info.get("scale")),
    }


def build_structured_scene(
    scan_id,
    cat_ids,
    boxes,
    angles,
    triples,
    classes,
    predicate_names,
    instance_ids=None,
    mesh_dir=None,
    scene_mesh_path=None,
    render_type="echoscene",
    room_type=None,
    epoch=None,
    exp_path=None,
    dataset_path=None,
    source_object_metadata=None,
    excluded_render_categories=None,
    layout_guidance=None,
):
    cat_ids = _to_int_list(cat_ids)
    boxes = _to_list(boxes)
    angles = _to_list(angles)
    triples = _to_list(triples)
    instance_ids = _to_list(instance_ids) if instance_ids is not None else []
    excluded_render_categories = set(excluded_render_categories or [])

    objects = []
    mesh_export_index = 1
    for object_index, category_id in enumerate(cat_ids):
        category = _clean_label(classes[category_id])
        instance_id = instance_ids[object_index] if object_index < len(instance_ids) else None
        if instance_id is not None:
            instance_id = int(instance_id)

        yaw_value = angles[object_index]
        if isinstance(yaw_value, list):
            yaw_value = yaw_value[0]
        param7 = _to_float_list(list(boxes[object_index][:6]) + [yaw_value])
        corners = _bbox_corners(param7)
        aabb = _aabb_from_corners(corners)

        has_exported_object_mesh = category not in ARCHITECTURAL_CATEGORIES
        object_mesh_index = mesh_export_index if has_exported_object_mesh else None
        mesh_path = _expected_object_mesh_path(mesh_dir, category, category_id, object_mesh_index)
        if has_exported_object_mesh:
            mesh_export_index += 1

        top_y = param7[4] + param7[1]
        affordances = _infer_affordances(category)
        source_metadata = _source_metadata_for_instance(source_object_metadata, instance_id)

        objects.append(
            {
                "id": object_index,
                "instance_id": instance_id,
                "category_id": category_id,
                "category": category,
                "mesh_export_index": object_mesh_index,
                "mesh_path": _abs_path(mesh_path),
                "mesh_exists": bool(mesh_path and os.path.exists(mesh_path)),
                "mesh_coordinate_frame": "raw_object_mesh_fit_by_bbox",
                "included_in_scene_mesh": bool(
                    category not in ARCHITECTURAL_CATEGORIES
                    and category not in excluded_render_categories
                ),
                "bbox": {
                    "param7": param7,
                    "size": param7[:3],
                    "translation": param7[3:6],
                    "yaw_degrees": param7[6],
                    "up_axis": "Y",
                    "top_y": top_y,
                },
                "aabb": aabb,
                "corners": corners,
                "affordances": affordances,
                "support_surface": {
                    "enabled": "support_surface" in affordances,
                    "height": top_y,
                    "up_axis": "Y",
                },
                "physics": {
                    "suggested_body_type": _suggested_body_type(category),
                    "mass_kg": None,
                    "friction": None,
                    "restitution": None,
                    "collider": "mesh_or_convex_decomposition",
                },
                **source_metadata,
            }
        )

    relations = []
    for relation_index, triple in enumerate(triples):
        if len(triple) < 3:
            continue
        subject_id, predicate_id, object_id = [int(v) for v in triple[:3]]
        predicate = _clean_label(predicate_names[predicate_id])
        relation = {
            "id": relation_index,
            "subject_id": subject_id,
            "predicate_id": predicate_id,
            "predicate": predicate,
            "object_id": object_id,
        }
        if 0 <= subject_id < len(objects):
            relation["subject_category"] = objects[subject_id]["category"]
        if 0 <= object_id < len(objects):
            relation["object_category"] = objects[object_id]["category"]
        if predicate in {"standing on", "above"}:
            relation["affordance_relation"] = "support"
        relations.append(relation)

    bounds = _room_bounds(objects)
    return {
        "schema_version": "0.1",
        "generator": {
            "name": "EchoScene",
            "render_type": render_type,
            "epoch": str(epoch) if epoch is not None else None,
            "experiment_path": _abs_path(exp_path),
            "dataset_path": _abs_path(dataset_path),
        },
        "scan_id": str(scan_id),
        "room_type": room_type,
        "coordinate_system": {
            "up_axis": "Y",
            "bbox_format": "[length, height, width, x, y, z, yaw_degrees]",
            "units": "meters",
        },
        "scene_mesh_path": _abs_path(scene_mesh_path),
        "scene_mesh_exists": bool(scene_mesh_path and os.path.exists(scene_mesh_path)),
        "object_mesh_dir": _abs_path(mesh_dir),
        "objects": objects,
        "relations": relations,
        "room_bounds": bounds,
        "architectural_surfaces": _architectural_surfaces(bounds),
        "layout_guidance": _to_builtin(layout_guidance),
    }


def export_structured_scene(output_dir, **scene_kwargs):
    os.makedirs(output_dir, exist_ok=True)
    scene = build_structured_scene(**scene_kwargs)
    filename = f"{scene['scan_id']}_{scene['generator']['render_type']}.json"
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(scene, file, indent=2)
        file.write("\n")
    return output_path
