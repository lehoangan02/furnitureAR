"""One-file runner for the EchoScene layout model.

Run:
    python scenegraph_model.py input.json output.json
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ECHOSCENE = HERE / "echoscene_backend"
if str(ECHOSCENE) not in sys.path:
    sys.path.insert(0, str(ECHOSCENE))


def _graph(data):
    if not isinstance(data, dict) or not data.get("objects"):
        raise ValueError("input must contain a non-empty 'objects' list")
    objects = []
    ids = set()
    for index, item in enumerate(data["objects"]):
        if isinstance(item, str):
            item = {"id": f"object_{index}", "category": item}
        object_id = item.get("id", f"object_{index}")
        category = item.get("category", item.get("name"))
        if not object_id or not category or object_id in ids:
            raise ValueError(f"invalid or duplicate object at index {index}")
        ids.add(object_id)
        objects.append({"id": object_id, "category": category})

    relations = []
    for relation in data.get("relations", []):
        subject = relation.get("subject", relation.get("source"))
        target = relation.get("object", relation.get("target"))
        predicate = relation.get("predicate", relation.get("relation", "")).strip().lower()
        if subject not in ids or target not in ids or not predicate:
            raise ValueError(f"invalid relation: {relation}")
        relations.append({"subject": subject, "predicate": predicate, "object": target})
    return objects, relations, data.get("seed")


class EchoSceneRunner:
    def __init__(self, checkpoint=None, config=None, device=None, clip_features=True):
        import torch
        from omegaconf import OmegaConf

        self.torch = torch
        self.config_file = Path(config or HERE / "scene_config.json").expanduser().resolve()
        self.checkpoint = Path(checkpoint or HERE / "checkpoint/model2050.pth").expanduser().resolve()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_features = clip_features
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"checkpoint not found: {self.checkpoint}")
        if not self.config_file.exists():
            raise FileNotFoundError(f"application config not found: {self.config_file}")
        app_config = json.loads(self.config_file.read_text())
        self.classes = sorted(set(app_config["object_categories"]))
        self.class_to_id = {name: index for index, name in enumerate(self.classes)}
        self.relations = app_config["relations"]
        self.coordinate_space = app_config.get("output_coordinate_space", "normalized_echo_scene")
        if "_scene_" not in self.class_to_id:
            raise ValueError("scene_config.json must include the '_scene_' category")

        # ── Load model args (matches the released checkpoint's training config) ──
        model_args_file = HERE / "model_args.json"
        model_args = json.loads(model_args_file.read_text()) if model_args_file.exists() else {}

        # ── Resolve the training statistics file for box denormalization ──
        # The original eval pipeline uses:
        #   normalized_file = os.path.join(dataset, 'centered_bounds_{}_trainval.txt'.format(room_type))
        # and then calls descale_box_params(boxes, file=normalized_file)
        room_type = model_args.get("room_type", "all")
        self.normalized_file = str(HERE / "data" / f"centered_bounds_{room_type}_trainval.txt")
        if not Path(self.normalized_file).exists():
            # Fallback to 'all' if room-specific stats are missing
            self.normalized_file = str(HERE / "data" / "centered_bounds_all_trainval.txt")
        if not Path(self.normalized_file).exists():
            raise FileNotFoundError(
                f"Training stats file not found: {self.normalized_file}\n"
                "Copy centered_bounds_*_trainval.txt from the FRONT dataset into data/"
            )

        # ── Load diffusion config ──
        config_path = ECHOSCENE / "config/full_mp.yaml"
        cfg = OmegaConf.load(config_path)
        cfg.hyper.device = self.device

        # The original eval code sets train_stats_file from the dataset:
        #   diff_cfg.layout_branch.diffusion_kwargs.train_stats_file = dataset.box_normalized_stats
        # We set it to our bundled stats file so inference guidance can work.
        # For standalone inference without guidance, null is also safe.
        cfg.layout_branch.diffusion_kwargs.train_stats_file = self.normalized_file

        # Disable inference guidance for standalone inference (it needs floor_plan etc.)
        cfg.layout_branch.inference_guidance.enabled = False

        cfg.layout_branch.denoiser_kwargs.using_clip = clip_features
        cfg.shape_branch.vq_ckpt = str(HERE / "checkpoint/vqvae_threedfront_best.pth")
        for key in ("df_cfg", "vq_cfg"):
            path = Path(str(cfg.shape_branch[key]))
            if not path.is_absolute():
                cfg.shape_branch[key] = str((config_path.parent / path).resolve())

        from model.SGDiff import SGDiff
        # Build vocab exactly as the original dataset/eval code does:
        # The dataset adds '\n' to category names; our vocab must match.
        vocab = {
            "object_idx_to_name": [x + "\n" for x in self.classes],
            "object_idx_to_name_grained": [x + "\n" for x in self.classes],
            "pred_idx_to_name": [x + "\n" for x in self.relations],
        }
        self.model = SGDiff(
            type=model_args.get("network_type", "echoscene"), diff_opt=cfg, vocab=vocab,
            replace_latent=model_args.get("replace_latent", True),
            with_changes=model_args.get("with_changes", True),
            residual=model_args.get("residual", True),
            gconv_pooling=model_args.get("pooling", "avg"),
            with_angles=model_args.get("with_angles", True),
            clip=clip_features, separated=model_args.get("separated", False),
        )
        checkpoint_stem = self.checkpoint.stem
        epoch = checkpoint_stem[len("model"):] if checkpoint_stem.startswith("model") else checkpoint_stem
        self.model.load_networks(str(HERE), epoch,
                                 restart_optim=True, load_shape_branch=True)
        self.model.to(self.device).eval()

        self.clip = None
        if clip_features:
            try:
                import clip
                self.clip, _ = clip.load("ViT-B/32", device=self.device)
                self.clip.eval()
            except ImportError as error:
                raise ImportError("install OpenAI CLIP or run with --no-clip") from error

    def _features(self, values):
        import clip
        with self.torch.no_grad():
            return self.clip.encode_text(clip.tokenize(values).to(self.device)).float()

    def generate(self, data, output_path):
        from helpers.util import descale_box_params, postprocess_sincos2arctan

        objects, relations, seed = _graph(data)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            self.torch.manual_seed(seed)

        # Build graph: user objects + _scene_ room node at the end
        nodes = objects + [{"id": "room", "category": "_scene_"}]
        index = {item["id"]: i for i, item in enumerate(nodes)}
        unknown = sorted({item["category"] for item in objects} - set(self.class_to_id))
        if unknown:
            raise ValueError("unknown categories: " + ", ".join(unknown))

        # Build triples: user relations + "in" relation for each object → room
        triples, relation_text = [], []
        for relation in relations:
            if relation["predicate"] not in self.relations:
                raise ValueError(f"unknown relation: {relation['predicate']}")
            subject = nodes[index[relation["subject"]]]["category"]
            target = nodes[index[relation["object"]]]["category"]
            triples.append([index[relation["subject"]], self.relations.index(relation["predicate"]), index[relation["object"]]])
            relation_text.append(f"{subject} {relation['predicate']} {target}")
        for item in objects:
            triples.append([index[item["id"]], 0, index["room"]])  # 0 = "in"
            relation_text.append(f"{item['category']} in room")

        torch = self.torch
        obj_ids = torch.tensor([self.class_to_id[item["category"]] for item in nodes], dtype=torch.long, device=self.device)
        triple_ids = torch.tensor(triples, dtype=torch.long, device=self.device)

        # ── Compute CLIP features (matching original dataset code) ──
        if self.clip_features:
            # Object features: CLIP text encoding of each category name
            # The last node is _scene_ → use "room" as text (matching dataset line 400)
            obj_texts = [item["category"] for item in objects] + ["room"]
            object_features = self._features(obj_texts)
            # Relation features: CLIP text encoding of "<subject_cat> <predicate> <object_cat>"
            relation_features = self._features(relation_text)
        else:
            object_features = relation_features = None

        # ── Set objectness mask (original eval code lines 399-406) ──
        # Mark _scene_ as non-object so diffusion can handle it correctly
        objectness_mask = torch.ones(len(nodes), dtype=torch.bool, device=obj_ids.device)
        for i, node in enumerate(nodes):
            if node["category"] in ("_scene_", "floor"):
                objectness_mask[i] = False
        self.model.diff.current_objectness = objectness_mask
        # No ground-truth boxes for standalone inference
        self.model.diff.current_gt_boxes = None

        with torch.no_grad():
            result = self.model.sample_box_and_shape(
                obj_ids, triple_ids, object_features, relation_features, gen_shape=True
            )

        # ── Post-process layout: exactly like eval_3dfront.py ──
        # Concatenate sizes and translations: (N, 6) = [l, h, w, x, y, z]
        boxes_pred = torch.cat((result["sizes"], result["translations"]), dim=-1)

        # Angles: model outputs sin/cos → convert to angle in radians → degrees
        angles_pred = result["angles"]
        angles_pred = postprocess_sincos2arctan(angles_pred)  # (N, 1) radians
        angles_deg = angles_pred / np.pi * 180  # (N, 1) degrees

        # Denormalize boxes from [-1, 1] to metric using training stats
        # This is exactly what eval_3dfront.py does:
        #   boxes_pred_den = descale_box_params(boxes_pred, file=normalized_file)
        boxes_den = descale_box_params(boxes_pred, file=self.normalized_file)

        boxes_np = boxes_den.cpu().numpy()
        angles_np = angles_deg.cpu().numpy()
        angles_rad = angles_pred.cpu().numpy()

        # ── Export meshes if shapes were generated ──
        mesh_files = [None] * len(objects)
        glb_file = None
        shapes = result.get("shapes")
        if shapes is not None:
            try:
                mesh_files, glb_file = self._export_shapes(
                    shapes, boxes_np, angles_np, objects, output_path
                )
            except Exception as e:
                print(f"[!] mesh export failed: {e}", file=sys.stderr)

        # ── Build output (only user objects, not _scene_) ──
        out_objects = []
        for i, item in enumerate(objects):
            entry = {
                "id": item["id"],
                "category": item["category"],
                "size": boxes_np[i, :3].tolist(),       # [l, h, w] in meters
                "position": boxes_np[i, 3:6].tolist(),  # [x, y, z] in meters
                "rotation_y": float(angles_rad[i, 0]),  # radians
                "rotation_y_deg": float(angles_np[i, 0]),  # degrees (convenience)
            }
            if mesh_files[i] is not None:
                entry["mesh"] = mesh_files[i]
            out_objects.append(entry)

        output = {
            "objects": out_objects,
            "relations": relations,
            "coordinate_space": "metric",
        }
        if glb_file is not None:
            output["mesh_scene"] = glb_file
        return output

    def _export_shapes(self, shapes, boxes, angles_deg, objects, output_path):
        """Convert generated SDFs to placed OBJ meshes and one combined GLB."""
        import trimesh

        from helpers.util import fit_shapes_to_box_v2, pytorch3d_to_trimesh
        from model.diff_utils.util_3d import sdf_to_mesh

        output_path = Path(output_path).resolve()
        mesh_dir = output_path.with_name(output_path.stem + "_meshes")
        mesh_dir.mkdir(parents=True, exist_ok=True)

        mesh_batch = sdf_to_mesh(shapes, render_all=True)
        if mesh_batch is None or len(mesh_batch) < len(objects):
            raise RuntimeError("shape generation did not produce enough meshes")

        meshes = []
        mesh_files = []
        for index, item in enumerate(objects):
            mesh = pytorch3d_to_trimesh(mesh_batch[index])
            # fit_shapes_to_box_v2 expects [l, h, w, x, y, z, angle_degrees]
            box = np.concatenate((boxes[index], [float(angles_deg[index])]))
            _, mesh = fit_shapes_to_box_v2(mesh, box, degrees=True)
            mesh_path = mesh_dir / f"{index}_{item['id']}.obj"
            mesh.export(mesh_path)
            meshes.append(mesh)
            mesh_files.append(str(mesh_path.relative_to(output_path.parent)))

        glb_path = output_path.with_suffix(".glb")
        trimesh.Scene(meshes).export(glb_path)
        return mesh_files, str(glb_path.relative_to(output_path.parent))


def main():
    parser = argparse.ArgumentParser(description="Run EchoScene from a simple JSON scene graph")
    parser.add_argument("input", help="input JSON")
    parser.add_argument("output", help="output JSON")
    parser.add_argument("--checkpoint", help="model checkpoint; defaults to ./checkpoint/model2050.pth")
    parser.add_argument("--config", help="application scene config; defaults to ./scene_config.json")
    parser.add_argument("--device", choices=["cpu", "cuda"])
    args = parser.parse_args()

    runner = EchoSceneRunner(args.checkpoint, args.config, args.device, True)
    result = runner.generate(json.loads(Path(args.input).read_text()), Path(args.output))
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
