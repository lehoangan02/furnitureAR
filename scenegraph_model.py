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
        self.normalization = app_config.get("normalization")
        self.coordinate_space = app_config.get("output_coordinate_space", "normalized_echo_scene")
        if "_scene_" not in self.class_to_id:
            raise ValueError("scene_config.json must include the '_scene_' category")

        model_args_file = HERE / "model_args.json"
        model_args = json.loads(model_args_file.read_text()) if model_args_file.exists() else {}
        config_path = ECHOSCENE / "config/full_mp.yaml"
        if not config_path.is_absolute():
            config_path = (ECHOSCENE / config_path).resolve()
        cfg = OmegaConf.load(config_path)
        cfg.hyper.device = self.device
        cfg.layout_branch.diffusion_kwargs.train_stats_file = None
        cfg.layout_branch.denoiser_kwargs.using_clip = clip_features
        cfg.shape_branch.vq_ckpt = str(HERE / "checkpoint/vqvae_threedfront_best.pth")
        for key in ("df_cfg", "vq_cfg"):
            path = Path(str(cfg.shape_branch[key]))
            if not path.is_absolute():
                cfg.shape_branch[key] = str((config_path.parent / path).resolve())

        from model.SGDiff import SGDiff
        vocab = {
            "object_idx_to_name": [x + "\n" for x in self.classes],
            "object_idx_to_name_grained": [x + "\n" for x in self.classes],
            "pred_idx_to_name": [x + "\n" for x in self.relations],
        }
        self.model = SGDiff(
            type=model_args.get("network_type", "echoscene"), diff_opt=cfg, vocab=vocab,
            replace_latent=model_args.get("replace_latent", False),
            with_changes=model_args.get("with_changes", True),
            residual=model_args.get("residual", False),
            gconv_pooling=model_args.get("pooling", "avg"),
            with_angles=model_args.get("with_angles", False),
            clip=clip_features, separated=model_args.get("separated", False),
        )
        checkpoint_stem = self.checkpoint.stem
        epoch = checkpoint_stem[len("model"):] if checkpoint_stem.startswith("model") else checkpoint_stem
        self.model.load_networks(str(HERE), epoch,
                                 restart_optim=True, load_shape_branch=False)
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

    def generate(self, data):
        objects, relations, seed = _graph(data)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            self.torch.manual_seed(seed)

        nodes = objects + [{"id": "room", "category": "_scene_"}]
        index = {item["id"]: i for i, item in enumerate(nodes)}
        unknown = sorted({item["category"] for item in objects} - set(self.class_to_id))
        if unknown:
            raise ValueError("unknown categories: " + ", ".join(unknown))

        triples, relation_text = [], []
        for relation in relations:
            if relation["predicate"] not in self.relations:
                raise ValueError(f"unknown relation: {relation['predicate']}")
            subject = nodes[index[relation["subject"]]]["category"]
            target = nodes[index[relation["object"]]]["category"]
            triples.append([index[relation["subject"]], self.relations.index(relation["predicate"]), index[relation["object"]]])
            relation_text.append(f"{subject} {relation['predicate']} {target}")
        for item in objects:
            triples.append([index[item["id"]], 0, index["room"]])
            relation_text.append(f"{item['category']} in room")

        torch = self.torch
        obj_ids = torch.tensor([self.class_to_id[item["category"]] for item in nodes], dtype=torch.long, device=self.device)
        triple_ids = torch.tensor(triples, dtype=torch.long, device=self.device)
        if self.clip_features:
            object_features = self._features([item["category"] for item in objects] + ["room"])
            relation_features = self._features(relation_text)
        else:
            object_features = relation_features = None

        with torch.no_grad():
            result = self.model.sample_box_and_shape(obj_ids, triple_ids, object_features, relation_features)
        boxes = torch.cat((result["sizes"], result["translations"]), dim=1).cpu().numpy()
        boxes = self._descale(boxes)
        angles = result["angles"].cpu().numpy()
        angles = np.arctan2(angles[:, 0], angles[:, 1])
        return {
            "objects": [
                {"id": item["id"], "category": item["category"],
                 "size": boxes[i, :3].tolist(), "position": boxes[i, 3:6].tolist(),
                 "rotation_y": float(angles[i])}
                for i, item in enumerate(objects)
            ],
            "relations": relations,
            "coordinate_space": "metric" if self.normalization else self.coordinate_space,
        }

    def _descale(self, boxes):
        if not self.normalization:
            return boxes
        boxes = boxes.copy()
        size_min = np.asarray(self.normalization["size_min"], dtype=np.float32)
        size_max = np.asarray(self.normalization["size_max"], dtype=np.float32)
        position_min = np.asarray(self.normalization["position_min"], dtype=np.float32)
        position_max = np.asarray(self.normalization["position_max"], dtype=np.float32)
        boxes[:, :3] = (boxes[:, :3] + 1) / 2 * (size_max - size_min) + size_min
        boxes[:, 3:6] = (boxes[:, 3:6] + 1) / 2 * (position_max - position_min) + position_min
        return boxes


def main():
    parser = argparse.ArgumentParser(description="Run EchoScene from a simple JSON scene graph")
    parser.add_argument("input", help="input JSON")
    parser.add_argument("output", help="output JSON")
    parser.add_argument("--checkpoint", help="model checkpoint; defaults to ./checkpoint/model2050.pth")
    parser.add_argument("--config", help="application scene config; defaults to ./scene_config.json")
    parser.add_argument("--device", choices=["cpu", "cuda"])
    args = parser.parse_args()

    runner = EchoSceneRunner(args.checkpoint, args.config, args.device, True)
    result = runner.generate(json.loads(Path(args.input).read_text()))
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
