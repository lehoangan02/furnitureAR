# furnitureAR

## Run EchoScene

This folder contains a runnable EchoScene application package. It includes the
`model2050.pth` checkpoint and VQ-VAE checkpoint. The model implementation is
vendored in `echoscene_backend/`.

Install the runtime dependencies first. Use a PyTorch/PyTorch3D build matching
your machine, then install the application additions:

```bash
pip install -r requirements-app.txt
```

EchoScene's original runtime expects a compatible PyTorch3D installation. The
released environment was Linux/CUDA; PyTorch3D does not provide a normal
universal macOS wheel.

Run from this folder:

```bash
python scenegraph_model.py example_scene.json output/scene.json
```

No FRONT benchmark dataset is needed. `scene_config.json` contains the model's
object vocabulary and relation vocabulary. Input is only an individual scene
graph: furniture objects and optional spatial relations. Output is JSON
containing each object's generated `size`, `position`, and `rotation_y`
(radians). Coordinates are in EchoScene's normalized output space.

The first run may download OpenAI CLIP's ViT-B/32 weights if they are not
already cached.
