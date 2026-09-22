# furnitureAR Scene Graph Model

This subrepo packages the released EchoScene model for application use. It
accepts one small scene graph JSON file and writes generated furniture layout
JSON. It does not require the FRONT benchmark dataset at runtime.

The package includes:

- `scenegraph_model.py`: the simple command-line runner
- `scene_config.json`: model vocabulary and application output settings
- `example_scene.json`: working input example
- `checkpoint/model2050.pth`: released EchoScene checkpoint
- `checkpoint/vqvae_threedfront_best.pth`: required VQ-VAE checkpoint
- `echoscene_backend/`: vendored EchoScene model code and configuration
- `requirements-app.txt`: Python dependencies used by the runner

## Requirements

The original EchoScene environment was tested with Linux, CUDA, PyTorch 1.11,
and PyTorch3D. A GPU is strongly recommended because the model checkpoint is
large and diffusion inference is expensive.

PyTorch3D does not provide a universal macOS wheel. For reliable execution,
use a Linux/CUDA environment with a PyTorch3D build matching its PyTorch and
CUDA versions.

## Installation

Run all commands from this directory:

```bash
cd /Users/lehoangan/Documents/GitHub/ROOM/FunitureUnity/furnitureAR
```

Create or activate a compatible Python environment. Then install PyTorch and a
matching PyTorch3D build for the target machine. After that install the
application dependencies:

```bash
pip install -r requirements-app.txt
```

The first use of CLIP may download the `ViT-B/32` weights. Internet access is
needed unless those weights are already cached locally.

## Run

The simplest run uses the included example:

```bash
python scenegraph_model.py example_scene.json output/scene.json
```

The command writes the layout JSON plus generated shape assets:

- `output/scene.json`: layout and mesh references
- `output/scene.glb`: combined generated furniture scene
- `output/scene_meshes/*.obj`: one positioned mesh per object

The default input paths are:

- Checkpoint: `checkpoint/model2050.pth`
- Application config: `scene_config.json`
- Output: the path supplied as the second positional argument

Use custom paths when needed:

```bash
python scenegraph_model.py input.json output/scene.json \
  --checkpoint /path/to/model2050.pth \
  --config /path/to/scene_config.json \
  --device cuda
```

Available options:

```text
input                  Input scene graph JSON
output                 Output layout JSON
--checkpoint PATH      Model checkpoint
--config PATH          Application scene configuration
--device cuda|cpu      Inference device; default is CUDA when available
```

## Input Format

The input is an individual scene graph, not a dataset record:

```json
{
  "seed": 42,
  "objects": [
    {"id": "sofa", "category": "sofa"},
    {"id": "table", "category": "table"},
    {"id": "chair", "category": "chair"}
  ],
  "relations": [
    {"subject": "table", "predicate": "left", "object": "sofa"},
    {"subject": "chair", "predicate": "close by", "object": "table"}
  ]
}
```

Each object requires:

- `id`: unique identifier used by relations and output
- `category`: category listed in `scene_config.json`

Each relation requires:

- `subject`: source object ID
- `predicate`: relation listed in `scene_config.json`
- `object`: target object ID

The `seed` is optional. Set it when repeatable sampling is useful.

Objects may also be written as strings for quick tests:

```json
{
  "objects": ["sofa", "table"],
  "relations": []
}
```

The runner adds the internal `_scene_` room node automatically. Do not add it
to the input objects.

## Configuration

`scene_config.json` must match the vocabulary used to train the checkpoint.
The included configuration contains:

- 14 object categories, including `_scene_`
- 16 relation labels, including `in`
- `normalized_echo_scene` as the output coordinate space

If the application needs metric coordinates, add normalization ranges to the
config:

```json
{
  "object_categories": ["_scene_", "chair"],
  "relations": ["in", "left"],
  "output_coordinate_space": "metric",
  "normalization": {
    "size_min": [0.0, 0.0, 0.0],
    "size_max": [5.0, 5.0, 5.0],
    "position_min": [-5.0, -5.0, -5.0],
    "position_max": [5.0, 5.0, 5.0]
  }
}
```

Those ranges must correspond to the model's training normalization. Do not
invent application-specific ranges if exact metric dimensions are required.

## Output Format

The output JSON contains generated layout values:

```json
{
  "objects": [
    {
      "id": "sofa",
      "category": "sofa",
      "size": [1.2, 0.8, 2.1],
      "position": [0.1, 0.0, -0.4],
      "rotation_y": 1.57
    }
  ],
  "relations": [],
  "coordinate_space": "normalized_echo_scene"
}
```

`rotation_y` is in radians. `size` and `position` are normalized EchoScene
values unless the config provides valid normalization ranges.

## Unity Integration

Unity can invoke the runner as a process and then read the output JSON:

```text
python scenegraph_model.py scene.json generated_scene.json
```

The Unity side should map each output object's `id` to a prefab or asset,
apply `position`, `size`, and `rotation_y`, and convert from
`normalized_echo_scene` to the application's room coordinate system if no
metric normalization is configured.

For interactive applications, keep the model process alive or move inference
behind a small Python service rather than launching Python for every object.
The current runner generates one complete scene per invocation.

## Troubleshooting

### `ModuleNotFoundError: No module named 'pytorch3d'`

Install a PyTorch3D build compatible with the installed PyTorch and CUDA
versions. On macOS, use a Linux/CUDA machine or a supported remote inference
environment.

### `unknown categories: ...`

The category is not in `scene_config.json`. Add it only if it is a category
known by the checkpoint. Adding an arbitrary new category does not teach the
model how to generate it.

### `unknown relation: ...`

Use one of the predicates listed in `scene_config.json`, such as `left`,
`right`, `front`, `behind`, or `close by`.

### CLIP download or import errors

Install the OpenAI CLIP package through `requirements-app.txt` and make sure
the runtime can download or access the `ViT-B/32` weights.

### Out of memory

Use a CUDA GPU with sufficient memory. The released checkpoint is approximately
6.9 GB and model initialization also requires additional memory.
