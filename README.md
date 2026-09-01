![Screenshot](https://github.com/Materials-Informatics-Laboratory/Catalyst/blob/main/visuals/catalyst_logo.png?raw=true)

## What is Catalyst?

Catalyst is a modular graph-learning framework for materials and scientific machine learning. It provides tools for:

1. Building graph representations of materials and generic graph data.
2. Constructing GNN models from encoders, processors, and decoders.
3. Training models through a shared Catalyst backend.
4. Running inference, saving predictions, and plotting parity/performance results.
5. Defining generic ML tasks such as graph-level scalar regression, multiple independent graph-level scalar regression, and node-level vector regression.
6. If using CUDA, Catalyst is built with DDP support allowing you to train your GNN models in a highly parallelized GPU environment. 

A typical Catalyst model is assembled from three pieces:

```text
encoder -> processor -> decoder
```

The **encoder** converts raw graph attributes into hidden features. The **processor** performs message passing, such as ALIGNN/order-based updates or equivariant updates. The **decoder** maps hidden features to a task-specific output such as a graph scalar, multiple independent graph scalars, or a node vector.

The high-level training object is:

```python
from catalyst.ml.gnn import GNN
```

The task interface is:

```python
from catalyst.ml.gnn import GNNTask, build_task_model
```

The public model builder is:

```python
from catalyst.ml.gnn import build_model
```

---

## Installation

For a standard installation:

```bash
python -m pip install catalyst-gnn
```

For plotting helpers and the repository examples:

```bash
python -m pip install "catalyst-gnn[examples]"
```

If you need a particular CUDA-enabled PyTorch build, install the appropriate
PyTorch build for your system first, then install Catalyst. An already-installed
compatible PyTorch satisfies Catalyst's dependency and will not be replaced.
Catalyst supports Python 3.10 and newer; the Python versions supported by a
specific PyTorch release may be more restrictive.

For development from a local clone:

```bash
git clone https://github.com/Materials-Informatics-Laboratory/Catalyst.git
cd Catalyst
python -m pip install -e ".[dev]"
```

To verify the installation:

```bash
python -c "import catalyst; print(catalyst.__version__)"
```

Then run the test suite from the repository root with:

```bash
python -m pytest
```

---

## Configuration and staged validation

Catalyst 2.2 uses one validated configuration path. Runtime parameters may be
read directly from JSON, supplied as a Python mapping, or combined. Explicit
constructor overrides take precedence over JSON values:

```python
from catalyst.observer import Catalyst
from catalyst.ml.gnn import GNNTask

task = GNNTask.graph_scalar(target_key="target_scalar")

cat = Catalyst(
    parameter_file="parameters.json",
    parameters={
        "loader_dict": {"batch_size": [8, 8]},
        "model_dict": {"num_epochs": 100},
    },
    task=task,
)
```

A JSON file may either contain Catalyst parameters directly or place them under
a top-level `catalyst_parameters` key, as the repository workflow examples do.
Relative I/O paths are resolved relative to the JSON file location.

Validation is staged because a model does not need to exist when Catalyst is
constructed:

```text
Catalyst(...)          -> configuration validation
cat.set_task(task)     -> task/configuration validation
build_task_model(...) -> task/model construction validation
run_training(...)      -> final configuration/task/model/runtime preflight
```

The task may therefore be created before or after Catalyst:

```python
cat = Catalyst(parameter_file="parameters.json")
task = GNNTask.graph_multiscalar(num_targets=3)
cat.set_task(task)
```

Task-owned settings such as `target_key`, `accumulate_loss`, `output_type`,
`output_level`, and multiscalar channel metadata are authoritative. If JSON or
constructor parameters explicitly contradict the selected task, Catalyst raises
`CatalystParameterError` rather than silently overriding either value. Unknown
parameter names are also rejected, including suggestions for likely typos.

Post-construction parameter changes should use the atomic validator:

```python
cat.set_params(
    {"model_dict": {"validation_interval": 5}},
    save_params=False,
)
```

Invalid updates leave the existing configuration unchanged. Direct mutation of
`cat.parameters` is not the supported public configuration path. The final
effective configuration can be inspected with `cat.print_parameters()` or saved
with `cat.save_parameters("effective_parameters.json")`.

---

## Training performance controls

Catalyst keeps performance-sensitive behavior backward compatible by default, but exposes opt-in controls for larger CPU/GPU workloads. These settings live in the ordinary Catalyst parameter dictionary.

### Mixed precision

```python
cat = Catalyst(parameters={
    "device_dict": {"use_amp": True, "amp_dtype": "float16"}
})
```

CUDA FP16 uses gradient scaling. BF16 does not require gradient scaling and is useful on hardware with native BF16 support.

### `torch.compile`

```python
cat = Catalyst(parameters={
    "model_dict": {
        "compile_model": True,
        "compile_backend": "inductor",
        "compile_mode": "default",
        "compile_dynamic": True,
    }
})
```

Catalyst uses dynamic compilation by default when compilation is enabled because atomistic graph batches naturally vary in their numbers of nodes and edges. Compilation remains disabled by default so users can benchmark it for their own model and PyTorch/PyG versions.

### Data loading and GPU prefetch

```python
cat = Catalyst(parameters={
    "device_dict": {"device": "cuda", "pin_memory": True},
    "loader_dict": {
        "num_workers": 4,
        "persistent_workers": True,
        "prefetch_factor": 2,
        "prefetch_to_device": True,
    },
})
```

`prefetch_to_device` uses PyTorch Geometric's `PrefetchLoader` for single-GPU CUDA training. It is intentionally disabled for Catalyst DDP because DDP needs direct access to its distributed sampler.

### Dynamic node/edge-budget batching

For datasets containing structures with very different sizes, a fixed number of graphs per mini-batch can lead to highly variable GPU memory use. Catalyst can instead batch to a node or edge budget:

```python
cat = Catalyst(parameters={
    "loader_dict": {"batch_mode": "nodes", "max_nodes": 12000}
})
```

or:

```python
cat = Catalyst(parameters={
    "loader_dict": {"batch_mode": "edges", "max_edges": 150000}
})
```

The default remains `batch_mode="graphs"`. Dynamic node/edge batching is currently a single-device feature.

### Optimizer implementation

```python
cat = Catalyst(parameters={
    "model_dict": {"optimizer_params": {"implementation": "auto"}}
})
```

Accepted values are `default`, `auto`, `fused`, `foreach`, and `for_loop`. `auto` prefers a supported fused implementation on CUDA and otherwise uses a supported foreach implementation. The default is `default` to preserve ordinary PyTorch optimizer selection.

### FP32 matmul/TF32 controls

```python
cat = Catalyst(parameters={
    "device_dict": {
        "float32_matmul_precision": "high",
        "allow_tf32": True,
    }
})
```

These controls are opt-in because they can change floating-point numerics slightly.

### Less-frequent validation

```python
cat = Catalyst(parameters={
    "model_dict": {"validation_interval": 5}
})
```

This trains every epoch but runs the full validation loader every five epochs, plus the final epoch. The default value is `1`. Ordinary validation no longer copies all predictions and targets back to the CPU unless `io_dict["write_indv_pred"]` is enabled.

### Distributed training

Catalyst 2.2 validates its DDP path as CUDA/NCCL multi-GPU training. Optional DDP performance controls include:

```python
cat = Catalyst(parameters={
    "device_dict": {
        "device": "cuda",
        "run_ddp": True,
        "world_size": 2,
        "ddp_backend": "nccl",
        "ddp_gradient_as_bucket_view": True,
        "ddp_static_graph": True,
        "ddp_bucket_cap_mb": 25,
    }
})
```

Only enable `ddp_static_graph` when the model uses the same parameter graph on every iteration.

---

## What graph types does Catalyst support?

Catalyst currently supports three broad graph families.

### 1. Generic graphs

Generic graphs are used for non-atomistic or abstract graph-learning workflows. They can represent arbitrary node and edge features and can be useful for testing, algorithm development, or materials descriptors that are not tied directly to atomic species and coordinates.

Typical use cases include:

- synthetic graph regression examples,
- generic graph embeddings,
- latent-space characterization,
- non-atomistic structure/property learning.

Relevant modules include:

```python
from catalyst.graph import generic_graph_gen
```

and generic graph data classes from:

```python
from catalyst.graph import Generic_Graph_Data
```

### 2. ALIGNN/order atomistic graphs

ALIGNN-style graphs represent atomistic structures using multiple graph orders. In the current Catalyst stack, this usually means:

```text
order 1: atoms
order 2: bonds / pair interactions
order 3: angles / line-graph interactions
```

These graphs are useful for scalar atomistic properties such as total energy, stability labels, graph-level descriptors, or other structure-level targets.

Typical use cases include:

- atomistic graph-scalar regression,
- ALIGNN-style energy learning,
- order-aware local environment representation,
- atom/bond/angle feature processing.

Relevant modules include:

```python
from catalyst.graph import alignn_gen
from catalyst.ml.gnn import build_model
```

A typical ALIGNN-style model is built with:

```python
model = build_model(
    preset="alignn",
    num_species=1,
    cutoff=3.5,
    dim=128,
    num_convs=4,
)
```

### 3. Equivariant atomistic graphs

Equivariant graphs include atom positions and edge geometry so that models can learn geometry-aware outputs. These are especially useful for vector-valued targets where rotation behavior matters.

Typical use cases include:

- force prediction,
- node-level vector regression,
- scalar-gradient-style models,
- geometry-aware atomistic learning.

Relevant model construction uses:

```python
model = build_model(
    model_type="equivariant",
    output_type="vector",
    output_level="node",
    num_species=1,
    cutoff=3.5,
    dim=128,
    num_convs=4,
    out_dim=1,
)
```

For vector learning, the recommended route is now through the task interface:

```python
task = GNNTask.node_vector(target_key="target_vector")

model = build_task_model(
    task=task,
    model_type="equivariant",
    num_species=1,
    cutoff=3.5,
    dim=128,
    num_convs=4,
)
```

---

## What tasks does Catalyst support?

Catalyst defines tasks using generic names. The task names describe the **shape and level of the prediction**, not the physical meaning of the target.

The current task names are:

```text
graph_scalar
graph_multiscalar
node_scalar
node_vector
graph_vector
scalar_gradient
```

These are defined in:

```python
from catalyst.ml.gnn import GNNTask
```

A task controls four things that must agree during training:

1. The model output type.
2. The output level, such as graph-level or node-level.
3. The graph target field, such as `target_scalar` or `target_vector`.
4. Catalyst backend settings for prediction and loss accumulation.

For example:

```python
from catalyst.observer import Catalyst

task = GNNTask.graph_scalar(target_key="target_scalar")
cat = Catalyst(parameter_file="parameters.json", task=task)
```

The task configures the Catalyst backend contract during staged validation so it knows where to find the target and how to accumulate the loss.

---

## What is the difference between `graph_scalar`, `graph_multiscalar`, `node_scalar`, `node_vector`, and `graph_vector`?

### `graph_scalar`

Use `graph_scalar` when each graph has one scalar target.

Examples:

- total energy,
- formation energy,
- graph-level property,
- stability score,
- scalar descriptor label.

Expected prediction shape:

```text
[B] or [B, 1]
```

where `B` is the number of graphs in the batch.

Example:

```python
task = GNNTask.graph_scalar(target_key="target_scalar")
```

This tells Catalyst that the target is stored on each graph or batch as:

```python
batch.target_scalar
```

and that the loss should compare one scalar prediction per graph.

### `graph_multiscalar`

Use `graph_multiscalar` when each graph has **K independent scalar targets** that should be predicted together by one model.

This is the task to use when the output is conceptually several separate scalar regression problems sharing the same GNN representation. The channels are ordinary invariant scalars and **do not** have the geometric or equivariant meaning of the three components of a vector.

Expected prediction and batched target shape:

```text
[B, K]
```

where `B` is the number of graphs in the batch and `K = num_targets`.

The canonical task-builder interface is:

```python
task = GNNTask.graph_multiscalar(
    num_targets=3,
    target_key="target_scalars",
    target_names=["property_a", "property_b", "property_c"],
)
```

`num_targets` must be at least 2. `target_names` is optional, but when provided it must contain exactly `num_targets` names.

The default task configuration is:

```text
output_type = "scalar"
output_level = "graph"
out_dim = num_targets
accumulate_loss = "exact"
normalize_by = "primary_nodes"
```

For non-equivariant Catalyst presets, `build_task_model` automatically selects the `MultiScalarDecoder` when no custom decoder is supplied. `MultiScalarDecoder` produces `K` scalar channels for each available graph order, and `GraphMultiScalarAdapter` performs the graph-level pooling needed to produce the final `[B, K]` prediction.

These channels are **not** passed through `VectorChannelAdapter`. For example, an output with shape `[B, 3]` from `graph_multiscalar(num_targets=3)` means three independent scalar predictions per graph, whereas `[B, 3]` from `graph_vector` means one three-component geometric vector per graph.

The task builder also accepts the aliases:

```text
graph_scalar_multichannel
scalar_multichannel
```

but `graph_multiscalar` is the canonical task name used by Catalyst.

### `node_scalar`

Use `node_scalar` when each node has one scalar target.

Examples:

- per-atom charge,
- per-node classification score,
- local scalar descriptor,
- atomic contribution label.

Expected prediction shape:

```text
[N] or [N, 1]
```

where `N` is the total number of nodes in the batch.

Example:

```python
task = GNNTask.node_scalar(target_key="target_scalar")
```

This tells Catalyst to accumulate the loss over nodes rather than over whole graphs.

### `node_vector`

Use `node_vector` when each node has one complete 3D vector target.

Examples:

- atomic forces,
- vector displacement targets,
- local vector fields.

Expected prediction shape:

```text
[N, 3]
```

where `N` is the total number of nodes in the batch.

Example:

```python
task = GNNTask.node_vector(target_key="target_vector")
```

Catalyst 2.2 supports one geometric vector channel for `node_vector` and `graph_vector`, so `vector_channels` must be `1`. Multiple independent scalar outputs belong in `graph_multiscalar`; a true multi-vector task is not part of the supported 2.2 API.

This does **not** define three independent scalar tasks. It defines one vector-valued task per node.

For equivariant models, the raw decoder may emit:

```text
[N, 1, 3]
```

Catalyst's task wrapper uses `VectorChannelAdapter` to convert this to:

```text
[N, 3]
```

before the loss function sees the prediction.

### `graph_vector`

Use `graph_vector` when each graph has one complete 3D vector target.

Examples:

- graph-level dipole vector,
- net polarization vector,
- global displacement vector.

Expected prediction shape:

```text
[B, 3]
```

where `B` is the number of graphs in the batch.

Example:

```python
task = GNNTask.graph_vector(target_key="target_vector")
```

Use `graph_vector` only when the three output components belong to one geometric vector. If the graph has several unrelated scalar labels, use `graph_multiscalar` instead.

### `scalar_gradient`

Use `scalar_gradient` for models that predict a scalar and train against the gradient of that scalar with respect to an input such as atomic positions.

Examples:

- force as negative energy gradient,
- conservative vector-field learning,
- scalar-potential-derived vector targets.

This task is experimental in the current release.

---

## Task summary

| Task | Prediction level | Output shape | Typical target | Catalyst loss accumulation |
|---|---:|---:|---|---|
| `graph_scalar` | Graph | `[B]` or `[B, 1]` | One graph-level scalar | `exact` |
| `graph_multiscalar` | Graph | `[B, K]` | K independent graph-level scalars | `exact` |
| `node_scalar` | Node | `[N]` or `[N, 1]` | Atomic charge, local scalar | `node` |
| `node_vector` | Node | `[N, 3]` | Forces, local vector field | `node` |
| `graph_vector` | Graph | `[B, 3]` | Dipole, polarization, global vector | `exact` |
| `scalar_gradient` | Node/gradient | Usually `[N, 3]` target | Forces from scalar potential | `node` |

---

## Minimal example: graph-level scalar task

This example shows the core task pattern without tying the task name to a physical property.

```python
from catalyst.observer import Catalyst
from catalyst.ml.gnn import GNNTask, build_task_model

task = GNNTask.graph_scalar(target_key="target_scalar")
cat = Catalyst(parameter_file="parameters.json", task=task)

model = build_task_model(
    task=task,
    preset="alignn",
    apply_task_model_kwargs=False,
    num_species=1,
    cutoff=3.5,
    dim=64,
    num_convs=2,
)
```

The Catalyst constructor validates the general configuration and binds the task contract. The model is then built independently:

```python
model = build_task_model(...)
```

For existing preset/custom-decoder models, use:

```python
apply_task_model_kwargs=False
```

This preserves the exact model architecture and uses the task only for backend consistency.

---

## Minimal example: multiple independent graph-level scalar targets

Use `graph_multiscalar` when one graph should produce several independent scalar predictions.

```python
from catalyst.observer import Catalyst
from catalyst.ml.gnn import GNNTask, build_task_model

task = GNNTask.graph_multiscalar(
    num_targets=3,
    target_key="target_scalars",
    target_names=["property_a", "property_b", "property_c"],
)
cat = Catalyst(parameter_file="parameters.json", task=task)

model = build_task_model(
    task=task,
    preset="alignn",
    num_species=1,
    cutoff=3.5,
    dim=64,
    num_convs=2,
)
```

For a batch containing `B` graphs, the final prediction has shape:

```text
[B, 3]
```

and is compared directly with a target tensor of the same shape.

With the standard non-equivariant preset route, the task builder automatically configures:

```text
decoder_type = "multiscalar"
output_type = "scalar"
output_level = "graph"
out_dim = 3
```

and wraps the model with `GraphMultiScalarAdapter` so the order-wise output from `MultiScalarDecoder` is accumulated into one independent scalar value per target and per graph.

If exact summed contributions are desired rather than the default normalization by the number of primary nodes, set:

```python
task = GNNTask.graph_multiscalar(
    num_targets=3,
    target_key="target_scalars",
    normalize_by=None,
)
```

---

## Minimal example: node-level vector task

This is the recommended pattern for direct force-like vector prediction.

```python
from catalyst.observer import Catalyst
from catalyst.ml.gnn import GNNTask, build_task_model

task = GNNTask.node_vector(
    target_key="target_vector",
    vector_channels=1,
)
cat = Catalyst(parameter_file="parameters.json", task=task)

model = build_task_model(
    task=task,
    model_type="equivariant",
    num_species=1,
    cutoff=3.5,
    dim=64,
    num_convs=2,
)
```

The task supplies:

```text
output_type = "vector"
output_level = "node"
out_dim = 1
```

The equivariant decoder may produce:

```text
[N, 1, 3]
```

The task wrapper converts it to:

```text
[N, 3]
```

so it matches:

```python
batch.target_vector
```

---

## Running smoke examples

v2.2 includes five smoke examples:

```bash
python examples/gnn_examples/smoke/01_generic_graph_scalar_smoke.py
python examples/gnn_examples/smoke/02_al_fcc_alignn_graph_scalar_smoke.py
python examples/gnn_examples/smoke/03_al_fcc_equivariant_node_vector_smoke.py
python examples/gnn_examples/smoke/04_al_fcc_alignn_graph_multiscalar_smoke.py
python examples/gnn_examples/smoke/05_al_fcc_train_checkpoint_inference_smoke.py
```

The examples are intended to check that the public API works, not to train a production model.

### Smoke example 1: generic graph scalar

This validates:

```python
GNNTask.graph_scalar(...)
cat.set_task(task)
validate_task_batch(...)
```

It uses a small dummy model and should run with minimal dependencies.

### Smoke example 2: Al FCC ALIGNN graph scalar

This validates the ALIGNN preset/custom-decoder passthrough route:

```python
model = build_task_model(
    task=task,
    preset="alignn",
    apply_task_model_kwargs=False,
    decoder=CustomReadout(...),
    ...
)
```

This is the safe route for existing graph-scalar models where you do not want the task object to alter the architecture.

### Smoke example 3: Al FCC equivariant node vector

This validates the equivariant task-native route:

```python
model = build_task_model(
    task=GNNTask.node_vector(...),
    model_type="equivariant",
    ...
)
```

This is the recommended route for direct vector-field prediction.

### Smoke example 4: Al FCC graph multiscalar

This exercises `GNNTask.graph_multiscalar`, `MultiScalarDecoder`, `GraphMultiScalarAdapter`, and the final `[B, K]` task contract on real ASE-generated atomic graphs.

### Smoke example 5: training, checkpoint, reload, and inference

This exercises the high-level Catalyst training backend end to end: graph input, optimization, loss decrease, checkpoint serialization, checkpoint reload, and inference consistency.

---

## Running tests

From the repository root, run the complete suite with:

```bash
python -m pytest -q
```

The functional GitHub Actions workflow runs the same suite on Python 3.10, 3.12, and 3.13. The suite includes task contracts, executable smoke examples, ASE graph-construction regressions, periodic-neighbor checks, equivariance/invariance checks, checkpoint/restart tests, staged parameter-validation tests, and Pass 1/Pass 2 release regressions.

The task tests check that:

- `GNNTask.graph_scalar` applies the correct backend contract.
- `GNNTask.graph_multiscalar` configures `num_targets`, target metadata, and `[B, K]` prediction/target validation correctly.
- `MultiScalarDecoder` produces independent K-channel scalar outputs and `GraphMultiScalarAdapter` pools them without vector semantics.
- `build_task_model` automatically selects the multiscalar decoder for the standard non-equivariant multiscalar route.
- `GNNTask.node_vector` applies the correct backend contract.
- `VectorChannelAdapter` converts `[N, 1, 3]` to `[N, 3]`.
- `VectorChannelAdapter` rejects `[N, 3, 3]` for one-vector tasks.
- `build_task_model` consumes `apply_task_model_kwargs` instead of forwarding it to the low-level builder.
- `build_task_model` can route preset-based models without leaking `model_type` into the wrong construction path.

---


## Recommended public API

Most users should start with:

```python
from catalyst.ml.gnn import (
    GNN,
    GNNTask,
    build_task_model,
    validate_task_batch,
    build_model,
)
```

For multiple independent graph-level scalar targets, use the canonical task-builder interface:

```python
task = GNNTask.graph_multiscalar(
    num_targets=3,
    target_key="target_scalars",
)
```

For graph builders:

```python
from catalyst.graph import alignn_gen, generic_graph_gen
```

For runtime parameters:

```python
from catalyst.observer import Catalyst
```

For direct model construction:

```python
from catalyst.ml.gnn.modules.models import build_model
```

