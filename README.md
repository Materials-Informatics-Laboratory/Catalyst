![Screenshot](https://github.com/Materials-Informatics-Laboratory/Catalyst/blob/main/visuals/catalyst2.png?raw=true)


**Catalyst** is a research-oriented Python package for building, training, and analyzing graph neural networks on atomistic and generic materials graphs. It is designed around a modular pipeline:

```text
graph data -> encoder -> processor/message passing -> decoder/readout -> Catalyst training backend
```

It is intended to make the core graph-learning workflow usable and reproducible while the package API is still being consolidated.

Catalyst currently supports:

- Generic graph data for non-atomistic or abstract graph-learning tasks.
- Atomistic ALIGNN-style graph data with atom, bond, and angle/order features.
- Equivariant atomistic graph models for vector-valued node targets such as force fields.
- A task interface that keeps model outputs, graph target fields, and Catalyst backend loss/prediction settings consistent.

---

## What is Catalyst?

Catalyst is a modular graph-learning framework for materials and scientific machine learning. It provides tools for:

1. Building graph representations of materials and generic graph data.
2. Constructing GNN models from encoders, processors, and decoders.
3. Training models through a shared Catalyst backend.
4. Running inference, saving predictions, and plotting parity/performance results.
5. Defining generic ML tasks such as graph-level scalar regression and node-level vector regression.

A typical Catalyst model is assembled from three pieces:

```text
encoder -> processor -> decoder
```

The **encoder** converts raw graph attributes into hidden features. The **processor** performs message passing, such as ALIGNN/order-based updates or equivariant updates. The **decoder** maps hidden features to a task-specific output such as a graph scalar or a node vector.

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

For force learning, the recommended route is now through the task interface:

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
task = GNNTask.graph_scalar(target_key="target_scalar")
task.apply_to_catalyst_parameters(parameters)
```

sets the Catalyst backend so it knows where to find the target and how to accumulate the loss.

---

## What is the difference between `graph_scalar`, `node_scalar`, `node_vector`, and `graph_vector`?

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
| `graph_scalar` | Graph | `[B]` or `[B, 1]` | Energy, stability, graph property | `exact` |
| `node_scalar` | Node | `[N]` or `[N, 1]` | Atomic charge, local scalar | `node` |
| `node_vector` | Node | `[N, 3]` | Forces, local vector field | `node` |
| `graph_vector` | Graph | `[B, 3]` | Dipole, polarization, global vector | `exact` |
| `scalar_gradient` | Node/gradient | Usually `[N, 3]` target | Forces from scalar potential | `node` |

---

## Installation

The recommended installation mode for the v2.1 is editable installation from the repository root.

```bash
git clone <your-catalyst-repository-url>
cd catalyst
python -m pip install -e .
```

For a fresh conda environment, a typical setup is:

```bash
conda create -n catalyst-dev python=3.10 -y
conda activate catalyst-dev

python -m pip install --upgrade pip
python -m pip install -e .
```

Depending on which examples you want to run, you may also need:

```bash
python -m pip install numpy scipy matplotlib scikit-learn networkx ase
python -m pip install torch torch-geometric
```

If you are using CUDA-enabled PyTorch, install the PyTorch build that matches your CUDA version using the official PyTorch installation instructions.

For development and testing:

```bash
python -m pip install pytest
```

Then run:

```bash
python -m pytest
```

---

## Minimal example: graph-level scalar task

This example shows the core task pattern without tying the task name to a physical property.

```python
from catalyst.ml.gnn import GNNTask, build_task_model

task = GNNTask.graph_scalar(target_key="target_scalar")

parameters = {
    "model_dict": {
        "prediction_params": {}
    }
}

task.apply_to_catalyst_parameters(parameters)

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

This does two separate things.

First, the task configures the Catalyst backend:

```python
task.apply_to_catalyst_parameters(parameters)
```

Second, the model is built:

```python
model = build_task_model(...)
```

For existing preset/custom-decoder models, use:

```python
apply_task_model_kwargs=False
```

This preserves the exact model architecture and uses the task only for backend consistency.

---

## Minimal example: node-level vector task

This is the recommended pattern for direct force-like vector prediction.

```python
from catalyst.ml.gnn import GNNTask, build_task_model

task = GNNTask.node_vector(
    target_key="target_vector",
    vector_channels=1,
)

parameters = {
    "model_dict": {
        "prediction_params": {}
    }
}

task.apply_to_catalyst_parameters(parameters)

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

v2.1 includes three smoke examples:

```bash
python examples/gnn_examples/smoke/01_generic_graph_scalar_smoke.py
python examples/gnn_examples/smoke/02_al_fcc_alignn_graph_scalar_smoke.py
python examples/gnn_examples/smoke/03_al_fcc_equivariant_node_vector_smoke.py
```

The examples are intended to check that the public API works, not to train a production model.

### Smoke example 1: generic graph scalar

This validates:

```python
GNNTask.graph_scalar(...)
task.apply_to_catalyst_parameters(...)
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

---

## Running tests

From the repository root:

```bash
python -m pytest unit_tests/test_gnn_tasks/test_gnn_tasks.py unit_tests/test_smoke_examples/test_smoke_examples_static.py
```

The task tests check that:

- `GNNTask.graph_scalar` applies the correct backend contract.
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

