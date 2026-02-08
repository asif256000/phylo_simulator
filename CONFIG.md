# Configuration Reference

This document describes all configuration fields accepted by the Phylo Simulator. Configurations can be provided in **YAML** or **JSON** format. See `sample_config/` for complete examples.

## Overview

A configuration file specifies how phylogenetic trees and sequences are generated. The top-level structure contains five main sections:

1. **`seed`** - Random number generator seed
2. **`parallel_cores`** - Parallelism control
3. **`tree`** - Tree generation parameters
4. **`sequence`** - Sequence simulation parameters
5. **`simulation`** - Backend and simulation options
6. **`dataset`** - Output locations and counts

---

## Top-Level Fields

### `seed`

**Type**: Integer  
**Required**: Yes  
**Default**: None  
**Description**: Random number generator seed for reproducibility. All tree generation, sequence simulation, and parallelization use this seed to ensure identical results across runs.

**Example**:
```yaml
seed: 123
```

---

### `parallel_cores`

**Type**: Non-negative integer  
**Required**: No  
**Default**: `0` (auto-detect all available CPU cores)  
**Description**: Number of cores to use for parallel tree and sequence generation. By default (`0`), the generator automatically detects and uses all available CPU cores for the current process. Set to `1` to disable parallelism (useful for debugging). When set to a value greater than `1`, the generator uses Python's multiprocessing with the "spawn" context to generate multiple trees concurrently using the specified number of cores.

**Constraints**:
- Must be a non-negative integer
- `0` enables auto-detection (falls back to `1` if detection fails)
- If not specified in the configuration, defaults to `0`
- Actual parallelization may be limited by available system cores

**Example**:
```yaml
parallel_cores: 4
```

---

## Tree Configuration (`tree`)

The `tree` section controls phylogenetic tree generation.

### `tree.taxa_labels`

**Type**: List of strings  
**Required**: Yes  
**Default**: None  
**Description**: Names of the taxa (species/sequences) to appear in the generated trees. Must contain at least one label. Each label must be unique within a configuration and must match those referenced in `tree.topologies`.

**Constraints**:
- Non-empty list
- Each label must be unique
- Labels are case-sensitive

**Example**:
```yaml
tree:
  taxa_labels:
    - species_A
    - species_B
    - species_C
```

---

### `tree.branch_length_distributions`

**Type**: Mapping of distribution name (string) to weight (positive float)  
**Required**: Yes  
**Default**: None  
**Description**: Mixture of branch length distributions with weights. Each weight must be positive, and all weights together must sum to 1.0 (the parser will error if they don't). Distributions are assigned **per tree** using the requested weights, then balanced across configured topologies. The generator may produce more trees than `dataset.tree_count` to ensure equal counts for every topology-distribution combination.

**Supported Distributions**:
- `uniform` - Uniform distribution over a specified range
- `exponential` - Exponential distribution with specified rate
- `truncated_exponential` - Exponential distribution truncated to a bounded range

**Weights**:
- Each weight must be a positive float (> 0)
- Weights must sum to exactly 1.0
- Non-integer weights are supported (e.g., 0.33, 0.67)

**Balancing rules**:
- For each distribution, the minimum count is $\lceil weight \times tree\_count \rceil$.
- Counts are then distributed evenly across topologies.
- If equal distribution requires rounding up, the total number of generated trees will increase beyond `dataset.tree_count`.


**Example**:
```yaml
tree:
  branch_length_distributions:
    uniform: 0.5
    exponential: 0.3
    truncated_exponential: 0.2
```
---
### `tree.branch_length_params`

**Type**: Mapping  
**Required**: Yes (when `branch_length_distributions` is specified)  
**Default**: `{}`

**Description**: Distribution-specific parameters. Each distribution listed in `branch_length_distributions` must have corresponding parameters defined here.

**Supported Parameters**:

#### **uniform**
- `range: [min, max]` - The range for uniform sampling
  - Must provide two numeric values
  - `min` ≥ 0, `max` > 0, `max` > `min`
  - Example: `range: [0.05, 0.8]`

#### **exponential**
- `rate: <positive number>` - The λ (lambda) parameter for exponential distribution
  - Must be a positive float
  - Example: `rate: 5.0`

#### **truncated_exponential**
- `rate: <positive number>` - The λ (lambda) parameter
  - `max: <positive number>` - Upper bound (required)
  - `min: <non-negative number>` - Lower bound (optional, defaults to 0.0)
  - Example: `rate: 2.0`, `min: 0.01`, `max: 1.0`

**Example with mixed distributions**:
```yaml
tree:
  branch_length_distributions:
    uniform: 0.5
    exponential: 0.3
    truncated_exponential: 0.2
  branch_length_params:
    uniform:
      range: [0.05, 0.8]
    exponential:
      rate: 5.0
    truncated_exponential:
      rate: 2.0
      min: 0.01
      max: 1.0
```

---

### `tree.rooted`

**Type**: Boolean  
**Required**: No  
**Default**: `true`  
**Description**: Whether generated trees are rooted or unrooted. Rooted trees require one child to be marked with `:` in each topology string.

**Possible Values**:
- `true` - Generate rooted trees
- `false` - Generate unrooted trees

**Example**:
```yaml
tree:
  rooted: true
```

---

### `tree.split_root_branch`

**Type**: Boolean  
**Required**: No  
**Default**: `true`  
**Description**: Applies only when `tree.rooted: true`. Controls how the root branch (the edge connecting the two root-side groups in the unrooted skeleton) is handled:

- `true` (default): The sampled root branch segment is split into two child edges. A pivot point is drawn uniformly between the minimum branch length and the sampled length. The left child gets the pivot value, the right child gets the remainder, ensuring both edges are positive and sum to the original sample.
- `false`: Both root-side edges are independently sampled from the configured distributions, with no relationship between them.

**Possible Values**:
- `true` - Split the root branch
- `false` - Draw root branches independently

**Example**:
```yaml
tree:
  rooted: true
  split_root_branch: true
```

---

### `tree.topologies`

**Type**: List of strings  
**Required**: Yes  
**Default**: None  
**Description**: One or more Newick-format tree structure templates. The generator cycles through these topologies to distribute trees evenly across the dataset.

**Topology Format**:

#### For Rooted Trees (`tree.rooted: true`):
- Wrap the two root children in parentheses and prefix exactly one with `:` to mark the rooted edge
- Examples:
  - Two taxa: `(A,:B)` – taxon A vs. rooted edge to B
  - Three taxa: `((A,B),:C)` – cherry (A,B) vs. rooted edge to C
  - Three taxa: `((A,C),:B)` – cherry (A,C) vs. rooted edge to B
  - Four taxa: `(((A,B),C),:D)` – subtree containing A, B, C vs. rooted edge to D

#### For Unrooted Trees (`tree.rooted: false`):
- Use standard binary Newick notation without `:` markers
- Examples:
  - Two taxa: `(A,B)`
  - Three taxa: `((A,B),C)`
  - Four taxa: `((A,B),(C,D))`

**Constraints**:
- At least one topology must be provided
- Each topology must be a non-empty string
- Every topology must reference **all** configured taxa exactly once (no duplicates, no missing taxa)
- For rooted trees, exactly one child must carry the `:` marker
- For rooted trees, both children must have taxa on their sides (cannot be empty)
- Inside each child, only:
  - Single taxa (leaves) are allowed, or
  - Cherries like `(taxon_1, taxon_2)` (exactly two leaves) are allowed
- Duplicate topologies are silently deduplicated
- The literal topology string (including any `:` marker) is stored as metadata in the output

**Example**:
```yaml
tree:
  topologies:
    - "((A,B),:C)"
    - "((A,C),:B)"
    - "((B,C),:A)"
```

---

## Sequence Configuration (`sequence`)

The `sequence` section controls sequence simulation parameters.

### `sequence.length`

**Type**: Positive integer  
**Required**: No  
**Default**: `1000`  
**Description**: The length of each simulated sequence (in nucleotides or amino acids, depending on the model).

When indels are enabled, this value is treated as the minimum sequence length. Padding to a common length is applied when generating NumPy datasets.

**Constraints**:
- Must be a positive integer (> 0)

**Example**:
```yaml
sequence:
  length: 1000
```

### `sequence.model`

**Type**: String  
**Required**: Yes  
**Default**: `JC`  
**Description**: Substitution model used for sequence simulation.

**Possible Values** (examples):
- `JC` - Jukes-Cantor (equal rates)
- `HKY` - Hasegawa-Kishino-Yano (transition/transversion bias)
- `GTR` - General Time Reversible
- `3.3b` - IQ-TREE specific notation

**Backend Specifics**:
- **IQ-TREE backend** (`simulation.backend: iqtree`): Accepts IQ-TREE model notation
- **Seq-Gen backend** (`simulation.backend: seqgen`): Accepts Seq-Gen model notation; additional parameters can be passed via `simulation.seqgen_kwargs`

**Example**:
```yaml
sequence:
  model: HKY
```

---

## Simulation Configuration (`simulation`)

The `simulation` section specifies how sequences are simulated given each tree.

### `simulation.backend`

**Type**: String  
**Required**: No  
**Default**: `"iqtree"`  
**Description**: The external program used to simulate sequences along the phylogenetic tree.

**Possible Values**:
- `"iqtree"` - Use IQ-TREE's sequence simulator
- `"seqgen"` - Use Seq-Gen

**Constraints**:
- Exactly one of the two values must be chosen
- The corresponding executable path must be provided (`simulation.iqtree_path` or `simulation.seqgen_path`)

**Example**:
```yaml
simulation:
  backend: iqtree
```

---

### `simulation.iqtree_path`

**Type**: String or null  
**Required**: No (but required if `simulation.backend: iqtree`)  
**Default**: `null`  
**Description**: Absolute or relative path to the IQ-TREE 3 executable (typically `iqtree3` or `iqtree3.exe` on Windows).

**Constraints**:
- Must be a non-empty string if provided
- Required when `simulation.backend: iqtree`
- Can be `null` if not using IQ-TREE

**Example**:
```yaml
simulation:
  iqtree_path: "/usr/local/bin/iqtree3"
```

---

### `simulation.seqgen_path`

**Type**: String or null  
**Required**: No (but required if `simulation.backend: seqgen`)  
**Default**: `null`  
**Description**: Absolute or relative path to the Seq-Gen executable (typically `seq-gen`).

**Constraints**:
- Must be a non-empty string if provided
- Required when `simulation.backend: seqgen`
- Can be `null` if not using Seq-Gen

**Example**:
```yaml
simulation:
  seqgen_path: "/usr/local/bin/seq-gen"
```

---

### `simulation.seqgen_kwargs`

**Type**: Dictionary/object or empty  
**Required**: No  
**Default**: `{}`  
**Description**: Optional keyword arguments to pass to the Seq-Gen simulator. These override or supplement default Seq-Gen parameters.

**Common Keys**:
- `ts_tv_ratio` (float) - Transition/transversion ratio (used with HKY, GTR models)
- `frequencies` (list of 4 floats) - Base frequencies [A, C, G, T], must sum to 1.0
- `additional_args` (list of strings) - Extra command-line arguments for Seq-Gen

**Example**:
```yaml
simulation:
  seqgen_kwargs:
    ts_tv_ratio: 2.0
    frequencies: [0.25, 0.25, 0.25, 0.25]
    additional_args: ["-k1"]
```

---

### `simulation.indel`

**Type**: Object/mapping  
**Required**: No  
**Default**: `{ enabled: false, rates: null }`  
**Description**: Configuration for indel (insertion/deletion) simulation. When enabled, indels are simulated along branches and the resulting alignment is stored in the output dataset with dedicated padding (`+`) and gap (`-`) channels. Padding is applied during NumPy dataset creation, not in the XML output.

---

### `simulation.indel.enabled`

**Type**: Boolean  
**Required**: No  
**Default**: `false`  
**Description**: Whether to enable indel simulation.

**Possible Values**:
- `false` - No indels; sequences remain unaligned
- `true` - Simulate indels; alignment with padding and gap channels is recorded

**Example**:
```yaml
simulation:
  indel:
    enabled: true
```

---

### `simulation.indel.rates`

**Type**: List of two floats or null  
**Required**: No  
**Default**: `null`  
**Description**: Indel rate parameters `[insertion_rate, deletion_rate]`. Only used if `simulation.indel.enabled: true`.

**Constraints**:
- Must be a list of exactly two floats if provided
- Can be `null` (rates use backend defaults)
- Each rate should be non-negative

**Example**:
```yaml
simulation:
  indel:
    enabled: true
    rates: [0.05, 0.05]
```

---

### `simulation.indel.sizes`

**Type**: List of two strings or null  
**Required**: No  
**Default**: `null`  
**Description**: Indel size distributions for insertion and deletion, passed directly to IQ-TREE AliSim. Provide two values: `[insertion_size, deletion_size]` (e.g., `POW{1.5/50}`, `GEO{5}`). Only used if `simulation.indel.enabled: true`.

**Constraints**:
- Must be a list of exactly two non-empty strings if provided
- AliSim size specification syntax is passed through as-is

**Example**:
```yaml
simulation:
  indel:
    enabled: true
    sizes: ["POW{1.5/50}", "GEO{5}"]
```

---

## Dataset Configuration (`dataset`)

The `dataset` section specifies output locations and the number of trees to generate.

### `dataset.tree_count`

**Type**: Positive integer  
**Required**: No  
**Default**: `1`  
**Description**: Minimum number of trees (and associated sequence alignments) to generate in this dataset. When multiple branch length distributions and topologies are configured, the generator may round up to keep the number of trees per topology-distribution combination balanced.

**Constraints**:
- Must be a positive integer (> 0)

**Balancing example**:
- `tree_count: 15`
- Distributions: uniform 0.4, exponential 0.3, truncated_exponential 0.3
- Topologies: 3
- Minimum distribution counts: $\lceil 6 \rceil, \lceil 4.5 \rceil, \lceil 4.5 \rceil$ → 6, 5, 5
- Balanced per-topology counts: 2 per topology per distribution → 6, 6, 6
- Total trees generated: 18

**Example**:
```yaml
dataset:
  tree_count: 100
```

---

### `dataset.tree_chunk_size`

**Type**: Positive integer  
**Required**: No  
**Default**: `10000`  
**Description**: Number of phylogenies to generate in memory before appending them to the output PhyloXML file. Larger values reduce I/O overhead but use more memory. Smaller values reduce peak memory usage when generating very large datasets.

**Constraints**:
- Must be a positive integer (> 0)

**Example**:
```yaml
dataset:
  tree_chunk_size: 10000
```

---

### `dataset.output_name`

**Type**: String  
**Required**: No  
**Default**: Uses legacy field `output_xml` if present, otherwise `"generated_dataset"`  
**Description**: Base name for output files (without extension). The generator creates `<output_name>.xml` and `<output_name>.npy` files using this name.

**Constraints**:
- Must be a non-empty string after stripping whitespace
- Path separators are stripped; only the filename stem is used
- Case-sensitive

**Processing**:
- If the value contains a path, only the file stem (name without extension) is extracted
- For example, `"path/to/my_dataset.txt"` becomes `"my_dataset"`

**Example**:
```yaml
dataset:
  output_name: "my_generated_trees"
```

---

### `dataset.xml_directory`

**Type**: String or null  
**Required**: No  
**Default**: `null` (uses `xml_data/` relative to the configuration file's parent)  
**Description**: Custom absolute or relative directory path for XML output. When `null`, the default location `xml_data/` is used (resolved relative to the configuration file).

**Constraints**:
- Must be a non-empty string if provided
- Can be `null` to use defaults
- Relative paths are resolved relative to the current working directory

**Default Resolution**:
When `null`, the generator searches for `xml_data/` or `npy_data/` directories relative to the config file. If found, that parent directory is used. Otherwise, it falls back to the config file's parent directory.

**Example**:
```yaml
dataset:
  xml_directory: "/absolute/path/to/xml/output"
```

---

### `dataset.npy_directory`

**Type**: String or null  
**Required**: No  
**Default**: `null` (uses `npy_data/` relative to the configuration file's parent)  
**Description**: Custom absolute or relative directory path for NumPy array output. When `null`, the default location `npy_data/` is used (resolved relative to the configuration file).

**Constraints**:
- Must be a non-empty string if provided
- Can be `null` to use defaults
- Relative paths are resolved relative to the current working directory

**Default Resolution**:
When `null`, the generator searches for `xml_data/` or `npy_data/` directories relative to the config file. If found, that parent directory is used. Otherwise, it falls back to the config file's parent directory.

**Example**:
```yaml
dataset:
  npy_directory: "/absolute/path/to/npy/output"
```

---

## Example Configurations

### Minimal Two-Taxa Configuration (IQ-TREE)

```yaml
seed: 42
parallel_cores: 2

tree:
  taxa_labels: [A, B]
  branch_length_distributions:
    uniform: 1.0
  branch_length_params:
    uniform:
      range: [0.1, 1.0]
  rooted: true
  topologies:
    - "(A,:B)"

sequence:
  length: 1000
  model: JC

simulation:
  backend: iqtree
  iqtree_path: "/path/to/iqtree3"

dataset:
  tree_count: 100
  output_name: "simple_2t_dataset"
```

---

### Three-Taxa with HKY Model (Seq-Gen)

```yaml
seed: 99
parallel_cores: 4

tree:
  taxa_labels: [Species_1, Species_2, Species_3]
  branch_length_distributions:
    uniform: 1.0
  branch_length_params:
    uniform:
      range: [0.05, 0.5]
  rooted: true
  split_root_branch: true
  topologies:
    - "((Species_1,Species_2),:Species_3)"
    - "((Species_1,Species_3),:Species_2)"

sequence:
  length: 750
  model: HKY

simulation:
  backend: seqgen
  seqgen_path: "/path/to/seq-gen"
  seqgen_kwargs:
    ts_tv_ratio: 2.0
    frequencies: [0.25, 0.25, 0.25, 0.25]

dataset:
  tree_count: 500
  output_name: "hky_3t_dataset"
  xml_directory: "/data/xml_output"
  npy_directory: "/data/npy_output"
```

---

### Four-Taxa with Indels

```yaml
seed: 777
parallel_cores: 8

tree:
  taxa_labels: [A, B, C, D]
  branch_length_distributions:
    uniform: 1.0
  branch_length_params:
    uniform:
      range: [0.05, 0.3]
  rooted: true
  topologies:
    - "(((A,B),C),:D)"
    - "(((A,C),B),:D)"

sequence:
  length: 2000
  model: GTR

simulation:
  backend: iqtree
  iqtree_path: "/usr/bin/iqtree3"
  indel:
    enabled: true
    rates: [0.1, 0.1]

dataset:
  tree_count: 5000
  output_name: "gtr_4t_indels"
```

---

## Error Handling & Validation

The configuration parser performs extensive validation and raises `ConfigurationError` for invalid inputs:

| Issue | Error Message |
|-------|---------------|
| Missing `seed` | `"Configuration missing required key: 'seed'"` |
| Non-integer `seed` | Raised during int conversion |
| `tree.taxa_labels` is empty | `"'tree.taxa_labels' must contain at least one label"` |
| `branch_length_distributions` not a mapping | `"'tree.branch_length_distributions' must be a mapping of name to weight"` |
| `branch_length_distributions` weights invalid | `"Branch length distribution weights must sum to 1"` |
| `branch_length_params` missing or invalid | `"'tree.branch_length_params' must be provided as a mapping"` |
| `sequence.length` ≤ 0 | `"'sequence.length' must be positive"` |
| `simulation.backend` not in ["iqtree", "seqgen"] | `"'simulation.backend' must be either 'iqtree' or 'seqgen'"` |
| `topologies` not provided or empty | `"'tree.topologies' must contain at least one unique definition"` |
| Topology references unknown taxon | `"Topology references unknown taxon '<name>'"` |
| Topology has duplicate taxa | `"Duplicate taxa found in topology '...'"` |
| Topology missing taxa | `"Each topology must reference all taxa exactly once; missing: ..."` |
| Rooted topology has wrong number of `:` markers | `"Rooted trees must mark exactly one child with ':'` |
| `parallel_cores` not non-negative | `"'parallel_cores' must be a non-negative integer (0 for auto-detect)"` |
| Invalid file format | `"Configuration file must be in YAML or JSON format"` |

---

## Default Values Summary

| Field | Default |
|-------|---------|
| `parallel_cores` | `0` (auto-detect all cores) |
| `tree.branch_length_distributions` | None (required) |
| `tree.branch_length_params` | None (required) |
| `tree.rooted` | `true` |
| `tree.split_root_branch` | `true` |
| `sequence.length` | `1000` |
| `sequence.model` | `"JC"` |
| `simulation.backend` | `"iqtree"` |
| `simulation.iqtree_path` | `null` |
| `simulation.seqgen_path` | `null` |
| `simulation.seqgen_kwargs` | `{}` (empty) |
| `simulation.indel.enabled` | `false` |
| `simulation.indel.rates` | `null` |
| `simulation.indel.sizes` | `null` |
| `dataset.tree_count` | `1` |
| `dataset.tree_chunk_size` | `10000` |
| `dataset.xml_directory` | `null` (resolves to `xml_data/`) |
| `dataset.npy_directory` | `null` (resolves to `npy_data/`) |

---

## Notes on Special Behaviors

### Root Branch Splitting
When a tree is rooted and `split_root_branch: true` (the default), the unrooted tree's root branch is split as follows:
1. A segment is sampled uniformly from `[min_length, max_length]`
2. A pivot point is drawn uniformly from `[min_length, segment_length]`
3. The left child gets the pivot value; the right child gets `segment_length - pivot`
4. Both child edges are positive and sum to the original segment

This ensures the split branch respects the range constraints while maintaining a consistent total.

### Two-Taxon Unrooted Trees
For unrooted two-taxon trees, only a single branch segment is generated and attached to the first taxon in the topology. The second taxon receives an implicit zero-length edge, representing the midpoint branch length.

### Topology Deduplication
If multiple identical topologies are provided, the generator silently deduplicates them. The deduplication uses the full topology structure (including `:` markers for rooted trees), so `(A,:B)` and `(B,:A)` are considered different.

### Metadata Storage
The literal topology string (as written in the configuration, including any `:` markers and whitespace) is stored in the PhyloXML output as a `<topology>` metadata tag for reproducibility and inspection.

