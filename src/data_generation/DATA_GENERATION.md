# Data Generation Module

The `data_generation` module generates phylogenetic trees and aligned sequences in PhyloXML format. It supports configurable tree topologies, branch length distributions, and multiple sequence simulation backends.

## Configuration

Simulation inputs are specified in YAML or JSON configuration files. Templates are available in `sample_config/generation.{yaml,json}`.

**For comprehensive documentation of all configuration fields, defaults, and constraints, see [../../CONFIG.md](../../CONFIG.md).**

### Configuration Overview

The configuration file contains five main sections:

- **`seed`**: RNG seed for reproducibility.
- **`tree`**: Taxa labels, branch length range, rootedness flag, optional `branch_length_distribution`, optional `split_root_branch`, and required `topologies`.
- **`sequence`**: Sequence length and substitution model.
- **`simulation`**: Backend (`iqtree` or `seqgen`), executable paths, optional Seq-Gen keyword arguments, and indel parameters.
- **`dataset`**: Number of trees to simulate (`tree_count`) and output file basename (`output_name`).
- **`parallel_cores`**: Level of parallelization. Defaults to `0` (auto-detect all available CPU cores). Set to `1` to disable parallelism when debugging.

### Custom Output Directories

By default, generated files are saved to `xml_data/` and `npy_data/` directories relative to the configuration file location. Override these by specifying custom paths:

```yaml
dataset:
  tree_count: 100
  output_name: "generated_trees"
  xml_directory: "/absolute/path/to/xml/output"  # Optional
  npy_directory: "/absolute/path/to/npy/output"  # Optional
```

When custom directories are specified, all output files use those locations.

### Topology Strings

Provide one or more entries under `tree.topologies` using binary Newick format.

**For rooted trees** (`tree.rooted: true`):
- Wrap the two root children in parentheses and prefix exactly one with `:` to mark the rooted edge.
- Examples: `(A,:B)`, `((A,B),:C)`, `(((A,B),C),:D)`

**For unrooted trees** (`tree.rooted: false`):
- Use standard Newick notation without `:` markers.
- Examples: `(A,B)`, `((A,B),C)`, `((A,B),(C,D))`

Rules:
- Each topology must reference all configured taxa exactly once (no duplicates, no missing taxa).
- Inside each child, only single taxa or cherries `(taxon_1,taxon_2)` are permitted.
- Duplicate topologies are silently deduplicated.
- The literal topology string is stored as metadata in the output.

### Branch Lengths

The generator first treats every topology as unrooted and draws independent branch segments from the configured `branch_length_range` or mixture of distributions.

**Distribution options**:
- **Single distribution** (legacy): Use `branch_length_distribution` (uniform, exponential, etc.) with `branch_length_range` parameters.
- **Mixture distributions** (recommended): Use `branch_length_distributions` (mapping of distribution names to weights) with `branch_length_params` for per-distribution parameters. Supported: `uniform` (with `range` parameter) and `exponential` (with `rate` parameter).

Example mixture with 70% uniform and 30% exponential:
```yaml
tree:
  branch_length_distributions:
    uniform: 0.7
    exponential: 0.3
  branch_length_params:
    uniform:
      range: [0.0, 0.1]
    exponential:
      rate: 10.0
```

**When rooted** (`tree.rooted: true`) **with default split behavior** (`tree.split_root_branch: true`):
- The root branch segment is split into two child edges.
- A pivot point is drawn uniformly between the minimum branch length and the sampled length.
- Both child edges are positive and sum to the original sample.

**When rooted with independent sampling** (`tree.split_root_branch: false`):
- Both root-side edges are independently sampled from the range (no splitting).

**For unrooted two-taxon trees**:
- Only a single branch segment is generated and attached to the first taxon.
- The second taxon receives an implicit zero-length edge.

## Generate Trees and Sequences

### Command-Line Usage

```bash
python -m src.data_generation --config path/to/your/config.yaml
```

Example:

```bash
python -m src.data_generation --config config/generation.yaml
```

This writes `xml_data/<output_name>.xml` (or to the custom `xml_directory` if specified) containing the simulated phylogenies and sequences.

### Programmatic Usage

```python
from src.data_generation import TreeSequenceGenerator

generator = TreeSequenceGenerator.from_config_file("path/to/config.yaml")
output_path = generator.write_xml()
```

## Verify Generated Trees

### Export Newick Trees

```bash
python -m src.data_generation.verify --config path/to/your/config.yaml
```

This emits `<xml_directory>/verify/<output_name>.txt` with one Newick tree per line for quick inspection.

Programmatic use:

```python
from src.data_generation import verify_from_config

verify_from_config("path/to/your/config.yaml")
```

### Export Sequences to FASTA

```bash
python -m src.data_generation.verify_sequences --config path/to/your/config.yaml
```

This emits `<xml_directory>/verify/<output_name>_sequences.fasta` containing all sequences in FASTA format.

#### FASTA Format

Each sequence header includes the taxon label suffixed with underscore and tree index (1-based). For example, with taxa `[A, B, C]` and 2 trees:

```
>A_1
ATGCATGCATGC...
>B_1
GCTAGCTAGCTA...
>C_1
TTAAATTAAATT...
>A_2
ATGCATGCATGC...
>B_2
GCTAGCTAGCTA...
>C_2
TTAAATTAAATT...
```

Sequences are written in tree order, and within each tree, in the order of configured taxa labels.

Programmatic use:

```python
from src.data_generation import verify_sequences_from_config

verify_sequences_from_config("path/to/your/config.yaml")
```

## Output

The generated PhyloXML file contains:

- **Phylogenies**: Each tree with simulated sequences.
- **Metadata**: Topology definition, branch lengths, taxon labels, and sequence length.
- **Sequences**: Aligned sequences for each taxon in each tree.

The file is compatible with standard PhyloXML tools and can be parsed by the `xml_parser` module for conversion to NumPy arrays.

## Key Classes

### `GenerationConfig`

Represents the parsed configuration. Created via:

```python
from src.data_generation.config import GenerationConfig

config = GenerationConfig.from_mapping(data, base_path=Path("."))
# or
config = load_generation_config("path/to/config.yaml")
```

### `TreeSequenceGenerator`

Main generator class:

```python
from src.data_generation import TreeSequenceGenerator

generator = TreeSequenceGenerator(config)
phylogenies, aligned = generator.generate_phylogenies()
output_path = generator.write_xml()
```

### `TreeSequenceResult`

Container for a single tree-sequence pair:

```python
@dataclass
class TreeSequenceResult:
    tree: BaseTree
    sequences: dict[str, str]
    aligned: bool
    topology: TopologySpec
```

