# Data Generation Module

The `data_generation` module generates phylogenetic trees and aligned sequences in PhyloXML format. It supports configurable tree topologies, branch length distributions, and multiple sequence simulation backends.

## Configuration

Simulation inputs are specified in YAML or JSON configuration files. Templates are available in `sample_config/generation.{yaml,json}`.

**For comprehensive documentation of all configuration fields, defaults, and constraints, see [../../CONFIG.md](../../CONFIG.md).**

### Configuration Overview

The configuration file contains five main sections:

- **`seed`**: RNG seed for reproducibility.
- **`tree`**: Taxa labels, branch length distributions with weights, distribution-specific parameters, rootedness flag, optional `split_root_branch`, and required `topologies`.
- **`sequence`**: Sequence length and substitution model.
- **`simulation`**: Backend (`iqtree` or `seqgen`), executable paths, optional Seq-Gen keyword arguments, and indel parameters.
- **`dataset`**: Number of trees to simulate (`tree_count`), output file basename (`output_name`), and optional chunking (`tree_chunk_size`).
- **`parallel_cores`**: Level of parallelization. Defaults to `0` (auto-detect all available CPU cores). Set to `1` to disable parallelism when debugging.

### Custom Output Directories

By default, generated files are saved to `xml_data/` and `npy_data/` directories relative to the configuration file location. Override these by specifying custom paths:

```yaml
dataset:
  tree_count: 100
  output_name: "generated_trees"
  tree_chunk_size: 10000  # Optional: trees per write chunk (default 10000)
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

The generator assigns **one distribution per tree** based on the configured weights, then balances the number of trees per distribution across all topologies. This may increase the total number of trees beyond `dataset.tree_count` to keep sampling balanced.

**Supported Distributions**:
- **`uniform`**: Sample uniformly from a specified range `[min, max]`
- **`exponential`**: Sample from an exponential distribution with rate parameter λ
- **`truncated_exponential`**: Sample from an exponential distribution truncated to a bounded range `[min, max]`

**Configuration**:
```yaml
tree:
  branch_length_distributions:
    uniform: 0.5
    exponential: 0.3
    truncated_exponential: 0.2
  branch_length_params:
    uniform:
      range: [0.0, 0.1]
    exponential:
      rate: 10.0
    truncated_exponential:
      rate: 5.0
      min: 0.01
      max: 0.5
```

In this example:
- 50% of branch lengths are drawn uniformly from [0.0, 0.1]
- 30% of branch lengths are drawn from an exponential distribution with λ=10
- 20% of branch lengths are drawn from a truncated exponential (λ=5) bounded by [0.01, 0.5]

**Weight constraints**:
- All weights must be positive floats
- All weights must sum to exactly 1.0
- The parser will validate this during configuration loading

**Balancing rules**:
- For each distribution, the minimum count is $\lceil weight \times tree\_count \rceil$.
- These counts are split evenly across topologies.
- If rounding is needed, the total number of generated trees increases.

**Distribution-specific parameters**:
- **uniform**: Must provide `range: [min, max]` where min ≥ 0, max > 0, max > min
- **exponential**: Must provide `rate: <positive>` where rate > 0
- **truncated_exponential**: Must provide `rate: <positive>` and `max: <positive>`; `min` defaults to 0.0 if not specified, must satisfy 0.0 ≤ min < max. Sampling uses the closed-form inverse CDF so values are always within bounds (no rejection/resampling).

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

If `verify.padding_for_fasta: true`, sequences are padded with `*` to the maximum length observed across the XML file.

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
- **Sequences**: Aligned sequences for each taxon in each tree. Indel padding is applied during NumPy dataset creation as all-zero rows; deletion gaps remain `-`. IQ-TREE indel sizes can be configured via `simulation.indel.sizes` (passed through to AliSim).

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

