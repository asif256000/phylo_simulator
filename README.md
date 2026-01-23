# Phylogenetic Data Simulator

Phylo Simulator generates phylogenetic trees and aligned sequences, writes them to PhyloXML, and converts the results into NumPy-friendly arrays. The repository contains only simulation, XML parsing, and lightweight verification utilities.

## Setup

1. Use Python 3.10+ and create an isolated environment (`venv` or Conda recommended).
2. Install dependencies: `pip install -r requirements.txt`
3. Install an external sequence simulator (required for data generation):
   - [IQ-TREE](http://www.iqtree.org/)
   - [Seq-Gen](http://tree.bio.ed.ac.uk/software/seqgen/)

## Configure generation

Simulation inputs live in YAML or JSON configuration files (templates available in `sample_config/generation.{yaml,json}`). The generator currently targets datasets with two, three, or four taxa. Key fields:

- `seed`: RNG seed for reproducibility.
- `tree`: taxa labels, rootedness flag, branch-length distribution mix (`branch_length_distributions`, e.g., `{uniform: 0.7, exponential: 0.3}`; weights must sum to 1), per-distribution parameters (`branch_length_params` with `uniform.range` and `exponential.rate`), optional `split_root_branch` (defaults to `true`; when `false`, rooted trees draw both root edges independently instead of splitting the unrooted connector), and a required `topologies` list describing permitted tree structures.
- `sequence`: sequence length and substitution model.
- `simulation`: backend (`iqtree` or `seqgen`), executable paths, optional Seq-Gen keyword arguments, and indel parameters.
- `dataset`: number of trees to simulate (`tree_count`) and the output file basename (`output_name`, no extension). By default, files are written to `xml_data/<output_name>.xml` and `npy_data/<output_name>.npy`. Optionally specify custom directories with `xml_directory` and `npy_directory` (see Custom Output Directories below).
- `parallel_cores`: controls the level of multiprocessing/threading used during tree generation and dataset encoding. Set to `1` to disable parallelism when debugging.

### Custom Output Directories

By default, generated files are saved to `xml_data/` and `npy_data/` directories relative to the configuration file location. You can override these defaults by specifying custom paths in your configuration:

```yaml
seed: 42
parallel_cores: 0  # Auto-detect available cores

tree:
  taxa_labels: [A, B, C]
  branch_length_range: [0.1, 0.9]
  topologies:
    - "((A,B),:C)"

sequence:
  length: 1000
  model: JC

simulation:
  backend: iqtree
  iqtree_path: "/path/to/iqtree3"

dataset:
  tree_count: 100
  output_name: "generated_trees"
  xml_directory: "/absolute/path/to/xml/output"  # Optional
  npy_directory: "/absolute/path/to/npy/output"  # Optional
```

Or in JSON:

```json
{
  "dataset": {
    "tree_count": 100,
    "output_name": "generated_trees",
    "xml_directory": "/absolute/path/to/xml/output",
    "npy_directory": "/absolute/path/to/npy/output"
  }
}
```

When custom directories are specified, all output files (including verify outputs) use those locations. If omitted, the default `xml_data/` and `npy_data/` directories are used.

### Topology strings

Provide one or more entries under `tree.topologies` for every configuration. Each entry must be written as a binary Newick fragment. When `tree.rooted` is `true`, wrap the two root children in parentheses and prefix exactly one child with `:` to mark the edge that carries the root—for instance, `((A,B),:C)` separates the `(A,B)` cherry from the rooted leaf `C`, and `(A,:(B,(C,D)))` represents `A` opposite a subtree where `B` splits before the cherry `(C,D)`. When `tree.rooted` is `false`, omit `:` entirely and use standard Newick notation such as `((p1,p2),(p3,p4))`. Inside each child, only single taxa or cherries like `(taxon_1,taxon_2)` are permitted, and every topology must reference each configured taxon exactly once. The generator cycles through the supplied strings to keep datasets evenly distributed, and the literal topology (including any root marker) is stored as a `<topology>` metadata entry.

### Branch lengths

Branch lengths are drawn from a distribution mixture specified by `branch_length_distributions`, where each distribution name maps to a weight (all weights must sum to 1). For each tree, the generator selects one distribution according to these weights and uses it for all branches in that tree. Parameters for each distribution are provided in `branch_length_params`.

**Supported distributions:**
- `uniform`: requires `branch_length_params.uniform.range` with two values `[min, max]`, where `min >= 0` and `max > min`. Branch lengths are sampled uniformly within this range.
- `exponential`: requires `branch_length_params.exponential.rate` as a positive rate parameter. Branch lengths are sampled from an exponential distribution with no upper bound.
- `truncated_exponential`: requires `branch_length_params.truncated_exponential.rate` (positive), `branch_length_params.truncated_exponential.max` (positive upper bound), and optionally `branch_length_params.truncated_exponential.min` (non-negative lower bound, defaults to 0). Branch lengths are drawn from an exponential distribution truncated to `[min, max]` using inverse-CDF sampling.

**Rooted tree handling:**
When `tree.rooted` is `true`, the generator first samples branch lengths for the unrooted tree backbone, then handles the root:
- If `split_root_branch: true` (default): the root connector edge is split into two parts with a random pivot point. The minimum pivot position uses the lower bound from the configured bounded distributions: `uniform.range[0]` if uniform is present, or `truncated_exponential.min` if truncated_exponential is present, otherwise zero.
- If `split_root_branch: false`: both root edges are sampled independently as regular branches.

For two-taxon unrooted trees, the single sampled branch length is assigned to the first taxon's edge.

### 2. Generate Trees and Sequences

```bash
python -m src.data_generation --config config/generation.yaml
```

This generates `xml_data/<output_name>.xml` containing your phylogenies and sequences.

### 3. Parse to NumPy

```bash
python -m src.xml_parser --config config/generation.yaml
```

This generates `npy_data/<output_name>.npy` with structured arrays ready for machine learning.

## Modules

### [Data Generation](src/data_generation/DATA_GENERATION.md)

Generates phylogenetic trees and aligned sequences in PhyloXML format. Features:

- Configurable tree topologies (rooted or unrooted)
- Branch length distributions
- Multiple sequence simulation backends (IQ-TREE, Seq-Gen)
- Indel simulation
- Automatic parallelization across available CPU cores
- Verification and export utilities

**Quick commands**:
```bash
python -m src.data_generation --config config.yaml  # Generate trees
python -m src.data_generation.verify --config config.yaml  # Export Newick
python -m src.data_generation.verify_sequences --config config.yaml  # Export FASTA
```

### [XML Parser](src/xml_parser/XML_PARSER.md)

Parses PhyloXML files and converts to NumPy arrays. Features:

- Sequence one-hot encoding
- Branch length extraction
- Topology encoding
- Support for 2–4 taxa datasets
- Gap channel for indel-enabled datasets

**Quick command**:
```bash
python -m src.xml_parser --config config.yaml  # Parse XML to NumPy
```

### Utils

Utility functions for phylogenetic operations (topology formatting, encoding, etc.).

## Configuration

For comprehensive documentation of all configuration fields and options, see [CONFIG.md](CONFIG.md).

Key highlights:

- **`seed`**: Reproducibility control
- **`parallel_cores`**: Defaults to `0` (auto-detect all cores); set to `1` for single-threaded debugging
- **`tree`**: Taxa labels, branch length range, rooted/unrooted, topologies
- **`sequence`**: Sequence length and evolutionary model
- **`simulation`**: Backend choice, executable paths, optional parameters
- **`dataset`**: Output count and naming

## Testing

Run the test suite:

```bash
pytest
```

Tests cover tree/sequence generation workflows, XML parsing, dataset encoding, and utility functions.

## Workflow

The typical phylogenetic data generation and parsing workflow:

```
1. Create config.yaml (or use sample from sample_config/)
              ↓
2. python -m src.data_generation --config config.yaml
   (generates xml_data/<name>.xml)
              ↓
3. [Optional] python -m src.data_generation.verify --config config.yaml
   (verify Newick trees)
              ↓
4. python -m src.xml_parser --config config.yaml
   (generates npy_data/<name>.npy)
              ↓
5. Load and use: data = np.load("npy_data/<name>.npy")
```

## Documentation Structure

- **[CONFIG.md](CONFIG.md)** - Complete configuration field reference
- **[DATA_GENERATION.md](src/data_generation/DATA_GENERATION.md)** - Tree and sequence generation details
- **[XML_PARSER.md](src/xml_parser/XML_PARSER.md)** - NumPy conversion and output format

Tests cover tree/sequence generation workflows, XML parsing, dataset encoding, and one-hot encoding utilities.
