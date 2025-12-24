# Phylogenetic Data Simulator

Phylo Simulator generates phylogenetic trees and aligned sequences, writes them to PhyloXML, and converts the results into NumPy-friendly arrays. The repository contains only simulation, XML parsing, and lightweight verification utilities.

## Setup

1. Use Python 3.10+ and create an isolated environment (`venv` or Conda recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

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
- If `split_root_branch: true` (default): the root connector edge is split into two parts with a random pivot point. The minimum pivot position uses the lower bound from `uniform.range` if a uniform distribution is present, otherwise zero.
- If `split_root_branch: false`: both root edges are sampled independently as regular branches.

For two-taxon unrooted trees, the single sampled branch length is assigned to the first taxon's edge.

## Generate trees and sequences (PhyloXML)

```bash
python -m src.data_generation --config path/to/your/config.yaml
```

For example, using the sample configuration:

```bash
python -m src.data_generation --config config/generation.yaml
```

This writes `xml_data/<output_name>.xml` (or to the custom `xml_directory` if specified) containing the simulated phylogenies and sequences.

## Optional: verify Newick dumps

```bash
python -m src.data_generation.verify --config path/to/your/config.yaml
```

This emits `<xml_directory>/verify/<output_name>.txt` (where `<xml_directory>` is either the default `xml_data/` or your custom directory) with one Newick tree per line for quick inspection.

Programmatic use is also available:

```python
from src.data_generation import verify_from_config

verify_from_config("path/to/your/config.yaml")
```

Each invocation overwrites the corresponding `<xml_directory>/verify/<output_name>.txt` file with one Newick tree per line.

## Optional: export sequences to FASTA

```bash
python -m src.data_generation.verify_sequences --config path/to/your/config.yaml
```

This emits `<xml_directory>/verify/<output_name>_sequences.fasta` (where `<xml_directory>` is either the default `xml_data/` or your custom directory) containing all sequences from the PhyloXML dataset in FASTA format. The output location is automatically determined from the configuration file.

### FASTA format

The FASTA file includes sequences for all trees and all taxa in the order they appear in the configuration. Each sequence header includes the taxon label suffixed with an underscore and the tree index (1-based) to avoid duplicate taxa identifiers. For example, with taxa labels `[A, B, C]` and 2 trees, the output structure is:

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

Each sequence entry corresponds to a single alignment from a single tree. Sequences are written sequentially in tree order, and within each tree, in the order of the configured taxa labels.

Programmatic use is also available:

```python
from src.data_generation import verify_sequences_from_config

verify_sequences_from_config("path/to/your/config.yaml")
```

Each invocation overwrites the corresponding `<xml_directory>/verify/<output_name>_sequences.fasta` file.

## Parse PhyloXML to NumPy

```bash
python -m src.xml_parser --config path/to/your/config.yaml
```

The parser writes `npy_data/<output_name>.npy` (or to the custom `npy_directory` if specified) with fields:

- `X`: one-hot encoded sequences shaped `[taxa, length, channels]` (gap channel added when indels enabled).
- `y_br`: branch lengths in the fixed 2–4 taxa layouts (rooted doubles edges; unrooted stores only present edges).
- `branch_mask`: boolean mask for present branches.
- `y_top`: one-hot topology indicator.
- `tree_index`: original index from the XML file.

### Branch representation for NumPy matrices

The NumPy writer supports only 2–4 taxa and emits different layouts for rooted versus unrooted datasets:

- Rooted: branch vectors double each edge to keep a stable ordering regardless of where the root lands. Lengths are 2 (2 taxa), 6 (3 taxa), and 10 (4 taxa). The paired slots for a branch are both populated when the root splits that branch.
- Unrooted: branch vectors contain only present edges in deterministic order (no doubling). Lengths are 1 (2 taxa), 3 (3 taxa), and 5 (4 taxa). The `branch_mask` marks which slots were filled in the observed tree.

Datasets with more than four taxa are not supported by the NumPy writer.

## Testing

Run the test suite:

```bash
pytest
```

Tests cover tree/sequence generation workflows, XML parsing, dataset encoding, and one-hot encoding utilities.
