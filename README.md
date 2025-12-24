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
- `tree`: taxa labels, branch length range (applied per branch), rootedness flag, optional `branch_length_distribution` (currently only `uniform`), optional `split_root_branch` (defaults to `true`; when `false`, rooted trees draw both root edges independently instead of splitting the unrooted connector), and a required `topologies` list describing permitted tree structures.
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

Regardless of the requested rootedness, the generator first treats every topology as unrooted and draws independent branch segments from the configured range. These segments cover all edges of the unrooted skeleton. When the configuration is rooted, the default behavior (`tree.split_root_branch: true`) randomly splits the segment that connects the two root-side groups into two values—one for each child of the root—by drawing a pivot between the lower bound of the branch range and the sampled segment length. The two new edges therefore sum to the original unrooted branch, while downstream branches keep their original samples. When `tree.split_root_branch` is set to `false`, rooted trees draw every branch independently from the configured range (no splitting), so the root-side edges are unrelated samples. For unrooted two-taxon datasets, only a single segment is emitted and it is attached to the first taxon mentioned in the topology, leaving the companion tip with an implicit zero-length edge. All other unrooted trees retain the sampled lengths directly.

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
- `y_br`: branch lengths ordered deterministically for 2–4 taxa (or canonical ordering for larger cases).
- `branch_mask`: boolean mask for present branches.
- `y_top`: one-hot topology indicator.
- `tree_index`: original index from the XML file.

### Branch representation for NumPy matrices

The parser supports specialized branch vector formats for 2-, 3-, and 4-taxa trees. For these cases, the branch vector `y_br` has length equal to `2 × (2n - 3)` (the number of branches in the unrooted tree times two), with deterministic slot assignments:

- **2-taxa**: `[a, b]`
- **3-taxa**: `[a1, a2, b1, b2, c1, c2]`
- **4-taxa**: `[a1, a2, b1, b2, c1, c2, d1, d2, i1, i2]`

In each case, the `*1` slots correspond to the standard leaf branches (always present), while the `*2` slots hold zero by default. When a tree is rooted on a particular branch, that branch is split into two segments that populate both the `*1` and `*2` slots for the corresponding taxon or internal branch. This ensures consistent ordering regardless of root placement.

## Testing

Run the test suite:

```bash
pytest
```

Tests cover tree/sequence generation workflows, XML parsing, dataset encoding, and one-hot encoding utilities.
