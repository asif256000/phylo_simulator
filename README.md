# Phylogenetic Data Simulator

Phylo Simulator generates phylogenetic trees and aligned sequences, writes them to PhyloXML, and converts the results into NumPy-friendly arrays. The repository contains only simulation, XML parsing, and lightweight verification utilities.

## Setup

1. Use Python 3.10+ and create an isolated environment (`venv` or Conda recommended).
2. Install dependencies: `pip install -r requirements.txt`
3. Install an external sequence simulator (required for data generation):
   - [IQ-TREE](http://www.iqtree.org/)
   - [Seq-Gen](http://tree.bio.ed.ac.uk/software/seqgen/)

## Quick Start

### 1. Configure Generation

Create a configuration file in YAML or JSON (see templates in `sample_config/generation.{yaml,json}`):

```yaml
seed: 42
parallel_cores: 0  # Auto-detect available cores

tree:
  taxa_labels: [A, B, C]
  branch_length_distributions:
    uniform: 0.7
    exponential: 0.3
  branch_length_params:
    uniform:
      range: [0.1, 0.9]
    exponential:
      rate: 1.0
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
  output_name: "my_dataset"
  tree_chunk_size: 10000
```

For all available configuration options, see [CONFIG.md](CONFIG.md) or the [Data Generation Module documentation](src/data_generation/DATA_GENERATION.md).

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
- Branch length distributions (mixture of uniform, exponential, and truncated exponential)
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
