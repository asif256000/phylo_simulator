# Tests Reference

This document summarizes test inputs, fixtures, and expectations for the test suite.

## Fixtures And Helpers

### Common Inputs

- **Two-taxa IQ-TREE config (uniform)**
  - seed: 42 or 7/11/13 in specific tests
  - parallel_cores: 1
  - tree: taxa_labels [A, B], rooted true, topologies ["(A,:B)"], uniform range varies
  - sequence length varies by test (4, 6, 8)
  - simulation backend: iqtree, fake paths, indel disabled unless noted
  - dataset: tree_count varies, output_name varies

- **Three-taxa IQ-TREE config (uniform)**
  - taxa_labels [A, B, C]
  - rooted true, topologies ["((A,B),:C)"] or two-topology mix
  - sequence length 4 or 6

- **Four-taxa IQ-TREE config (uniform or truncated exponential)**
  - taxa_labels [A, B, C, D] or [t1, t2, t3, t4]
  - rooted true or false depending on test

### test_inputs.py

- **branch_lengths(tree)**
  - Input: Bio.Phylo tree
  - Output: list of non-null branch lengths
  - Used in: test_generate_tree_and_sequences, test_root_split_preserves_total_length, test_rooted_no_split_draws_independent_edges, test_unrooted_two_taxa_assigns_single_branch, test_three_taxa_tree_respects_topology, test_four_taxa_tree_supports_double_cherries

- **generation_config**
  - Input: two-taxa IQ-TREE config (seed 42, length 8, tree_count 3, tree_chunk_size 2)
  - Used in: test_generate_tree_and_sequences, test_generate_phylogenies_respects_count, test_write_xml_creates_expected_phyloxml, test_phylogeny_stores_newick_metadata

- **config**
  - Input: two-taxa IQ-TREE config (seed 7, length 6, tree_count 1)
  - Used in: test_parse_examples_extracts_clades, test_build_dataset_creates_structured_npy, test_one_hot_encode_validates_length

- **phyloxml_file**
  - Input: config + monkeypatched IQ-TREE simulator returning A/C sequences
  - Output: XML file path created by TreeSequenceGenerator.write_xml
  - Used in: test_parse_examples_extracts_clades

## test_data_generation.py

- **test_generate_tree_and_sequences**
  - Inputs: generation_config, monkeypatched _simulate_with_iqtree
  - Expect: 2 terminals, branch lengths within uniform range, correct count, sequences match, aligned true, topology from config

- **test_generate_phylogenies_respects_count**
  - Inputs: generation_config, monkeypatched _simulate_with_iqtree
  - Expect: number of phylogenies equals tree_count, aligned true, topology metadata equals "(A,:B)"

- **test_write_xml_creates_expected_phyloxml**
  - Inputs: generation_config, monkeypatched _simulate_with_iqtree
  - Expect: XML exists, parsed count equals tree_count, topology metadata matches

- **test_verify_module_emits_newick_dump**
  - Inputs: two-taxa config with tree_count 2
  - Expect: verify output path in xml_data/verify, 2 Newick lines, each ends with ";"

- **test_verify_module_with_custom_output_path**
  - Inputs: custom output path for Newick dump
  - Expect: output path matches input, file exists, correct line count

- **test_verify_module_raises_when_xml_missing**
  - Inputs: config without generating XML
  - Expect: FileNotFoundError

- **test_indel_sizes_parsed_from_config**
  - Inputs: indel enabled with sizes ["POW{1.5/50}", "GEO{5}"]
  - Expect: config.simulation.indel.sizes matches tuple

- **test_verify_module_with_custom_xml_directory**
  - Inputs: custom xml_directory
  - Expect: verify output path under custom directory, count and format correct

- **test_verify_sequences_module_emits_fasta_dump**
  - Inputs: two-taxa config with tree_count 2
  - Expect: FASTA output path and exact sequence order A_1, B_1, A_2, B_2

- **test_verify_sequences_preserves_gaps**
  - Inputs: indel-enabled config, sequences with '-' gaps
  - Expect: FASTA output preserves '-' characters

- **test_verify_sequences_module_with_custom_xml_directory**
  - Inputs: custom xml_directory
  - Expect: FASTA output path under custom directory and correct contents

- **test_seqgen_stdout_parsing**
  - Inputs: seqgen backend with fake subprocess stdout
  - Expect: parsed sequences match stdout, command includes -of and tree file

- **test_seqgen_reads_output_file**
  - Inputs: seqgen backend with output file written in temp dir
  - Expect: sequences read from file

- **test_seqgen_rejects_multiple_replicates**
  - Inputs: seqgen backend, replicates=2
  - Expect: ValueError

- **test_topologies_required**
  - Inputs: missing tree.topologies
  - Expect: ConfigurationError

- **test_rooted_topology_requires_colon**
  - Inputs: rooted topology without ':'
  - Expect: ConfigurationError

- **test_unrooted_topology_ignores_colon**
  - Inputs: unrooted topology with ':'
  - Expect: RuntimeWarning, rooted flag false

- **test_topology_rejects_duplicate_taxa**
  - Inputs: topology with duplicate taxa
  - Expect: ConfigurationError with Duplicate taxa

- **test_branch_length_distribution_validation**
  - Inputs: invalid distribution name
  - Expect: ConfigurationError

- **test_exponential_distribution_uses_rate**
  - Inputs: exponential distribution with rate 2.5, patched RNG
  - Expect: expovariate called with 2.5, returned value propagated

- **test_truncated_exponential_bounds**
  - Inputs: truncated exponential with min 0.1 max 0.5, RNG returns 0.0
  - Expect: sampled branch length equals min (0.1)

- **test_indel_sizes_passed_to_iqtree**
  - Inputs: indel enabled with rates and sizes
  - Expect: _simulate_with_iqtree receives indel_rate and indel_size tuples

- **test_split_root_branch_flag_parsing**
  - Inputs: split_root_branch false
  - Expect: config.tree.split_root_branch is False

- **test_topology_cycle_even_distribution**
  - Inputs: two topologies, tree_count 5, monkeypatched simulator
  - Expect: 3 trees of first topology, 2 of second

- **test_root_insertion_preserves_neighbor_pairs**
  - Inputs: three topologies with cherries
  - Expect: each built tree contains expected cherry pairs

- **test_branch_sampling_uses_unrooted_count**
  - Inputs: rooted 3-taxa
  - Expect: _sample_branch_length called infer_branch_output_count(rooted=False) times

- **test_root_split_preserves_total_length**
  - Inputs: split_root_branch true, deterministic samples
  - Expect: left+right branch lengths equal sampled connector length, total branch count correct

- **test_rooted_no_split_draws_independent_edges**
  - Inputs: split_root_branch false, deterministic samples
  - Expect: branch lengths equal the provided samples

- **test_unrooted_two_taxa_assigns_single_branch**
  - Inputs: unrooted 2 taxa
  - Expect: exactly one branch length assigned, other child branch_length None

- **test_three_taxa_tree_respects_topology**
  - Inputs: rooted 3 taxa with topology ((sp1,sp2),:sp3)
  - Expect: internal clade {sp1,sp2} present, lengths within range, rooted true

- **test_four_taxa_tree_supports_double_cherries**
  - Inputs: unrooted 4 taxa with two cherries
  - Expect: both cherries present, lengths within range, rooted false

- **test_phylogeny_stores_newick_metadata**
  - Inputs: generation_config
  - Expect: topology and newick stored in phylogeny.other, newick ends with ';'

- **test_topology_validation_requires_all_taxa**
  - Inputs: topology missing taxa
  - Expect: ConfigurationError

- **test_custom_xml_directory**
  - Inputs: custom xml_directory
  - Expect: xml_path uses custom directory

- **test_custom_npy_directory**
  - Inputs: custom npy_directory
  - Expect: output_npy_path uses custom directory

- **test_both_custom_directories**
  - Inputs: custom xml_directory and npy_directory
  - Expect: both paths use custom directories

- **test_default_directories_when_not_specified**
  - Inputs: no custom directories
  - Expect: xml_data and npy_data under base path

- **test_tree_chunk_size_must_be_positive**
  - Inputs: tree_chunk_size 0
  - Expect: ConfigurationError

- **test_empty_xml_directory_raises_error**
  - Inputs: xml_directory ""
  - Expect: ConfigurationError

- **test_empty_npy_directory_raises_error**
  - Inputs: npy_directory ""
  - Expect: ConfigurationError

## test_xml_parser.py

- **test_parse_examples_extracts_clades**
  - Inputs: phyloxml_file fixture
  - Expect: 1 example, clade names [A, B], equal sequence lengths, topology metadata present, branches contain A and B splits

- **test_build_dataset_creates_structured_npy**
  - Inputs: TreeExample with two clades, topology metadata
  - Expect: structured dtype, y_br order [A, B], branch_mask all true, y_top one-hot, X shape (2, length, 4)

- **test_one_hot_encode_validates_length**
  - Inputs: sequence shorter than config length
  - Expect: ValueError

- **test_one_hot_encode_supports_gap_when_enabled**
  - Inputs: "AT-C" with include_gap true
  - Expect: shape (4, 5), gap channel index 4, ValueError when include_gap false

- **test_write_dataset_uses_gap_channel**
  - Inputs: indel-enabled config, sequences length 4 with '-' gaps
  - Expect: X shape (2, 4, 5), gap channel set for '-' positions

- **test_write_dataset_pads_shorter_sequences**
  - Inputs: indel-enabled config with sequence length 5, two trees with equal lengths per tree (length 2 and length 3)
  - Expect: X shape (2, 5, 5), rows past each tree's length are all zeros

- **test_branch_mapping_three_taxa**
  - Inputs: rooted 3 taxa branches mapping
  - Expect: y_br shape 6, mask [True, True, True, False, True, False]

- **test_branch_mapping_unrooted_two_taxa**
  - Inputs: unrooted 2 taxa
  - Expect: y_br shape 1, mask [True], values match

- **test_branch_mapping_unrooted_three_taxa**
  - Inputs: unrooted 3 taxa
  - Expect: y_br shape 3, mask all true, values match

- **test_branch_mapping_unrooted_four_taxa**
  - Inputs: unrooted 4 taxa
  - Expect: y_br shape 5, mask all true, values match

- **test_branch_mapping_four_taxa**
  - Inputs: rooted 4 taxa with internal splits
  - Expect: y_br shape 10, mask positions set per expected ordering, values match

- **test_xml_parser_supports_multi_taxa_shapes**
  - Inputs: param sets for 3 and 4 taxa, monkeypatched IQ-TREE simulator
  - Expect: dataset X shape matches taxa count and length, y_br shape valid, y_top present
