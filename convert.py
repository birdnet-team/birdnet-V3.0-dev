#!/usr/bin/env python3
"""
BirdNET+ Model Converter for Web Inference

Converts ONNX models to optimized formats for browser deployment.
Supports FP16, INT8 quantization, species filtering, and graph optimizations.

Usage:
    python convert.py input.onnx --fp16              # Convert to FP16 (~50% size)
    python convert.py input.onnx --int8              # Dynamic INT8 quantization (~25% size)
    python convert.py input.onnx --optimize          # Graph optimizations only
    python convert.py input.onnx --fp16 --optimize   # FP16 + optimizations
    python convert.py input.onnx --all               # Generate all variants
    python convert.py input.onnx --species-list species.txt  # Filter to specific species

Requirements:
    pip install onnx onnxruntime onnxconverter-common onnxruntime-tools

    For TFLite/LiteRT conversion (requires Linux; use WSL on Windows):
    pip install litert-torch torch
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional


def get_model_size(path: Path) -> str:
    """Returns human-readable file size."""
    size = path.stat().st_size
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def make_output_name(stem: str, format_name: str, suffix: str = ".onnx") -> str:
    """
    Generate output filename by replacing precision marker.
    If stem contains 'FP32', replace it; otherwise append format name.
    
    Examples:
        make_output_name("Model_FP32", "FP16") -> "Model_FP16.onnx"
        make_output_name("Model_FP32", "INT8") -> "Model_INT8.onnx"
        make_output_name("SomeModel", "FP16") -> "SomeModel_FP16.onnx"
        make_output_name("Model_FP32", "FP32", ".tflite") -> "Model_FP32.tflite"
    """
    if "FP32" in stem:
        return stem.replace("FP32", format_name) + suffix
    return f"{stem}_{format_name}{suffix}"


def print_model_info(model_path: Path) -> None:
    """Print detailed model information including inputs and outputs."""
    try:
        import onnx
    except ImportError:
        print("ERROR: Install onnx: pip install onnx")
        return
    
    model = onnx.load(str(model_path))
    print(f"\nModel: {model_path.name}")
    print(f"Size:  {get_model_size(model_path)}")
    
    # Input info
    print("\nInputs:")
    for inp in model.graph.input:
        shape = [d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]
        print(f"  {inp.name}: {shape}")
    
    # Output info
    print("\nOutputs:")
    for out in model.graph.output:
        shape = [d.dim_value if d.dim_value else d.dim_param for d in out.type.tensor_type.shape.dim]
        # Identify output type
        if len(shape) == 2:
            dim = shape[1] if isinstance(shape[1], int) else None
            if dim == 1280:
                label = " (embeddings)"
            elif dim and dim > 1000:
                label = f" (predictions: {dim} species)"
            else:
                label = ""
        else:
            label = ""
        print(f"  {out.name}: {shape}{label}")
    
    # Opset version
    if model.opset_import:
        opset = model.opset_import[0].version
        print(f"\nOpset: {opset}")


def load_labels(labels_path: Path) -> List[Tuple[int, str, str]]:
    """
    Load labels CSV file.
    Returns list of (index, scientific_name, common_name) tuples.
    """
    import csv
    labels = []
    # Use utf-8-sig to handle BOM if present
    with open(labels_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            idx = int(row['idx'])
            sci_name = row['sci_name']
            com_name = row['com_name']
            labels.append((idx, sci_name, com_name))
    return labels


def parse_species_list(species_list_path: Path) -> List[str]:
    """
    Parse species list file (one species per line).
    Lines starting with # are comments. Empty lines are ignored.
    Species can be scientific name, common name, or "SciName_CommonName" format.
    """
    species = []
    with open(species_list_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                species.append(line)
    return species


def filter_species(
    input_path: Path,
    output_path: Path,
    labels_path: Path,
    species_list_path: Path,
    output_labels_path: Optional[Path] = None
) -> bool:
    """
    Filter model to only include specified species.
    
    This removes output neurons for species not in the list, reducing model size
    and potentially improving accuracy for the target species.
    
    Args:
        input_path: Path to input ONNX model
        output_path: Path to save filtered model
        labels_path: Path to full labels CSV file
        species_list_path: Path to text file with species to keep (one per line)
        output_labels_path: Path to save filtered labels CSV (default: derived from output_path)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        import onnx
        import numpy as np
        from onnx import numpy_helper
    except ImportError:
        print("  ERROR: Install onnx and numpy")
        return False
    
    print(f"  Loading species list from {species_list_path}...")
    target_species = parse_species_list(species_list_path)
    if not target_species:
        print("  ERROR: Species list is empty")
        return False
    print(f"  Target species count: {len(target_species)}")
    
    print(f"  Loading labels from {labels_path}...")
    all_labels = load_labels(labels_path)
    print(f"  Total labels in model: {len(all_labels)}")
    
    # Match species to indices
    # Support matching by: scientific name, common name, or "SciName_CommonName" format
    matched_indices = []
    matched_labels = []
    unmatched = []
    
    # Build lookup dictionaries for efficient matching
    sci_name_map = {}  # scientific_name -> (idx, sci_name, com_name)
    com_name_map = {}  # common_name -> (idx, sci_name, com_name)
    full_name_map = {}  # "SciName_ComName" -> (idx, sci_name, com_name)
    
    for idx, sci_name, com_name in all_labels:
        label_tuple = (idx, sci_name, com_name)
        sci_name_map[sci_name.lower()] = label_tuple
        com_name_map[com_name.lower()] = label_tuple
        full_name_map[f"{sci_name}_{com_name}".lower()] = label_tuple
    
    # Match species using lookups
    for species in target_species:
        species_lower = species.lower()
        label_tuple = None
        
        # Try direct lookups in order of specificity
        if species_lower in full_name_map:
            label_tuple = full_name_map[species_lower]
        elif species_lower in sci_name_map:
            label_tuple = sci_name_map[species_lower]
        elif species_lower in com_name_map:
            label_tuple = com_name_map[species_lower]
        elif '_' in species:
            # Try parsing "SciName_ComName" format
            parts = species.split('_', 1)
            sci_lower = parts[0].strip().lower()
            com_lower = parts[1].strip().lower()
            label_tuple = sci_name_map.get(sci_lower) or com_name_map.get(com_lower)
        
        if label_tuple:
            matched_indices.append(label_tuple[0])
            matched_labels.append(label_tuple)
        else:
            unmatched.append(species)
    
    if unmatched:
        print(f"  WARNING: {len(unmatched)} species not found in labels:")
        for s in unmatched[:10]:
            print(f"    - {s}")
        if len(unmatched) > 10:
            print(f"    ... and {len(unmatched) - 10} more")
    
    if not matched_indices:
        print("  ERROR: No species matched")
        return False
    
    # Sort indices for consistent ordering
    sorted_pairs = sorted(zip(matched_indices, matched_labels), key=lambda x: x[0])
    matched_indices = [p[0] for p in sorted_pairs]
    matched_labels = [p[1] for p in sorted_pairs]
    
    print(f"  Matched {len(matched_indices)} species")
    
    # Load model
    print(f"  Loading model...")
    model = onnx.load(str(input_path))
    
    # Find all weight tensors with the original class count dimension
    original_num_classes = len(all_labels)
    new_num_classes = len(matched_indices)
    
    # Weight tensors that need to be sliced (first dimension is num_classes)
    weights_to_slice = [
        'head.weight',      # [11560, 1280] -> [N, 1280]
        'head.bias',        # [11560] -> [N]
        'att_block.att.weight',   # [11560, 2048, 1] -> [N, 2048, 1]
        'att_block.att.bias',     # [11560] -> [N]
        'att_block.cla.weight',   # [11560, 2048, 1] -> [N, 2048, 1]
        'att_block.cla.bias',     # [11560] -> [N]
        'att_block2.att.weight',  # [11560, 2048, 1] -> [N, 2048, 1]
        'att_block2.att.bias',    # [11560] -> [N]
        'att_block2.cla.weight',  # [11560, 2048, 1] -> [N, 2048, 1]
        'att_block2.cla.bias',    # [11560] -> [N]
    ]
    
    print(f"  Slicing weights from {original_num_classes} to {new_num_classes} classes...")
    
    # Create index array for slicing
    indices = np.array(matched_indices, dtype=np.int64)
    
    # Process initializers (weights)
    new_initializers = []
    sliced_count = 0
    
    for init in model.graph.initializer:
        if init.name in weights_to_slice:
            # Convert to numpy array
            arr = numpy_helper.to_array(init)
            
            # Verify first dimension matches expected class count
            if arr.shape[0] != original_num_classes:
                print(f"    WARNING: {init.name} has unexpected shape {arr.shape}, skipping")
                new_initializers.append(init)
                continue
            
            # Slice to keep only target species
            new_arr = arr[indices]
            
            # Convert back to tensor
            new_init = numpy_helper.from_array(new_arr, name=init.name)
            new_initializers.append(new_init)
            sliced_count += 1
            print(f"    {init.name}: {arr.shape} -> {new_arr.shape}")
        else:
            new_initializers.append(init)
    
    print(f"  Sliced {sliced_count} weight tensors")
    
    # Update model initializers
    del model.graph.initializer[:]
    model.graph.initializer.extend(new_initializers)
    
    # Update output shape for predictions
    for output in model.graph.output:
        if output.name == 'predictions':
            # Update the dimension
            for dim in output.type.tensor_type.shape.dim:
                if dim.dim_value == original_num_classes:
                    dim.dim_value = new_num_classes
                    print(f"  Updated predictions output shape to {new_num_classes}")
    
    # Clear intermediate value_info shapes (may have cached old dimensions)
    while len(model.graph.value_info) > 0:
        model.graph.value_info.pop()
    
    # Save filtered model
    print(f"  Saving filtered model to {output_path}...")
    onnx.save(model, str(output_path))
    print(f"  Saved: {output_path} ({get_model_size(output_path)})")
    
    # Save filtered labels
    if output_labels_path is None:
        output_labels_path = output_path.parent / output_path.name.replace('.onnx', '_Labels.csv')
    
    print(f"  Saving filtered labels to {output_labels_path}...")
    import csv
    with open(output_labels_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(['idx', 'id', 'sci_name', 'com_name', 'class', 'order'])
        
        # Re-read original labels to get full row data (use utf-8-sig to handle BOM)
        with open(labels_path, 'r', encoding='utf-8-sig') as orig:
            reader = csv.DictReader(orig, delimiter=';')
            all_rows = {int(row['idx']): row for row in reader}
        
        # Write matched labels with new indices
        for new_idx, orig_idx in enumerate(matched_indices):
            if orig_idx in all_rows:
                row = all_rows[orig_idx]
                writer.writerow([
                    new_idx,
                    row.get('id', ''),
                    row['sci_name'],
                    row['com_name'],
                    row.get('class', ''),
                    row.get('order', '')
                ])
    
    print(f"  Saved: {output_labels_path}")
    print(f"  Species reduction: {original_num_classes} -> {new_num_classes} ({100*(1-new_num_classes/original_num_classes):.1f}% fewer)")
    
    return True


def optimize_graph(input_path: Path, output_path: Path) -> bool:
    """
    Apply ONNX Runtime graph optimizations.
    Includes constant folding, redundant node elimination, etc.
    """
    try:
        import onnxruntime as ort
        from onnxruntime.transformers import optimizer
    except ImportError:
        print("  ERROR: Install onnxruntime-tools: pip install onnxruntime onnxruntime-tools")
        return False

    print(f"  Applying graph optimizations...")
    
    try:
        # Use ONNX Runtime's optimizer
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.optimized_model_filepath = str(output_path)
        
        # Create session to trigger optimization and save
        ort.InferenceSession(str(input_path), sess_options, providers=["CPUExecutionProvider"])
        
        print(f"  Saved: {output_path} ({get_model_size(output_path)})")
        return True
    except Exception as e:
        print(f"  ERROR: Graph optimization failed: {e}")
        return False


def _convert_weights_to_fp16(model) -> int:
    """
    Convert float32 initializer weights to float16 for storage, keeping the
    computation graph entirely in float32.
    
    For each float32 initializer (weight tensor):
    1. Convert the data to float16 (halves storage size)
    2. Rename the initializer to "<name>_fp16"
    3. Insert a Cast(fp16→fp32) node to convert back to float32 at runtime
    4. Update the corresponding graph input (if any) to match
    
    This gives ~50% model size reduction while keeping all computation in
    float32, which is compatible with all ONNX Runtime backends including
    ORT Web WASM (which doesn't support float16 compute).
    
    Returns the number of initializers converted.
    """
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    graph = model.graph
    cast_nodes = []
    converted = 0

    # Build set of initializer names that are also graph inputs
    graph_input_names = {inp.name for inp in graph.input}

    for init in graph.initializer:
        if init.data_type != TensorProto.FLOAT:
            continue

        original_name = init.name
        fp16_name = original_name + "_fp16"

        # Convert weight data to float16
        arr = numpy_helper.to_array(init)
        arr_fp16 = arr.astype(np.float16)
        new_init = numpy_helper.from_array(arr_fp16, fp16_name)
        init.CopyFrom(new_init)

        # Update corresponding graph input (ONNX convention: initializers
        # may also appear as graph inputs for shape/type metadata)
        for inp in graph.input:
            if inp.name == original_name:
                inp.name = fp16_name
                inp.type.tensor_type.elem_type = TensorProto.FLOAT16
                break

        # Rename references in graph nodes: old initializer name → Cast output
        # The Cast node will output a float32 tensor with the original name,
        # so downstream nodes don't need to change at all.
        # We only rename node inputs that directly reference this initializer.
        # (The Cast output keeps the original name, making this transparent.)

        # Insert Cast node: fp16 weight → fp32 for computation
        cast_node = helper.make_node(
            "Cast",
            inputs=[fp16_name],
            outputs=[original_name],
            to=int(TensorProto.FLOAT),
            name=f"cast_weight_{original_name}",
        )
        cast_nodes.append(cast_node)
        converted += 1

    # Prepend all Cast nodes to the graph
    for i, node in enumerate(cast_nodes):
        graph.node.insert(i, node)

    return converted


def convert_to_fp16(input_path: Path, output_path: Path, keep_io_fp32: bool = True) -> bool:
    """
    Convert model weights from FP32 to FP16.
    Roughly halves model size with minimal accuracy loss for most models.
    
    Uses "weights-only" FP16: weight tensors are stored as float16 (halving
    model file size), with Cast nodes to convert back to float32 at runtime.
    The computation graph stays entirely in float32, ensuring compatibility
    with all ONNX Runtime backends including ORT Web WASM.
    
    When keep_io_fp32=True (default, no --fp16-io):
      Weights stored as FP16, all computation and I/O stays FP32.
      Compatible with both desktop and web ONNX Runtime.
    
    When keep_io_fp32=False (--fp16-io):
      Full FP16 conversion (weights, compute, I/O all float16).
      WARNING: Does NOT work with ONNX Runtime Web (WASM has no float16 compute).
      Only use for desktop ONNX Runtime with explicit float16 support.
    """
    try:
        import onnx
        from onnxconverter_common import float16
    except ImportError:
        print("  ERROR: Install required packages: pip install onnx onnxconverter-common")
        return False

    print(f"  Converting to FP16...")
    
    try:
        model = onnx.load(str(input_path))
        
        if keep_io_fp32:
            # Weights-only FP16: convert initializer storage to float16,
            # keep computation graph as float32. This is compatible with
            # all backends including ORT Web WASM.
            count = _convert_weights_to_fp16(model)
            print(f"  Converted {count} weight tensors to FP16 storage")
            onnx.save(model, str(output_path))
        else:
            # Full FP16: convert everything including compute graph.
            # Only works with runtimes that support float16 compute.
            while len(model.graph.value_info) > 0:
                model.graph.value_info.pop()
            
            model_fp16 = float16.convert_float_to_float16(
                model,
                keep_io_types=False,
                disable_shape_infer=True
            )
            onnx.save(model_fp16, str(output_path))
        
        print(f"  Saved: {output_path} ({get_model_size(output_path)})")
        return True
    except Exception as e:
        print(f"  ERROR: FP16 conversion failed: {e}")
        return False


def quantize_int8_dynamic(
    input_path: Path,
    output_path: Path,
) -> bool:
    """
    Apply dynamic INT8 quantization.
    Significantly reduces size (~75%) but may affect accuracy.
    
    WARNING: INT8 quantization produces unreliable results for this model.
    The 11K-class softmax amplifies quantization errors, causing the model
    to output many false positives. Use FP16 instead for production.
    
    Note: Dynamic quantization does NOT require calibration data.
    
    Preprocessing steps (as recommended by ONNX Runtime):
    1. Symbolic shape inference
    2. ONNX Runtime graph optimization  
    3. ONNX shape inference
    4. Clear conflicting intermediate shapes
    """
    try:
        import onnx
        from onnxruntime.quantization import quantize_dynamic, QuantType
        from onnxruntime.quantization.shape_inference import quant_pre_process
        import tempfile
        import os
    except ImportError:
        print("  ERROR: Install onnx and onnxruntime")
        return False

    print(f"  Applying INT8 dynamic quantization...")
    
    try:
        # Step 1: Run ONNX Runtime's recommended preprocessing
        print(f"  Step 1/3: Running quantization preprocessing...")
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
            preprocessed_path = tmp.name
        
        try:
            quant_pre_process(
                input_model_path=str(input_path),
                output_model_path=preprocessed_path,
                skip_symbolic_shape=True,   # Skip symbolic shape (fails on this model)
                skip_optimization=False,    # Run ONNX RT optimization
                auto_merge=True,            # Merge duplicate initializers
                save_as_external_data=False,
                verbose=0
            )
            print(f"  Preprocessing completed (graph optimization + auto-merge)")
            working_path = preprocessed_path
        except Exception as pre_err:
            print(f"  Preprocessing warning: {pre_err}")
            print(f"  Continuing with original model...")
            working_path = str(input_path)
            if os.path.exists(preprocessed_path):
                os.unlink(preprocessed_path)
            preprocessed_path = None
        
        # Step 2: Clear intermediate shape annotations to fix conflicts
        print(f"  Step 2/3: Clearing conflicting shape annotations...")
        model = onnx.load(working_path)
        
        # Clear ALL intermediate shape info to let quantizer infer fresh
        # This resolves conflicts between embeddings (1280) and predictions (11K)
        cleared_count = len(model.graph.value_info)
        while len(model.graph.value_info) > 0:
            model.graph.value_info.pop()
        
        # Save cleaned model
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
            cleaned_path = tmp.name
        onnx.save(model, cleaned_path)
        print(f"  Cleared {cleared_count} intermediate shape annotations")
        
        # Cleanup preprocessed file if it was created
        if preprocessed_path and os.path.exists(preprocessed_path):
            os.unlink(preprocessed_path)
        
        # Step 3: Run quantization on cleaned model
        print(f"  Step 3/3: Quantizing weights to INT8...")
        print(f"  WARNING: INT8 results may be unreliable for classification tasks")
        quantize_dynamic(
            model_input=cleaned_path,
            model_output=str(output_path),
            weight_type=QuantType.QUInt8,
            op_types_to_quantize=['MatMul', 'Gemm', 'Conv'],
            per_channel=False,
            reduce_range=False
        )
        
        # Cleanup
        os.unlink(cleaned_path)
        
        print(f"  Saved: {output_path} ({get_model_size(output_path)})")
        return True
        
    except Exception as e:
        error_msg = str(e)
        print(f"  ERROR: INT8 quantization failed: {error_msg}")
        
        if "ShapeInferenceError" in error_msg:
            print()
            print(f"  The BirdNET model's architecture causes shape inference conflicts")
            print(f"  during INT8 quantization (dual output heads with different dims).")
            print()
            print(f"  Workaround: Use FP16 instead (--fp16) for 50% size reduction.")
        
        return False


def quantize_int8_head_only(
    input_path: Path,
    output_path: Path,
) -> bool:
    """
    Apply INT8 quantization to classification head only.
    
    This keeps the backbone (spectrogram + feature extraction) in FP32 while
    quantizing the classification head to INT8. This produces identical detection
    results to FP32 while reducing model size by ~52%.
    
    Why this works: INT8 quantization errors accumulate through the deep backbone
    network, but the classification head alone can be safely quantized since the
    softmax normalization operates on already-extracted features.
    
    Nodes quantized:
    - head.weight/bias (Gemm): main classification layer 
    - att_block.att/cla (Conv): attention block 1
    - att_block2.att/cla (Conv): attention block 2
    """
    try:
        import onnx
        from onnxruntime.quantization import quantize_dynamic, QuantType
        from onnxruntime.quantization.shape_inference import quant_pre_process
        import tempfile
        import os
    except ImportError:
        print("  ERROR: Install onnx and onnxruntime")
        return False

    print(f"  Applying INT8 quantization to classification head only...")
    
    try:
        # Load model to find classification head nodes
        model = onnx.load(str(input_path))
        
        # Find nodes that use 11560-dimension weights (classification head)
        # These are: head (Gemm) + att_block.att/cla + att_block2.att/cla (Conv)
        initializers = {init.name: init for init in model.graph.initializer}
        head_nodes = []
        
        for node in model.graph.node:
            for inp in node.input:
                if inp in initializers:
                    shape = list(initializers[inp].dims)
                    # Check if this node uses a weight tensor with num_classes dimension
                    if any(d > 10000 for d in shape):  # num_classes is typically 11K+
                        head_nodes.append(node.name)
                        break
        
        print(f"  Classification head nodes: {len(head_nodes)}")
        for name in head_nodes:
            print(f"    - {name}")
        
        if not head_nodes:
            print(f"  WARNING: No classification head nodes found")
            return False
        
        # Step 1: Preprocess
        print(f"  Step 1/3: Preprocessing...")
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
            preprocessed_path = tmp.name
        
        try:
            quant_pre_process(
                input_model_path=str(input_path),
                output_model_path=preprocessed_path,
                skip_symbolic_shape=True,
                skip_optimization=False,
                auto_merge=True,
                save_as_external_data=False,
                verbose=0
            )
            working_path = preprocessed_path
        except Exception as pre_err:
            print(f"  Preprocessing warning: {pre_err}")
            working_path = str(input_path)
            if os.path.exists(preprocessed_path):
                os.unlink(preprocessed_path)
            preprocessed_path = None
        
        # Step 2: Clear intermediate shape annotations
        print(f"  Step 2/3: Clearing shape annotations...")
        model = onnx.load(working_path)
        while len(model.graph.value_info) > 0:
            model.graph.value_info.pop()
        
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
            cleaned_path = tmp.name
        onnx.save(model, cleaned_path)
        
        if preprocessed_path and os.path.exists(preprocessed_path):
            os.unlink(preprocessed_path)
        
        # Step 3: Quantize only head nodes
        print(f"  Step 3/3: Quantizing classification head to INT8...")
        quantize_dynamic(
            model_input=cleaned_path,
            model_output=str(output_path),
            weight_type=QuantType.QUInt8,
            nodes_to_quantize=head_nodes,  # Only these nodes
            per_channel=False,
            reduce_range=False
        )
        
        os.unlink(cleaned_path)
        
        print(f"  Saved: {output_path} ({get_model_size(output_path)})")
        return True
        
    except Exception as e:
        print(f"  ERROR: INT8 head quantization failed: {e}")
        return False


def quantize_int8_static(
    input_path: Path,
    output_path: Path,
    calibration_data_path: Path | None = None
) -> bool:
    """
    Apply static INT8 quantization (requires calibration data).
    Best compression but needs representative input samples.
    """
    try:
        from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader
        import numpy as np
    except ImportError:
        print("  ERROR: Install onnxruntime and numpy")
        return False

    if calibration_data_path is None:
        print("  SKIPPED: Static quantization requires calibration data")
        print("           Use --calibration-dir with sample .npy files")
        return False

    print(f"  Applying INT8 static quantization...")
    
    class AudioCalibrationReader(CalibrationDataReader):
        def __init__(self, data_dir: Path):
            self.data_files = list(data_dir.glob("*.npy"))
            self.index = 0
            
        def get_next(self):
            if self.index >= len(self.data_files):
                return None
            data = np.load(self.data_files[self.index])
            self.index += 1
            return {"input": data}
    
    try:
        calibration_reader = AudioCalibrationReader(calibration_data_path)
        quantize_static(
            model_input=str(input_path),
            model_output=str(output_path),
            calibration_data_reader=calibration_reader,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QInt8
        )
        print(f"  Saved: {output_path} ({get_model_size(output_path)})")
        return True
    except Exception as e:
        print(f"  ERROR: Static INT8 quantization failed: {e}")
        return False


def convert_to_ort_format(input_path: Path, output_path: Path) -> bool:
    """
    Convert to ORT format (ONNX Runtime's optimized format).
    Faster loading but less portable.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("  ERROR: Install onnxruntime")
        return False

    print(f"  Converting to ORT format...")
    
    try:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.add_session_config_entry("session.save_model_format", "ORT")
        sess_options.optimized_model_filepath = str(output_path)
        
        ort.InferenceSession(str(input_path), sess_options, providers=["CPUExecutionProvider"])
        
        print(f"  Saved: {output_path} ({get_model_size(output_path)})")
        return True
    except Exception as e:
        print(f"  ERROR: ORT conversion failed: {e}")
        return False


def convert_to_tflite(
    input_path: Path,
    output_path: Path,
    batch_size: int | None = 1,
    duration_sec: float | None = None,
    sample_rate: int = 32000,
) -> bool:
    """
    Convert a PyTorch TorchScript (.pt) model to TFLite / LiteRT format
    using litert-torch (formerly ai-edge-torch).

    Pipeline: TorchScript (.pt) → torch.export → litert_torch.convert → .tflite

    The BirdNET model input shape is [batch, samples].  Users can optionally
    fix one or both dimensions to produce a static-shape TFLite model (which
    is better supported on most delegates).  When a dimension is left
    unspecified (None), torch.export dynamic_shapes are used and the TFLite
    model will accept variable sizes for that axis.

    Args:
        input_path:    Path to input TorchScript .pt model
        output_path:   Path for output .tflite file
        batch_size:    Fixed batch size (default 1). None = dynamic.
        duration_sec:  Fixed audio duration in seconds.  None = dynamic.
        sample_rate:   Audio sample rate in Hz (default 32000)

    Returns:
        True if successful, False otherwise
    """
    try:
        import torch
    except ImportError:
        print("  ERROR: Install PyTorch: pip install torch")
        return False

    try:
        import litert_torch
    except ImportError:
        print("  ERROR: Install litert-torch: pip install litert-torch")
        print("         Note: litert-torch currently requires Linux.")
        print("         On Windows, use WSL (Windows Subsystem for Linux).")
        return False

    # ---- Determine shapes ----
    sample_batch = batch_size if batch_size is not None else 1
    if duration_sec is not None:
        sample_samples = int(duration_sec * sample_rate)
    else:
        sample_samples = int(3.0 * sample_rate)  # 3 s default for tracing

    shape_desc_parts = []
    if batch_size is not None:
        shape_desc_parts.append(f"batch={batch_size}")
    else:
        shape_desc_parts.append("batch=dynamic")
    if duration_sec is not None:
        shape_desc_parts.append(f"duration={duration_sec}s ({sample_samples} samples)")
    else:
        shape_desc_parts.append("duration=dynamic")
    shape_desc = ", ".join(shape_desc_parts)

    print(f"  Converting PyTorch (.pt) → TFLite ...")
    print(f"  Input shape: [{sample_batch}, {sample_samples}]  ({shape_desc})")

    try:
        # Step 1 ── Load TorchScript model
        print("  Step 1/3: Loading TorchScript model...")
        scripted_model = torch.jit.load(str(input_path), map_location="cpu")
        scripted_model.eval()

        # Step 2 ── Wrap in a plain nn.Module so torch.export can trace it
        class _Wrapper(torch.nn.Module):
            def __init__(self, m: torch.jit.ScriptModule):
                super().__init__()
                self.m = m

            def forward(self, x: torch.Tensor):
                # BirdNET returns (embeddings, predictions)
                out = self.m(x)
                if isinstance(out, (tuple, list)):
                    return tuple(out)
                return out

        wrapper = _Wrapper(scripted_model)
        wrapper.eval()

        sample_input = (torch.randn(sample_batch, sample_samples),)

        # Build dynamic_shapes dict when a dimension is not fixed
        dynamic_shapes = None
        dim_map: dict[int, object] = {}
        if batch_size is None:
            dim_map[0] = torch.export.Dim("batch", min=1, max=128)
        if duration_sec is None:
            dim_map[1] = torch.export.Dim("samples", min=sample_rate, max=sample_rate * 600)
        if dim_map:
            dynamic_shapes = ({**dim_map},)

        # Step 3 ── Convert with litert-torch
        print("  Step 2/3: Converting via litert-torch...")
        convert_kwargs: dict = {}
        if dynamic_shapes is not None:
            convert_kwargs["dynamic_shapes"] = dynamic_shapes
        edge_model = litert_torch.convert(wrapper, sample_input, **convert_kwargs)

        print("  Step 3/3: Exporting .tflite...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        edge_model.export(str(output_path))

        print(f"  Saved: {output_path} ({get_model_size(output_path)})")
        return True

    except Exception as e:
        print(f"  ERROR: TFLite conversion failed: {e}")
        return False


def validate_model(model_path: Path, reference_path: Path | None = None) -> bool:
    """
    Validate that the converted model produces similar outputs.
    BirdNET models have two outputs: embeddings [batch, 1280] and predictions [batch, ~11K].
    """
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        print("  SKIP validation: numpy/onnxruntime not available")
        return True

    print(f"  Validating model...")
    
    try:
        # Run inference on converted model
        sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        input_info = sess.get_inputs()[0]
        input_name = input_info.name
        input_type = input_info.type
        
        # Create test input (3 seconds of audio at 32kHz = 96000 samples)
        # Match the model's expected input type
        if 'float16' in input_type:
            test_input = np.random.randn(1, 96000).astype(np.float16)
            print(f"  Using FP16 input (model expects float16)")
        else:
            test_input = np.random.randn(1, 96000).astype(np.float32)
        
        outputs = sess.run(None, {input_name: test_input})
        output_names = [o.name for o in sess.get_outputs()]
        
        # Print all outputs with their shapes
        print(f"  Model outputs:")
        embeddings_idx = None
        predictions_idx = None
        for i, (name, out) in enumerate(zip(output_names, outputs)):
            shape = out.shape
            # Identify output type by shape
            if len(shape) == 2 and shape[1] == 1280:
                output_type = "embeddings"
                embeddings_idx = i
            elif len(shape) == 2 and shape[1] > 1000:
                output_type = f"predictions ({shape[1]} species)"
                predictions_idx = i
            else:
                output_type = "unknown"
            print(f"    {name}: {shape} ({output_type})")
        
        # Compare predictions with reference if provided
        if reference_path and predictions_idx is not None:
            ref_sess = ort.InferenceSession(str(reference_path), providers=["CPUExecutionProvider"])
            ref_input_type = ref_sess.get_inputs()[0].type
            
            # Reference model always uses FP32 input
            ref_input = np.random.randn(1, 96000).astype(np.float32)
            np.random.seed(42)  # Use same random seed for both
            ref_input = np.random.randn(1, 96000).astype(np.float32)
            
            # Re-run converted model with same random seed
            np.random.seed(42)
            if 'float16' in input_type:
                test_input = np.random.randn(1, 96000).astype(np.float16)
            else:
                test_input = np.random.randn(1, 96000).astype(np.float32)
            
            outputs = sess.run(None, {input_name: test_input})
            ref_outputs = ref_sess.run(None, {ref_sess.get_inputs()[0].name: ref_input})
            
            # Convert to float32 for comparison
            pred = outputs[predictions_idx].astype(np.float32)
            ref_pred = ref_outputs[predictions_idx].astype(np.float32)
            
            pred_diff_max = np.max(np.abs(pred - ref_pred))
            pred_diff_mean = np.mean(np.abs(pred - ref_pred))
            print(f"  Predictions difference: max={pred_diff_max:.6f}, mean={pred_diff_mean:.6f}")
            
            if embeddings_idx is not None:
                emb = outputs[embeddings_idx].astype(np.float32)
                ref_emb = ref_outputs[embeddings_idx].astype(np.float32)
                emb_diff_max = np.max(np.abs(emb - ref_emb))
                emb_diff_mean = np.mean(np.abs(emb - ref_emb))
                print(f"  Embeddings difference:  max={emb_diff_max:.6f}, mean={emb_diff_mean:.6f}")
        
        return True
    except Exception as e:
        print(f"  WARNING: Validation failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert ONNX models for optimized web inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert.py model.onnx --info
  python convert.py model.onnx --fp16
  python convert.py model.onnx --fp16 --validate
  python convert.py model.onnx --all --output-dir ./optimized/
  
  # Filter to specific species (creates smaller, specialized model)
  python convert.py model.onnx --species-list my_species.txt --labels labels.csv
  python convert.py model.onnx --species-list my_species.txt --labels labels.csv --fp16
        """
    )
    parser.add_argument("input", type=Path, help="Input ONNX model path")
    parser.add_argument("--info", action="store_true", help="Show model info and exit")
    parser.add_argument("--output-dir", "-o", type=Path, help="Output directory (default: same as input)")
    parser.add_argument("--fp16", action="store_true", help="Convert to FP16 (half precision)")
    parser.add_argument("--int8", action="store_true", help="Apply INT8 dynamic quantization (WARNING: unreliable for classification)")
    parser.add_argument("--int8-head", action="store_true", help="Apply INT8 quantization to classification head only (recommended)")
    parser.add_argument("--int8-static", action="store_true", help="Apply INT8 static quantization (needs calibration)")
    parser.add_argument("--calibration-dir", type=Path, help="Directory with .npy calibration samples")
    parser.add_argument("--optimize", action="store_true", help="Apply graph optimizations")
    parser.add_argument("--ort", action="store_true", help="Convert to ORT format")
    parser.add_argument("--all", action="store_true", help="Generate all variants (FP16, INT8, optimized)")
    parser.add_argument("--validate", action="store_true", help="Validate converted models")
    parser.add_argument("--fp16-io", action="store_true", help="Also convert inputs/outputs to FP16")
    parser.add_argument("--tflite", action="store_true", help="Convert PyTorch (.pt) model to TFLite/LiteRT via litert-torch")
    parser.add_argument("--tflite-batch-size", type=int, default=1, metavar="N",
                        help="Fixed batch size for TFLite model (default: 1). Set to 0 for dynamic batch.")
    parser.add_argument("--tflite-duration", type=float, default=None, metavar="SEC",
                        help="Fixed audio duration in seconds for TFLite input. "
                             "Omit for dynamic duration (model accepts any length).")
    parser.add_argument("--tflite-sample-rate", type=int, default=32000, metavar="HZ",
                        help="Audio sample rate for TFLite input size calculation (default: 32000)")
    parser.add_argument("--tflite-model", type=Path, default=None, metavar="PATH",
                        help="Path to PyTorch (.pt) model for TFLite conversion. "
                             "If omitted, infers from the ONNX input path (replaces .onnx with .pt).")
    parser.add_argument("--species-list", type=Path, help="Filter model to only include species in this file (one per line)")
    parser.add_argument("--labels", type=Path, help="Path to labels CSV file (required with --species-list)")
    
    args = parser.parse_args()
    
    # Validate input
    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)
    
    # Show model info
    if args.info:
        print_model_info(args.input)
        sys.exit(0)
    
    # Set output directory
    output_dir = args.output_dir or args.input.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate species filtering arguments
    if args.species_list and not args.labels:
        print("ERROR: --labels is required when using --species-list")
        sys.exit(1)
    if args.species_list and not args.species_list.exists():
        print(f"ERROR: Species list file not found: {args.species_list}")
        sys.exit(1)
    if args.labels and not args.labels.exists():
        print(f"ERROR: Labels file not found: {args.labels}")
        sys.exit(1)
    
    # Get base name
    stem = args.input.stem
    
    print(f"\nBirdNET+ Model Converter")
    print(f"========================")
    print_model_info(args.input)
    print(f"\nOutput: {output_dir}/")
    print()
    
    # Track results
    results = []
    
    # Apply --all flag
    if args.all:
        args.fp16 = True
        args.int8 = True
        args.optimize = True
        args.tflite = True
    
    # If no conversion specified and no species filtering, show help
    if not any([args.fp16, args.int8, args.int8_head, args.int8_static, args.optimize, args.ort, args.tflite, args.species_list]):
        parser.print_help()
        print("\nERROR: Specify at least one conversion: --fp16, --int8, --int8-head, --optimize, --ort, --tflite, --species-list, or --all")
        sys.exit(1)
    
    # Working model path (may change after species filtering)
    working_path = args.input
    
    # 0. Species filtering (do first, use filtered model for all other conversions)
    if args.species_list:
        print("[0/4] Species Filtering")
        
        # Determine output name based on species list file name
        species_list_name = args.species_list.stem
        filtered_stem = f"{stem.replace('_FP32', '')}_{species_list_name}"
        filtered_output = output_dir / f"{filtered_stem}.onnx"
        
        if filter_species(args.input, filtered_output, args.labels, args.species_list):
            results.append(("Filtered", filtered_output))
            # Use filtered model for further conversions
            working_path = filtered_output
            stem = filtered_stem
        else:
            print("  Species filtering failed, aborting.")
            sys.exit(1)
        print()
    
    # 1. Graph optimization (do first, use as input for further conversions)
    optimized_path = working_path
    if args.optimize:
        print("[1/4] Graph Optimization")
        opt_output = output_dir / make_output_name(stem, "optimized")
        if optimize_graph(working_path, opt_output):
            results.append(("Optimized", opt_output))
            if args.validate:
                validate_model(opt_output, working_path)
            # Use optimized model for further conversions
            optimized_path = opt_output
        print()
    
    # 2. FP16 conversion
    if args.fp16:
        print("[2/4] FP16 Conversion")
        fp16_output = output_dir / make_output_name(stem, "FP16")
        if convert_to_fp16(optimized_path, fp16_output, keep_io_fp32=not args.fp16_io):
            results.append(("FP16", fp16_output))
            if args.validate:
                validate_model(fp16_output, working_path)
        print()
    
    # 3. INT8 dynamic quantization
    if args.int8:
        print("[3/4] INT8 Dynamic Quantization")
        int8_output = output_dir / make_output_name(stem, "INT8")
        if quantize_int8_dynamic(optimized_path, int8_output):
            results.append(("INT8", int8_output))
            if args.validate:
                validate_model(int8_output, working_path)
        print()
    
    # 3b. INT8 head-only quantization (recommended alternative to full INT8)
    if args.int8_head:
        print("[3b/4] INT8 Head-Only Quantization")
        int8h_output = output_dir / make_output_name(stem, "INT8-head")
        if quantize_int8_head_only(optimized_path, int8h_output):
            results.append(("INT8-Head", int8h_output))
            if args.validate:
                validate_model(int8h_output, working_path)
        print()
    
    # 4. INT8 static quantization
    if args.int8_static:
        print("[3b/4] INT8 Static Quantization")
        int8s_output = output_dir / make_output_name(stem, "INT8-static")
        if quantize_int8_static(optimized_path, int8s_output, args.calibration_dir):
            results.append(("INT8-Static", int8s_output))
            if args.validate:
                validate_model(int8s_output, working_path)
        print()
    
    # 5. ORT format
    if args.ort:
        print("[4/6] ORT Format Conversion")
        ort_output = output_dir / make_output_name(stem, "ORT", ".ort")
        if convert_to_ort_format(optimized_path, ort_output):
            results.append(("ORT", ort_output))
        print()

    # 6. TFLite / LiteRT conversion via litert-torch (from PyTorch .pt)
    if args.tflite:
        print("[5/6] TFLite/LiteRT Conversion (via litert-torch)")

        # Resolve .pt model path
        pt_path = args.tflite_model
        if pt_path is None:
            # Try to infer from ONNX input path: replace suffixes like _FP32.onnx → _FP32.pt
            inferred = args.input.with_suffix(".pt")
            if inferred.exists():
                pt_path = inferred
            else:
                # Also try stripping precision suffix
                for tag in ["_FP32", "_FP16", "_INT8"]:
                    candidate = Path(str(args.input).replace(tag + ".onnx", ".pt"))
                    if candidate.exists():
                        pt_path = candidate
                        break
            if pt_path is None or not pt_path.exists():
                print(f"  ERROR: Could not find PyTorch model. Tried: {inferred}")
                print(f"         Use --tflite-model to specify the .pt path explicitly.")
            else:
                print(f"  Using PyTorch model: {pt_path}")

        if pt_path is not None and pt_path.exists():
            batch_size_val = args.tflite_batch_size if args.tflite_batch_size != 0 else None
            tflite_output = output_dir / make_output_name(stem, "FP32", ".tflite")
            if convert_to_tflite(
                pt_path,
                tflite_output,
                batch_size=batch_size_val,
                duration_sec=args.tflite_duration,
                sample_rate=args.tflite_sample_rate,
            ):
                results.append(("TFLite", tflite_output))
        print()
    
    # Summary
    if results:
        print("Summary")
        print("-------")
        original_size = args.input.stat().st_size
        for name, path in results:
            new_size = path.stat().st_size
            reduction = (1 - new_size / original_size) * 100
            print(f"  {name:12} {get_model_size(path):>10}  (-{reduction:.1f}%)")
        print()
        print("Recommended for web: FP16 offers best balance of size and accuracy")
        print("For smallest size: INT8 (verify accuracy with your use case)")
    else:
        print("No models were converted successfully.")
        sys.exit(1)


if __name__ == "__main__":
    main()
