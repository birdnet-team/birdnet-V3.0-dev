#!/usr/bin/env python3
"""
BirdNET+ Model Converter for Web Inference

Converts ONNX models to optimized formats for browser deployment.
Supports FP16, INT8 quantization, and graph optimizations.

Usage:
    python convert.py input.onnx --fp16              # Convert to FP16 (~50% size)
    python convert.py input.onnx --int8              # Dynamic INT8 quantization (~25% size)
    python convert.py input.onnx --optimize          # Graph optimizations only
    python convert.py input.onnx --fp16 --optimize   # FP16 + optimizations
    python convert.py input.onnx --all               # Generate all variants

Requirements:
    pip install onnx onnxruntime onnxconverter-common onnxruntime-tools
"""

import argparse
import os
import sys
from pathlib import Path


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


def convert_to_fp16(input_path: Path, output_path: Path, keep_io_fp32: bool = True) -> bool:
    """
    Convert model weights from FP32 to FP16.
    Roughly halves model size with minimal accuracy loss for most models.
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
        
        # Clear intermediate shapes to avoid conflicts during conversion
        while len(model.graph.value_info) > 0:
            model.graph.value_info.pop()
        
        # Convert to FP16
        # For better compatibility with ONNX Runtime Web, convert everything to FP16
        # (keep_io_types=False means inputs/outputs also become FP16)
        model_fp16 = float16.convert_float_to_float16(
            model,
            keep_io_types=keep_io_fp32,
            disable_shape_infer=True,
            op_block_list=['Softmax', 'LayerNormalization']  # Keep these in FP32 for numerical stability
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
        """
    )
    parser.add_argument("input", type=Path, help="Input ONNX model path")
    parser.add_argument("--info", action="store_true", help="Show model info and exit")
    parser.add_argument("--output-dir", "-o", type=Path, help="Output directory (default: same as input)")
    parser.add_argument("--fp16", action="store_true", help="Convert to FP16 (half precision)")
    parser.add_argument("--int8", action="store_true", help="Apply INT8 dynamic quantization (WARNING: unreliable for classification)")
    parser.add_argument("--int8-static", action="store_true", help="Apply INT8 static quantization (needs calibration)")
    parser.add_argument("--calibration-dir", type=Path, help="Directory with .npy calibration samples")
    parser.add_argument("--optimize", action="store_true", help="Apply graph optimizations")
    parser.add_argument("--ort", action="store_true", help="Convert to ORT format")
    parser.add_argument("--all", action="store_true", help="Generate all variants (FP16, INT8, optimized)")
    parser.add_argument("--validate", action="store_true", help="Validate converted models")
    parser.add_argument("--fp16-io", action="store_true", help="Also convert inputs/outputs to FP16")
    
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
    
    # If no conversion specified, show help
    if not any([args.fp16, args.int8, args.int8_static, args.optimize, args.ort]):
        parser.print_help()
        print("\nERROR: Specify at least one conversion: --fp16, --int8, --optimize, --ort, or --all")
        sys.exit(1)
    
    # 1. Graph optimization (do first, use as input for further conversions)
    optimized_path = args.input
    if args.optimize:
        print("[1/4] Graph Optimization")
        opt_output = output_dir / make_output_name(stem, "optimized")
        if optimize_graph(args.input, opt_output):
            results.append(("Optimized", opt_output))
            if args.validate:
                validate_model(opt_output, args.input)
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
                validate_model(fp16_output, args.input)
        print()
    
    # 3. INT8 dynamic quantization
    if args.int8:
        print("[3/4] INT8 Dynamic Quantization")
        int8_output = output_dir / make_output_name(stem, "INT8")
        if quantize_int8_dynamic(optimized_path, int8_output):
            results.append(("INT8", int8_output))
            if args.validate:
                validate_model(int8_output, args.input)
        print()
    
    # 4. INT8 static quantization
    if args.int8_static:
        print("[3b/4] INT8 Static Quantization")
        int8s_output = output_dir / make_output_name(stem, "INT8-static")
        if quantize_int8_static(optimized_path, int8s_output, args.calibration_dir):
            results.append(("INT8-Static", int8s_output))
            if args.validate:
                validate_model(int8s_output, args.input)
        print()
    
    # 5. ORT format
    if args.ort:
        print("[4/4] ORT Format Conversion")
        ort_output = output_dir / make_output_name(stem, "ORT", ".ort")
        if convert_to_ort_format(optimized_path, ort_output):
            results.append(("ORT", ort_output))
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
