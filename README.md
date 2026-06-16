<div align="center"><img width="300" alt="BirdNET+ logo" src="img/logo-birdnet-circle.png"></div>

# BirdNET+ V3.0 Developer Preview

Analyze audio with BirdNET+ V3.0 developer preview models. This repository provides multiple ways to run the model:

| Tool | Description | Best For |
|------|-------------|----------|
| `analyze.py` | Command-line batch processing | Processing many files, scripting |
| `app.py` | Streamlit interactive UI | Quick experimentation, visualization |
| `web-demo/` | Browser-only demo (ONNX) | Sharing, no Python needed |

### About This Release

**Current version:** Developer Preview 3.1 - 11K Species (June 2026)

**Key changes vs 2.X:**
- Variable-length input (removed fixed 3s constraint)
- 32 kHz audio input (was 48 kHz)
- Improved architecture and training
- Larger, more diverse training dataset
- Expanded non-bird species

**Known limitations:**
- No human voice detection yet
- Limited non-target sound handling (rain, wind, engines)
- Species list needs cleanup

> ⚠️ **Developer Preview Notice:** Models, labels, and code will change before final release.

You can download the latest models and labels from [Zenodo](https://zenodo.org/records/20703646) or run the tools which will download them automatically on first use.

## Table of Contents

- [Quick Start](#quick-start)
- [Tools Overview](#tools-overview)
  - [Option A: Command-Line Analysis](#option-a-command-line-analysis-analyzepy)
  - [Option B: Streamlit Web App](#option-b-streamlit-web-app-apppy)
  - [Option C: Browser Demo](#option-c-browser-demo-web-demo)
- [Model Conversion](#model-conversion-optional)
- [Installation Details](#installation-details)
- [License](#license)
- [Terms of Use](#terms-of-use)
- [Citation](#citation)
- [Funding](#funding)
- [Partners](#partners)

---

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/birdnet-team/birdnet-V3.0-dev.git
cd birdnet-V3.0-dev

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Mac/Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies  
pip install -r requirements.txt

# 4. Run analysis (model downloads automatically on first run)
python analyze.py example/soundscape.wav
```

---

## Tools Overview

### Option A: Command-Line Analysis (`analyze.py`)

Best for batch processing and scripting.

```bash
python analyze.py /path/to/audio.wav
```

**Options:**
| Flag | Default | Description |
|------|---------|-------------|
| `--model` | PyTorch FP32 | Path to model (.pt or .onnx) |
| `--chunk_length` | 3.0 | Chunk length in seconds |
| `--overlap` | 0.0 | Chunk overlap in seconds |
| `--min-conf` | 0.15 | Minimum confidence threshold |
| `--device` | auto | `cpu` or `cuda` |
| `--out-csv` | `<audio>.results.csv` | Output CSV path |
| `--export-embeddings` | false | Include embeddings in output |

**Examples:**
```bash
# Basic usage (PyTorch model, downloads automatically)
python analyze.py example/soundscape.wav

# Use FP16 ONNX model (recommended: smaller, same accuracy)
python analyze.py example/soundscape.wav --model models/BirdNET+_V3.0-preview3.1_Global_11K_FP16_pruned.onnx

# Custom settings
python analyze.py example/soundscape.wav --chunk_length 2.0 --min-conf 0.2 --out-csv results.csv

# Use GPU
python analyze.py example/soundscape.wav --device cuda
```

**Output:** CSV with columns `name, start_sec, end_sec, confidence, label` (+ `embeddings` if requested)

---

### Option B: Streamlit Web App (`app.py`)

Best for interactive exploration with visual feedback.

![Streamlit app screenshot](img/streamlit-ui-screenshot.png)

```bash
# Start the app
streamlit run app.py

# Or with larger file upload limit (e.g., 2GB)
streamlit run app.py --server.maxUploadSize 2048
```

Opens at http://localhost:8501

**Features:**
- Upload audio (wav, mp3, ogg, flac, m4a)
- View mel spectrogram
- Select model format (FP32 PyTorch or FP16 ONNX)
- Adjust chunk length, overlap, confidence threshold
- Download results as CSV

**Headless/server mode:**
```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

---

### Option C: Browser Demo (`web-demo/`)

Runs entirely in the browser using ONNX Runtime Web. No Python required after build.

```bash
cd web-demo

# Download model and labels (one-time setup)
./scripts/download-model.sh     # Mac/Linux
# .\scripts\download-model.ps1  # Windows

# Install and run
npm install
npm run dev
```

Opens at http://localhost:5173

The default download is the pruned FP16 model (68 MB), which runs efficiently in the browser.

**Build for deployment:**
```bash
npm run build
npm run preview
```

---

## Model Conversion & Species Filtering

For model conversion, custom species filtering, or creating sub-models, please use our dedicated converter repository: [model-converter](https://github.com/birdnet-team/model-converter). This repository has been streamlined to keep only the execution demos, while all export/conversion operations are handled there.

---

## Installation Details

### Windows with CUDA (Optional)

If you have an NVIDIA GPU, enable CUDA support:

```bash
# Check CUDA availability
nvidia-smi

# Install CUDA-enabled PyTorch
pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Model Download

Models are downloaded automatically on first run. To download manually:
- [Download from Zenodo](https://zenodo.org/records/20703646)
- Place in `models/` directory

---

## License

- **Source Code**: The source code for this project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
- **Models**: The models used in this project are licensed under the [Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/).

Please ensure you review and adhere to the specific license terms provided with each model.

## Terms of Use

Please refer to the [TERMS OF USE](TERMS_OF_USE.md) file for detailed terms and conditions regarding the use of the BirdNET+ V3.0 developer preview models.

## Citation

Lasseck, M., Eibl, M., Klinck, H., & Kahl, S. (2026). BirdNET+ V3.0 model developer preview (Preview 3.1). Zenodo. https://doi.org/10.5281/zenodo.20703646

```bibtex
@dataset{lasseck2026birdnet,
  title     = {BirdNET+ V3.0 model developer preview (Preview 3.1)},
  author    = {Lasseck, M. and Eibl, M. and Klinck, H. and Kahl, S.},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20703646},
  url       = {https://doi.org/10.5281/zenodo.20703646}
}
```

## Funding

Our work in the Cornell K. Lisa Yang Center for Conservation Bioacoustics is made possible by the generosity of K. Lisa Yang to advance innovative conservation technologies to inspire and inform the conservation of wildlife and habitats.

The development of BirdNET is supported by the German Federal Ministry of Research, Technology and Space (FKZ 01|S22072), the German Federal Ministry for the Environment, Climate Action, Nature Conservation and Nuclear Safety (FKZ 67KI31040E), the German Federal Ministry of Economic Affairs and Energy (FKZ 16KN095550), the Deutsche Bundesstiftung Umwelt (project 39263/01) and the European Social Fund.

## Partners

BirdNET is a joint effort of partners from academia and industry.
Without these partnerships, this project would not have been possible.
Thank you!

![Our partners](https://tuc.cloud/index.php/s/KSdWfX5CnSRpRgQ/download/box_logos.png)
