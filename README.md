<div align="center"><img width="300" alt="BirdNET+ logo" src="logo-birdnet-circle.png"></div>

# birdnet-V3.0-dev
CLI to analyze audio with BirdNET+ V3.0 developer preview models and export of per-chunk detections.

**Key changes vs earlier model versions:**
- Variable-length input (removed fixed 3 s constraint)
- Model takes 32 kHz audio input (compared to 48 kHz previously)
- Improved architecture and training procedure
- Much larger and more diverse training dataset
- Expanded set of non-bird species

**Pending revisions:**
- Cross-platform / cross-framework packaging
- Species list curation (inclusion/exclusion based on data availability)
- Final architecture and model size
- Additional non-target / environmental classes (human, rain, wind, engines, etc.)

Note: Developer preview; models, labels, and code will change. Trained on a subset of data and may not reflect final performance.

## Install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Upon first run, the default model and labels will be automatically downloaded to the `models/` directory. You can download them manually from [Zenodo](https://zenodo.org/record/17571190).

Run the analysis with:

```bash
python analyze.py /path/to/audio.wav
```

### Options
- `--model` Path to model file (default: models/BirdNET+_V3.0-preview1_EUNA_1K_FP32.pt)
- `--labels` Path to labels CSV (default: models/BirdNET+_V3.0-preview1_EUNA_1K_Labels.csv)
- `--chunk_length` Chunk length in seconds (default: 3.0)
- `--device` cpu|cuda (default: auto)
- `--min-conf` Minimum confidence threshold for exporting detections (default: 0.15)
- `--out-csv` Output CSV path (default: <audio>.results.csv)

### Output
- Per-chunk CSV with columns: `name,start_sec,end_sec,confidence,label`
- One row per (chunk, label) with confidence ≥ `--min-conf`
- Multiple rows per chunk if multiple labels exceed threshold

## Examples
```bash
# Minimal (uses defaults where available)
python analyze.py example/soundscape.wav

# Specify model, chunk length, min confidence, and output CSV
python analyze.py example/soundscape.wav --chunk_length 2.0 --min-conf 0.2 --out-csv results.csv

# Specify model and run on GPU
python analyze.py example/soundscape.wav --model models/BirdNET+_V3.0-preview1_EUNA_1K_FP32.pt --device cuda
```

## Streamlit web app

An interactive UI to upload audio, view a spectrogram, run the model, and visualize results.

### Start the app
```bash
# Activate your venv first if you use one
source .venv/bin/activate

# Run Streamlit
streamlit run app.py
```

- The app opens in your browser (usually http://localhost:8501).
- On first run, if you keep the default paths, the model and labels will be downloaded into models/.

Headless/server usage (Linux):
```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
# then open http://<server-ip>:8501 in your browser
```

### How to use
- Upload an audio file (wav, mp3, ogg, flac, m4a).
- Adjust settings in the sidebar:
  - Chunk length (s), Overlap (s)
  - Min confidence threshold
  - Device (cpu/cuda, if available)
- The app will:
  - Render a mel spectrogram of the audio
  - Show an overall bar chart of aggregated scores (top-N)
  - List per-chunk detections (sorted by score) with a Download CSV button

## License

- **Source Code**: The source code for this project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
- **Models**: The models used in this project are licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/).

Please ensure you review and adhere to the specific license terms provided with each model.

## Terms of Use

Please refer to the [TERMS OF USE](TERMS_OF_USE.md) file for detailed terms and conditions regarding the use of the BirdNET+ V3.0 developer preview models.

## Citation

Lasseck, M., Eibl, M., Klinck, H., & Kahl, S. (2025). BirdNET+ V3.0 model developer preview (Preview 1). Zenodo. https://doi.org/10.5281/zenodo.17571190

```bibtex
@dataset{lasseck2025birdnet,
  title     = {BirdNET+ V3.0 model developer preview (Preview 1)},
  author    = {Lasseck, M. and Eibl, M. and Klinck, H. and Kahl, S.},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.17571190},
  url       = {https://doi.org/10.5281/zenodo.17571190}
}
```

## Funding

Our work in the K. Lisa Yang Center for Conservation Bioacoustics is made possible by the generosity of K. Lisa Yang to advance innovative conservation technologies to inspire and inform the conservation of wildlife and habitats.

The development of BirdNET is supported by the German Federal Ministry of Research, Technology and Space (FKZ 01|S22072), the German Federal Ministry for the Environment, Climate Action, Nature Conservation and Nuclear Safety (FKZ 67KI31040E), the German Federal Ministry of Economic Affairs and Energy (FKZ 16KN095550), the Deutsche Bundesstiftung Umwelt (project 39263/01) and the European Social Fund.

## Partners

BirdNET is a joint effort of partners from academia and industry.
Without these partnerships, this project would not have been possible.
Thank you!

![Our partners](https://tuc.cloud/index.php/s/KSdWfX5CnSRpRgQ/download/box_logos.png)
