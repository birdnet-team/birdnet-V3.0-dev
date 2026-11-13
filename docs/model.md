# BirdNET+ V3.0 model and training

This page describes how BirdNET+ V3.0 Global 12K is built and trained. For how to run it, see the [README](../README.md#tips-and-details). For how well it works, see the [WABAD benchmark](wabad.md).

## At a glance

| | |
| --- | --- |
| Backbone | EfficientNetV2-S, ImageNet-21k pretrained (`tf_efficientnetv2_s_in21k` from [timm](https://github.com/huggingface/pytorch-image-models)) |
| Classifier | Three multi-label heads: global GeM pooling, temporal attention, frequency attention |
| Head weights | 0.40 / 0.35 / 0.25 |
| Input | 32 kHz mono waveform, any length (3 s matches training best) |
| Spectrogram | 128-band log-mel, 2,048-point FFT, 512-sample hop, resized to 224 pixels high |
| Outputs | 11,725 class scores and a 1280-d embedding |
| Training data | Macaulay Library, iNaturalist, Xeno-canto, Animal Sound Archive Berlin, Observation.org, and non-public collections |
| Compute | 1.38 GMAC per second of audio |
| Optimizer | Adam, learning rate 0.001, cosine schedule with warmup |
| Batch size / epochs | 42 / 40 |
| Loss | BCE-with-logits plus sigmoid focal loss, for each head |

## Architecture

### From waveform to spectrogram

The model takes a raw waveform and computes the spectrogram itself, so you only need to resample your audio to 32 kHz.

1. Short-time Fourier transform with a 2,048-point FFT and a 512-sample hop. At 32 kHz this gives 62.5 frames per second, about 188 frames for a 3 s chunk.
2. A mel filterbank reduces the 1,025 frequency bins to 128 mel bands, up to about 15 kHz.
3. Log compression.
4. The spectrogram is resized to 224 pixels high, the input size the backbone was pretrained on. The width follows the input length, so the model accepts recordings of any duration.

Because the spectrogram is part of the model, you never need to reproduce this preprocessing in your own code.

### Backbone

An EfficientNetV2-S encoder, initialized from ImageNet-21k weights, turns the spectrogram into a feature map with 1,280 channels over a coarse grid of frequency and time. The embedding output (`embeddings_out`, 1280-d) comes from this feature map.

### Three classifier heads

Each head looks at the same feature map in a different way and predicts all classes independently (multi-label, one sigmoid per class):

- **Global GeM pooling.** Generalized mean pooling over frequency and time, followed by a linear classifier. A robust summary of the whole chunk.
- **Temporal attention.** Learns which time frames matter for each class and weighs them accordingly. Helps with short calls in long or noisy chunks.
- **Frequency attention.** Learns which frequency bands matter for each class. Helps separate species that call at the same time at different pitches. Lasseck (2023) describes this head in detail.

The final score combines the three heads, weighted 0.40 (GeM), 0.35 (temporal) and 0.25 (frequency).

The `_pruned` model files keep only one of the three heads. They are less than half the size, at the cost of some accuracy (scores differ by up to about 0.14 from the full model).

## Training

### Data

Most training recordings come from large public sound archives. The table shows the date each archive was last collected.

| Archive | Collected up to |
| --- | --- |
| [Macaulay Library](https://www.macaulaylibrary.org/) | 17 Mar 2026 |
| [iNaturalist](https://www.inaturalist.org/) | 12 Jan 2026 |
| [Xeno-canto](https://xeno-canto.org/) | 6 Jan 2026 |
| [Animal Sound Archive Berlin](https://www.tierstimmenarchiv.de/) (Tierstimmenarchiv) | 23 Apr 2025 |
| [Observation.org](https://observation.org/) | 28 Dec 2024 |

Training data also included non-public collections, such as validated snippets from collaborators.

### Data for evaluation

[WABAD](https://zenodo.org/records/14191524) and [BirdSet](https://huggingface.co/datasets/DBD-research-group/BirdSet) were not used for training, so you can use both to evaluate the model (see our [WABAD results](wabad.md)). For BirdSet, use the soundscape test sets: its training split comes from Xeno-canto, which overlaps with our training data. Recordings uploaded to the archives above after their cutoff dates were not used either. Keep in mind that test recordings may come from the same sites or recordists as the training data.

### Classes

The model was trained on 16,852 classes. For the release, the outputs were reduced to 11,725 classes: 11,709 animal species and 16 non-animal classes. The unpruned model with all training classes is in `_Source_Checkpoint.zip` on [Zenodo](https://zenodo.org/records/24162518).

### Optimization

- Adam optimizer, learning rate 0.001
- Cosine learning rate schedule with warmup
- Batch size 42
- 40 epochs

### Loss

Each head gets its own loss: binary cross-entropy with logits plus sigmoid focal loss. Focal loss puts more weight on hard examples, which helps with the large imbalance between common and rare classes. The training loss combines the losses of all three heads.

### Data augmentation

- **Audio and noise mixup.** Training samples are mixed with other recordings and with background noise, so the model learns to recognize overlapping calls and calls in noisy soundscapes.
- **Variable durations.** Training chunks vary in length, so the model works on inputs of different lengths.
- **Spectrogram transforms.** Transformations of the spectrogram in time and frequency.
- **Label smoothing.** Softens the targets to reduce overconfidence and the effect of label noise.
- **Pseudo-labels.** Where pseudo-labels exist for a training recording, they replace the original multi-label targets with a probability of 85%.

## References

More details on the architecture, training and data augmentation:

- Lasseck M (2023) Bird Species Recognition using Convolutional Neural Networks with Attention on Frequency Bands. In: Working Notes of CLEF 2023, CEUR Workshop Proceedings, Vol. 3497, pp. 2071–2079. [PDF](https://ceur-ws.org/Vol-3497/paper-175.pdf)
- Lasseck M (2019) Bird Species Identification in Soundscapes. In: Working Notes of CLEF 2019, CEUR Workshop Proceedings, Vol. 2380. [PDF](https://ceur-ws.org/Vol-2380/paper_86.pdf)

To cite the model itself, see [Citation](../README.md#citation) in the README.
