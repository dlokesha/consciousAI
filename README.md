# Neural Correlates of Visual Consciousness
### Using Allen Brain Observatory Neuropixels Data

A computational neuroscience project investigating **what sustained neural dynamics in posterior visual cortex distinguish conscious from non-conscious visual processing** — using large-scale electrophysiology from the Allen Brain Observatory.

---

## Research Question

When a mouse detects a visual change (HIT) vs misses it (MISS), the stimulus is identical — only the animal's internal state differs. What happens in the brain during those two conditions?

This project uses the **behavioral detection contrast** (hit vs miss trials) as a proxy for conscious vs non-conscious visual processing, and analyzes:

- **Sustained vs transient firing** in posterior visual cortex neurons
- **Cross-region LFP synchrony** between visual and frontal areas (gamma-band coherence)
- **Population-level dynamics** across the posterior "hot zone"

Motivated by the 2025 *Nature* adversarial collaboration (Koch, Dehaene et al.) which found that neither IIT nor GNWT fully account for the neural correlates of consciousness, leaving the **mechanism of sustained posterior activity** as an open question.

---

## Dataset

**Allen Brain Observatory — Visual Behavior Neuropixels**
- ~200,000 recorded neurons across 153 sessions
- Regions: visual cortex (V1–V5), thalamus, hippocampus, frontal areas
- Task: change detection (go/no-go) with natural images
- Behavioral labels: hit, miss, false alarm, correct rejection per trial

Access via [AllenSDK](https://allensdk.readthedocs.io/en/latest/visual_behavior_neuropixels.html)

---

## Project Structure

```
consciousness-neuro/
├── src/
│   ├── data_loader.py      # AllenSDK wrappers, hit/miss trial extraction
│   ├── spike_analysis.py   # PSTH, sustained index, hit vs miss comparison
│   ├── synchrony.py        # LFP coherence, spike count correlations
│   └── visualizations.py   # All publication-ready figures
├── models/                 # BMTK network models (Phase 3)
├── results/                # Generated figures
├── docs/                   # Notes, literature, paper drafts
├── main.py                 # Pipeline entry point
├── environment.yml         # Reproducible conda environment
└── .cursorrules            # AI agent context for Cursor
```

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/consciousness-neuro.git
cd consciousness-neuro

# 2. Create conda environment
conda env create -f environment.yml
conda activate consciousness-neuro

# 3. Run the pipeline
python main.py --phase data      # explore sessions
python main.py --phase spikes    # hit vs miss spike analysis
python main.py                   # full pipeline
```

> **Note:** First run will download AllenSDK manifest (~1MB). Session NWB files download on demand (1–5GB each) and are cached locally in `data/`.

---

## Key Analysis Modules

### `spike_analysis.py`
- **PSTH computation** aligned to stimulus onset
- **Sustained Index (SI)**: quantifies transient vs sustained response character
- **Hit vs Miss PSTH comparison**: per-neuron and population-level

### `synchrony.py`
- **Spike count correlation**: trial-to-trial co-variability between neuron pairs
- **LFP coherence**: gamma-band synchrony between posterior visual cortex and frontal regions

### `visualizations.py`
- Hit vs Miss PSTH overlay plots
- Population sustained index violin plots by region
- Hit vs Miss sustained index scatter (key figure)
- LFP coherence spectra

---

## Theoretical Framework

| Prediction | Theory | Measurement |
|---|---|---|
| Sustained activity in posterior cortex on HITs | IIT | Sustained Index (SI) |
| Long-range gamma coherence on HITs | GNWT | LFP coherence VIS ↔ Frontal |
| No prefrontal requirement for consciousness | Koch 2025 | Region-specific SI comparison |

---

## Roadmap

- [x] Data loading pipeline (AllenSDK)
- [x] Hit vs miss spike analysis
- [x] LFP coherence framework
- [ ] Population decoding (can you predict hit/miss from neural activity?)
- [ ] BMTK network model of sustained dynamics
- [ ] Quarto report / paper scaffold

---

## Tools & Stack

- **AllenSDK** — data access
- **BMTK** — network simulation (Phase 3)
- **Cursor + Claude** — agentic development
- **Quarto** — reproducible reports / paper generation
- **Python 3.9**, numpy, scipy, pandas, matplotlib, seaborn

---

## License

MIT
