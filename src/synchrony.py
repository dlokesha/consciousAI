"""
synchrony.py
------------
Cross-region neural synchrony analysis.

Why synchrony matters for consciousness:
    Global Neuronal Workspace Theory (Dehaene & Changeux) predicts that
    conscious access requires long-range synchrony — particularly gamma-band
    coherence between posterior sensory areas and frontal regions.

    IIT (Tononi) predicts high integrated information (phi) within posterior
    cortical complexes, measurable as correlated activity within the hot zone.

    This module computes both, letting the data speak.
"""

import numpy as np
from scipy import signal
from scipy.stats import pearsonr


# ----- SPIKE CORRELATION -----

def compute_spike_count_correlation(session, unit_id_1, unit_id_2, stimulus_times,
                                     window=(0.0, 0.4), bin_size=0.05):
    """
    Spike count correlation (r_sc / "noise correlation") between two neurons
    across repeated stimulus presentations.

    High correlation = neurons share trial-to-trial variability
    = potential indicator of common input or functional coupling.

    Args:
        session:        Allen session object
        unit_id_1/2:    neuron IDs to compare
        stimulus_times: array of stimulus onset times
        window:         (start, end) relative to stimulus onset (seconds)
        bin_size:       bin width for counting spikes

    Returns:
        r:   Pearson correlation coefficient
        p:   p-value
    """
    def count_spikes(unit_id):
        spike_times = session.spike_times[unit_id]
        counts = []
        for t0 in stimulus_times:
            mask = (spike_times >= t0 + window[0]) & (spike_times < t0 + window[1])
            counts.append(mask.sum())
        return np.array(counts, dtype=float)

    counts_1 = count_spikes(unit_id_1)
    counts_2 = count_spikes(unit_id_2)

    if counts_1.std() == 0 or counts_2.std() == 0:
        return 0.0, 1.0  # no variance, no correlation

    r, p = pearsonr(counts_1, counts_2)
    return r, p


# ----- LFP COHERENCE -----

def compute_lfp_coherence(lfp_1, lfp_2, fs=1250.0, freq_band=(30, 80)):
    """
    Compute spectral coherence between two LFP signals in a frequency band of interest.

    Gamma coherence (30–80 Hz) between regions = candidate marker for
    conscious broadcast in GNWT framework.

    Args:
        lfp_1, lfp_2: LFP time series arrays (same length)
        fs:           sampling rate in Hz (Allen default: 1250 Hz)
        freq_band:    (low, high) Hz — default is gamma band

    Returns:
        mean_coherence: average coherence in the specified band (0–1)
        freqs:          frequency array
        coherence:      full coherence spectrum
    """
    freqs, coherence = signal.coherence(lfp_1, lfp_2, fs=fs, nperseg=int(fs * 0.5))

    # Average coherence within the target frequency band
    band_mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])
    mean_coherence = np.mean(coherence[band_mask])

    return mean_coherence, freqs, coherence


def compute_cross_region_coherence(session, region_1, region_2,
                                    stimulus_times, pre=0.1, post=0.5,
                                    freq_band=(30, 80)):
    """
    Compute mean LFP coherence between two brain regions across stimulus trials.

    This operationalizes the long-range synchrony prediction of GNWT:
    if consciousness correlates with global broadcasting, HIT trials should show
    higher coherence between posterior visual cortex and frontal regions.

    Args:
        session:        Allen session object
        region_1/2:     structure acronyms (e.g., 'VISp', 'FRP')
        stimulus_times: stimulus onset times
        freq_band:      frequency band for coherence (Hz)

    Returns:
        dict: mean coherence per trial, split-able by hit/miss
    """
    lfp = session.get_lfp()  # xarray DataArray: channels × time

    # Get channel IDs for each region
    channels = session.channels
    ch_1 = channels[channels['structure_acronym'] == region_1].index
    ch_2 = channels[channels['structure_acronym'] == region_2].index

    if len(ch_1) == 0 or len(ch_2) == 0:
        raise ValueError(f"No channels found for {region_1} or {region_2}")

    # Use the first available channel per region (or average across channels)
    lfp_1 = lfp.sel(channel=ch_1[0]).values
    lfp_2 = lfp.sel(channel=ch_2[0]).values

    fs = 1.0 / np.median(np.diff(lfp.time.values))
    trial_coherences = []

    for t0 in stimulus_times:
        # Extract LFP segment around this stimulus
        time_mask = (lfp.time.values >= t0 - pre) & (lfp.time.values < t0 + post)
        seg_1 = lfp_1[time_mask]
        seg_2 = lfp_2[time_mask]

        if len(seg_1) < 10:
            continue

        coh, _, _ = compute_lfp_coherence(seg_1, seg_2, fs=fs, freq_band=freq_band)
        trial_coherences.append({'stimulus_time': t0, 'coherence': coh})

    return trial_coherences
