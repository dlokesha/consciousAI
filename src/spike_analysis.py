"""
spike_analysis.py
-----------------
Core spike train analysis for consciousness research.

Key distinction this module targets:
    TRANSIENT responses = early, stimulus-locked firing (feedforward sweep)
    SUSTAINED responses = prolonged firing after stimulus onset (recurrent/feedback)

The hypothesis: sustained activity in posterior visual cortex is the neural
correlate of conscious visual perception. This should be stronger on HIT trials
than MISS trials for identical stimuli.

References:
    Lamme & Roelfsema (2000) - Trends in Neurosciences
    Koch et al. (2016) - Nature Reviews Neuroscience
    Adversarial collaboration (2025) - Nature
"""

import numpy as np
import pandas as pd


# ----- SPIKE RATE COMPUTATION -----

def compute_firing_rate(spike_times, start_time, end_time, bin_size=0.01):
    """
    Compute binned firing rate for a single neuron over a time window.

    Args:
        spike_times: array of spike timestamps (seconds)
        start_time:  window start (seconds)
        end_time:    window end (seconds)
        bin_size:    bin width in seconds (default 10ms)

    Returns:
        rates: firing rate per bin (spikes/sec)
        bins:  bin center times
    """
    bins = np.arange(start_time, end_time, bin_size)
    counts, _ = np.histogram(spike_times, bins=bins)
    rates = counts / bin_size  # convert counts to spikes/sec
    bin_centers = bins[:-1] + bin_size / 2
    return rates, bin_centers


def compute_psth(session, unit_id, stimulus_times, pre=0.1, post=0.5, bin_size=0.01):
    """
    Peri-Stimulus Time Histogram (PSTH) — average firing rate aligned to stimulus onset.

    Args:
        session:        Allen session object
        unit_id:        neuron ID
        stimulus_times: array of stimulus onset times (seconds)
        pre:            time before stimulus to include (seconds)
        post:           time after stimulus to include (seconds)
        bin_size:       bin width (seconds)

    Returns:
        psth:       mean firing rate across trials (spikes/sec)
        bin_centers: time axis relative to stimulus onset
    """
    spike_times = session.spike_times[unit_id]
    bins = np.arange(-pre, post, bin_size)
    all_counts = []

    for t0 in stimulus_times:
        # Align spikes to this stimulus onset
        aligned = spike_times - t0
        counts, _ = np.histogram(aligned, bins=bins)
        all_counts.append(counts / bin_size)  # spikes/sec

    psth = np.mean(all_counts, axis=0)
    bin_centers = bins[:-1] + bin_size / 2
    return psth, bin_centers


# ----- TRANSIENT vs SUSTAINED DECOMPOSITION -----

def compute_sustained_index(psth, bin_centers, transient_window=(0.0, 0.1), sustained_window=(0.1, 0.4)):
    """
    Sustained Index (SI): quantifies how much of a neuron's response is sustained
    vs transient after stimulus onset.

    SI > 0 → neuron maintains elevated activity (sustained)
    SI < 0 → neuron fires early then goes quiet (transient)
    SI = 0 → equal transient and sustained components

    Formula: SI = (sustained_rate - transient_rate) / (sustained_rate + transient_rate)

    Args:
        psth:             firing rate array (spikes/sec)
        bin_centers:      time axis relative to stimulus onset
        transient_window: (start, end) in seconds for transient epoch
        sustained_window: (start, end) in seconds for sustained epoch

    Returns:
        float: sustained index (-1 to 1)
    """
    def mean_rate_in_window(start, end):
        mask = (bin_centers >= start) & (bin_centers < end)
        return np.mean(psth[mask]) if mask.any() else 0.0

    transient_rate = mean_rate_in_window(*transient_window)
    sustained_rate = mean_rate_in_window(*sustained_window)

    total = transient_rate + sustained_rate
    if total == 0:
        return 0.0

    return (sustained_rate - transient_rate) / total


# ----- HIT vs MISS COMPARISON -----

def compare_hit_miss_psth(session, unit_id, hit_times, miss_times, pre=0.1, post=0.5):
    """
    Compute PSTH separately for HIT and MISS trials for a single neuron.
    This is the core comparison for conscious vs non-conscious processing.

    Returns:
        dict with hit_psth, miss_psth, bin_centers, and delta_psth (hit - miss)
    """
    hit_psth,  bins = compute_psth(session, unit_id, hit_times,  pre=pre, post=post)
    miss_psth, _    = compute_psth(session, unit_id, miss_times, pre=pre, post=post)

    return {
        'hit_psth':   hit_psth,
        'miss_psth':  miss_psth,
        'bin_centers': bins,
        'delta_psth': hit_psth - miss_psth  # positive = more active on conscious trials
    }


def compute_population_sustained_index(session, units, stimulus_times, region=None):
    """
    Compute sustained index for all neurons in a region.
    Gives a population-level view of how much posterior cortex
    maintains stimulus representations (key for consciousness hypothesis).

    Returns:
        pd.DataFrame: unit_id, brain region, sustained index
    """
    if region:
        units = units[units['structure_acronym'] == region]

    results = []
    for unit_id in units.index:
        try:
            psth, bins = compute_psth(session, unit_id, stimulus_times)
            si = compute_sustained_index(psth, bins)
            results.append({
                'unit_id':           unit_id,
                'structure_acronym': units.loc[unit_id, 'structure_acronym'],
                'sustained_index':   si
            })
        except Exception:
            continue  # skip units with insufficient spikes

    return pd.DataFrame(results)
