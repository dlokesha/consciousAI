"""
data_loader.py
--------------
Wraps AllenSDK EcephysProjectCache to pull Visual Behavior Neuropixels sessions.

Research focus:
    Characterizing sustained neural dynamics in posterior visual cortex
    that distinguish detected (HIT) vs undetected (MISS) stimuli —
    a behavioral proxy for conscious visual awareness.

Data source:
    Allen Brain Observatory — Visual Behavior Neuropixels dataset
    https://allensdk.readthedocs.io/en/latest/visual_behavior_neuropixels.html
"""

import os
import pandas as pd
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorNeuropixelsProjectCache

# ----- CONFIG -----
DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
MANIFEST_PATH = os.path.join(DATA_DIR, 'manifest.json')

# Posterior visual cortex regions — the "hot zone" implicated in visual consciousness
# (Koch et al., 2016; 2025 Nature adversarial collaboration)
POSTERIOR_VISUAL_REGIONS = ['VISp', 'VISl', 'VISal', 'VISpm', 'VISam', 'VISrl']


# ----- CACHE -----

def get_cache():
    """
    Initialize AllenSDK project cache.
    Downloads manifest on first run (~1MB). NWB session files download on demand.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    return VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=DATA_DIR)


# ----- SESSION SELECTION -----

def get_sessions(cache, targeted_structures=None):
    """
    Returns filtered session table for sessions that recorded from
    posterior visual cortex regions of interest.

    Args:
        cache: initialized EcephysProjectCache
        targeted_structures: list of brain region acronyms to filter by
                             (defaults to POSTERIOR_VISUAL_REGIONS)

    Returns:
        pd.DataFrame: filtered session metadata table
    """
    if targeted_structures is None:
        targeted_structures = POSTERIOR_VISUAL_REGIONS

    sessions = cache.get_session_table()

    # Keep sessions that recorded from at least one target region
    mask = sessions['structure_acronyms'].apply(
        lambda structures: any(s in str(structures) for s in targeted_structures)
    )

    filtered = sessions[mask]
    print(f"Found {len(filtered)} sessions with recordings from {targeted_structures}")
    return filtered


# ----- SESSION DATA -----

def get_session_data(cache, session_id):
    """
    Load a single session by ID.
    First call downloads NWB file (1–5GB) — subsequent calls read from local cache.

    Returns:
        session: full session object
        units:   DataFrame of neuron metadata (brain region, quality metrics)
        stimuli: DataFrame of stimulus presentations (timing, image identity)
    """
    print(f"Loading session {session_id} (may download NWB file on first run)...")
    session = cache.get_ecephys_session(ecephys_session_id=session_id)

    units = session.units          # one row per neuron
    stimuli = session.stimulus_presentations  # one row per stimulus shown

    print(f"  Units: {len(units)} neurons")
    print(f"  Stimuli: {len(stimuli)} presentations")
    return session, units, stimuli


# ----- HIT vs MISS TRIALS (consciousness proxy) -----

def get_hit_miss_trials(session):
    """
    Core split for consciousness research:
    HITs  = animal licked in the response window → detected the change → conscious awareness proxy
    MISSes = animal did not lick               → change went unnoticed

    This contrast is the closest behavioral proxy for conscious vs non-conscious
    visual processing available in the Allen dataset.

    Returns:
        hits:   DataFrame of detected-change trials
        misses: DataFrame of missed-change trials
    """
    trials = session.trials

    hits   = trials[trials['hit']  == True].copy()
    misses = trials[trials['miss'] == True].copy()

    print(f"  Hits: {len(hits)} | Misses: {len(misses)}")
    return hits, misses


# ----- UNIT FILTERING -----

def get_quality_units(units, regions=None):
    """
    Filter neurons by quality metrics and optionally by brain region.
    Uses Allen's recommended thresholds for high-quality single units.

    Args:
        units:   DataFrame from session.units
        regions: list of structure acronyms to keep (None = keep all)

    Returns:
        pd.DataFrame: filtered high-quality units
    """
    # Allen-recommended quality thresholds
    quality_mask = (
        (units['snr'] >= 1.0) &
        (units['isi_violations'] <= 0.5) &
        (units['presence_ratio'] >= 0.9) &
        (units['amplitude_cutoff'] <= 0.1)
    )

    filtered = units[quality_mask]

    if regions:
        filtered = filtered[filtered['structure_acronym'].isin(regions)]

    print(f"  Quality units: {len(filtered)} / {len(units)} total")
    return filtered
