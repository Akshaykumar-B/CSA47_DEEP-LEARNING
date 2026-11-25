"""Separate two simulated microphone recordings with ICA and measure recovery."""

import itertools

import numpy as np
from sklearn.decomposition import FastICA


rng = np.random.default_rng(73)
sample_rate = 8000
time = np.arange(sample_rate * 2) / sample_rate
voice_a = np.sin(2 * np.pi * 180 * time) + 0.35 * np.sin(2 * np.pi * 360 * time)
voice_b = np.sin(2 * np.pi * 260 * time + 0.4) + 0.25 * np.sin(2 * np.pi * 520 * time)
sources = np.column_stack([voice_a, voice_b])
sources += rng.normal(0, 0.02, sources.shape)
sources = (sources - sources.mean(axis=0)) / sources.std(axis=0)
mixing = np.array([[1.0, 0.65], [0.35, 1.0]])
microphone_signals = sources @ mixing.T
recovered = FastICA(
    n_components=2, random_state=73, max_iter=1500, tol=0.001
).fit_transform(microphone_signals)
correlation = np.corrcoef(sources.T, recovered.T)[:2, 2:]
permutation = max(
    itertools.permutations(range(2)),
    key=lambda choice: sum(abs(correlation[index, choice[index]]) for index in range(2)),
)
for source_index, component_index in enumerate(permutation):
    estimate = recovered[:, component_index]
    scale = np.dot(sources[:, source_index], estimate) / np.dot(estimate, estimate)
    aligned = scale * estimate
    signal_power = np.mean(sources[:, source_index] ** 2)
    error_power = np.mean((sources[:, source_index] - aligned) ** 2)
    snr_db = 10 * np.log10(signal_power / max(error_power, 1e-12))
    print(
        f"Voice {source_index + 1}: correlation={abs(correlation[source_index, component_index]):.3f}, "
        f"aligned SNR={snr_db:.2f} dB"
    )
print("The microphone recordings are simulated tones; real voice separation needs real recordings.")
