import numpy as np
from sound_quilter import quilter as qtr
from matplotlib import pyplot as plt
from pycochleagram.utils import play_array
import scipy.io as spio

# TODO: how to test houskeeping methods in the app

# PATH_TO_REF_ARRAYS = "sound_quilter/tests/ref/"
PATH_TO_REF_ARRAYS = "ref/"

def test_split_array():
    signal = np.ones([30, 50])
    len_segment_samples = 3
    segments = qtr.split_array(signal, len_segment_samples)
    assert segments.shape[0] == signal.shape[-1] // len_segment_samples
    assert segments[0].shape == (30, 3)


def test_make_window():
    window_reference = np.load(PATH_TO_REF_ARRAYS + "window.npy")
    len_overlap_samples = 600
    len_segment_samples = 1200
    len_sides = len_overlap_samples
    len_middle = len_segment_samples - len_overlap_samples
    window = qtr.make_window(len_sides, len_middle)
    assert (window == window_reference).all()

def test_compute_distances():
    # TODO: test with borders where the 2 arrays compared are not the same
    arrays = np.load(PATH_TO_REF_ARRAYS + "arrays_for_compute_distance.npy")
    distance_matrix = qtr.compute_distances(arrays)
    assert np.trace(distance_matrix) == 0  # sum along diagonal should be zero
    assert np.all(distance_matrix.shape == (arrays.shape[0], arrays.shape[0]))
    # test case where we pass 2 sets of different number of arrays
    arrays_2 = arrays.copy()[0:3, ...]
    distance_matrix = qtr.compute_distances(arrays, arrays_2)
    assert np.trace(distance_matrix) == 0  # sum along diagonal should be zero
    assert np.all(distance_matrix.shape == (arrays.shape[0], arrays_2.shape[0]))


def test_find_sequence_similar_diagonal():
    # TODO: test with sensible distances
    distance_matrix = np.arange(1, 101).reshape([10, 10]).astype(np.float)
    len_sequence = 6
    index_sequence = qtr.find_sequence_similar_diagonal(distance_matrix, len_sequence)
    assert index_sequence.shape[0] == len_sequence

def test_find_optimal_location():
    vector = np.sin(2*np.pi * np.linspace(0, 1, 10))
    candidates = np.random.randn(100) * 0.3  # small amplitude noise
    actual_best_location = 20
    candidates[actual_best_location:30] = vector  # simulate a region where it should yield highest correlation
    obtained_best_location = qtr.find_optimal_location(vector, candidates)
    assert obtained_best_location == actual_best_location

def test_join_segments_psola():
    signal_size = int(2e4)
    signal = np.random.randn(signal_size)
    len_segment = 1200
    ordered_initial_locations = np.arange(0, signal_size, len_segment)
    window = np.load(PATH_TO_REF_ARRAYS + "window.npy")
    max_shift = 300
    len_overlap = 600
    joined_segments = qtr.join_segments_psola(
        signal, ordered_initial_locations,
        window, max_shift,
        len_segment, len_overlap
        )

def test_sound_quilter():
    from scipy.signal import spectrogram, resample_poly
    srate = int(2e4)
    num_secs = 10
    num_samples = num_secs * srate
    # signal = np.sin(2*np.pi * np.linspace(0, num_secs, num_samples))
    # signal = np.random.randn(num_samples)
    # path2signal = PATH_TO_REF_ARRAYS + "Laughter.wav"
    path2signal = "/Users/federico.adolfi/Projects/natural_sounds_db/stim387_typing.wav"
    srate_loaded, signal = spio.wavfile.read(path2signal)
    if srate_loaded != srate:
        signal = resample_poly(signal, srate, srate_loaded, axis=0)
    freqs, times, sgram = spectrogram(signal, srate)
    quilter = qtr.SoundQuilter()
    config = {
        # Attributes set by the user
        "srate": srate,
        "len_segment_samples": 1200,
        "len_overlap_samples": 600,
        "len_quilt_samples": int(srate * 1),
        "len_border_samples": 300,  # defines the extent to use for distance calculation
        # "_distance_metric": None,  # should work with broadcasting
        # for selecting via maximizing cross-correlation (PSOLA),
        "max_shift_samples": 300,
        "signal": signal,
        }
    quilter.configure(config)
    quilter.register_custom_transform(lambda x: spectrogram(x, srate)[2])
    quilt = quilter.make_quilt()
    plt.plot(quilt)
    play_array(quilt, srate, rescale="normalize", ignore_warning=True)
    play_array(signal, srate, rescale="normalize", ignore_warning=True)
    assert quilt.shape[0] == config["len_quilt_samples"]  # TODO: what to do about extremes