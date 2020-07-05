import numpy as np
import scipy.signal as sps
import warnings
import random

# The purpose of this module is to implement existing methods in a generalized way
#   and add new features made possible by the genralization.


# Quilting methods:
#   - PSOLA (Overath et al., 2015)
#   - Random shuffling at various time scales

# Algorithm steps (Quilting + PSOLA)
#   - Choose segment duration
#   - Divide source signal into segments of chosen length
#   - Choose randomly chosen segment (uniformly) as first segment of quilt
#   - Compute average *spectrogram* change (cgram cells; L2 error)
#       between right-hand border of chosen segment
#       and left-hand border of other segments, for all remaining segments.
#       Border of the cochleagram is 30 ms in length (n_filters x 30 ms).
#   - Choose the next stimulus segment to be that which gives a segment change as close to
#       the original change as possible (in the L2 sense).
#   - Repeat procedure until quilt is of the desired length
#   - Concatenate ordered segments using PSOLA
#       - Shift boundaries of the segment forward or backwards at most 15 ms to
#           maximize cross-correlation between the *waveforms* of adjacent segments
#       - Cross-fade between segments using 30-ms raised cosine ramps centered at boundary

# TODO: you can provide the features or the callable
# TODO: features computed every time per signal chunk?
#   this way the sampling rate of the features is not an issue and can accomodate any representation
#   Instead of splitting a pre-computed feature array of the whole signal,
#   compute a feature arrays once the signal has been split into segments
# TODO: consider adding dynamic range compression
# TODO: implement attribute consistency checks
# TODO: add ability to sample segments from signal *with replacement*
#   ...this way one can build quilts that are longer than the source signal
# TODO: add ability to reverse or shuffle within-segment samples

class SoundQuilter():
    def __init__(self):
        # Attributes set by the user
        self.srate = None

        # self.len_segment_secs = None
        self.len_segment_samples = None

        self.len_overlap_samples = None

        # self.len_quilt_secs = None
        self.len_quilt_samples = None
        self._num_quilt_segments = None

        self.len_border_samples = None  # defines the extent to use for distance calculation
        # self.len_border_secs = None

        # Assigned via register_custom_transform()
        self._feature_transform = identity_operation  # identity or function that transforms the signal into features

        # a callable that computes some kind of distance like L2
        self._distance_metric = None  # should work with broadcasting

        # for selecting via maximizing cross-correlation (PSOLA)
        # self.max_shift_secs = None
        self.max_shift_samples = None

        self.signal = None
        self._feature_repr = None

        # Attributes NOT set by the user
        self._window_segment = None
        self._leftover_segments = None  # these keep changing a global state
        self._leftover_indices = None
        self._used_indices = None  # segments used

        self._original_signal_segments = None  # these will be immutable once computed (tuples)
        self._original_signal_locations = None

        self._original_feature_segments = None
        self._original_feature_locations = None

        self._distance_matrix = None

        self._ordered_segments_indices = None

        self._quilt = None  # should not be assigned by user

    def make_quilt(self):
        self._compute_populate_app_attributes()
        return self._quilt

    def configure(self, config_dict):
        for attr_key, attr_value in config_dict.items():
            if attr_key[0] == "_":
                raise Exception(
                    "Attributes starting with '_' are not supposed to be set by the user"
                    )
            if hasattr(self, attr_key):
                setattr(self, attr_key, attr_value)
            else:
                raise KeyError(f"SoundQuilter does not have attribute '{attr_key}'")

    def register_custom_transform(self, custom_transform):
        if not hasattr(custom_transform, "__call__"):
            raise TypeError(
                f"Custom transform must be callable, got type {type(custom_transform)}"
                )
        self._feature_transform = custom_transform

    def register_feature_representation(self, representation):
        # TODO: how to handle overwriting
        self._feature_repr = representation

    def confirm_attributes_available(self):
        empty_attributes = []
        for attr_key, attr_val in vars(self).items():
            # only check existence of attributes assigned by the user
            if attr_key[0] != "_" and attr_val is None:
                empty_attributes.append(attr_key)
        if len(empty_attributes) > 0:
            raise Exception(
                f"Some attributes are supposed to be set by the user and are empty\n"
                f"Please assign the following attributes:\n{empty_attributes}"
                )

    def check_attribute_consistency(self):
        # TODO: if "custom" feature space, there should be a custom transform
        # TODO: if no custom transform, pass a spectrogram or define something with signal
        # TODO: len border samples should be <= than segment len in the relevant feature repr.
        # TODO: overlap should be smaller than border (which is used for similarity ordering)
        # TODO: signal length needs to accomodate an overlap len on each side without use
        # TODO: to pass a feature representation, it has to be at the same sampling rate...
        pass

    def _compute_populate_app_attributes(self):
        self.confirm_attributes_available()
        self._window_segment = self._build_window_segment()
        self._num_quilt_segments = self._compute_num_quilt_segments()
        # TODO: decide how to handle feature computation for whole signal if at all
        # self._feature_repr = self._project_onto_feature_space()

        # make segments and locations leaving out a slice of the signal at the beginning and end
        (self._original_signal_locations,
         self._original_signal_segments) = self._make_locations_segments_with_buffer()
        self._original_feature_segments = self._make_features_from_signal_segments()

        self._distance_matrix = self._build_border_distance_matrix()
        self._ordered_segments_indices = self._build_index_sequence_similar_distance()
        # TODO: with ordered signal locations and ordered segment indices we can move on to PSOLA
        self._quilt = self._pitch_synchronous_overlap_add()

    def _pitch_synchronous_overlap_add(self):
        ordered_signal_locations = self._original_signal_locations[self._ordered_segments_indices]
        return join_segments_psola(
            self.signal, ordered_signal_locations, self._window_segment,
            self.max_shift_samples, self.len_segment_samples, self.len_overlap_samples
            )

    def _build_window_segment(self):
        len_sides = self.len_overlap_samples
        len_middle = self.len_segment_samples - self.len_overlap_samples
        # half the fade covers the middle part, the other half is extra
        return make_window(len_sides, len_middle)

    def _make_locations_segments_with_buffer(self):
        # TODO: this assumes that features are sampled at the same rate as the signal
        # subselect signal leaving out a buffer at beginning and end
        # leave out first and last chunk to allow left-right shift
        middle_slice = slice(self.len_overlap_samples, -self.len_overlap_samples)
        signal_middle = self.signal[middle_slice]
        indices_middle = np.arange(
            self.len_overlap_samples,
            self.signal.shape[0] - self.len_overlap_samples
            )
        # for locations, keep only the initial index
        segment_locations = split_array(indices_middle, self.len_segment_samples)[:, 0]
        signal_segments = split_array(signal_middle, self.len_segment_samples)
        return segment_locations, signal_segments

    def _make_features_from_signal_segments(self):
        feature_segments = []
        for segment in self._original_signal_segments:
            features = self._feature_transform(segment)[np.newaxis, ...]
            feature_segments.append(features)
        return np.concatenate(feature_segments, axis=0)  # (num_subarrays, *feature.shape)

    def _build_border_distance_matrix(self):
        """Will compute the distance matrix independent of the input representation shape."""
        # need to first extract borders from arrays in order to compute distances based on those
        # borders may include the whole segment or just an edge of it
        # todo: features can be downsampled from original, so feature samples may be different
        # Right border for last segment is not included because it doesn't have a "next segment"
        #   this works if number of segments is even and also if it is odd
        # (segment_nr, segment_nr) diagonal in distance matrix has original transition distance
        #   for segment `segment_nr`.

        right_borders = self._original_feature_segments[0:-1:1, :, -self.len_border_samples:]
        left_borders = self._original_feature_segments[1::1, :, 0:self.len_border_samples]
        return compute_distances(right_borders, left_borders)

    def _project_onto_feature_space(self):
        return self.feature_transform(self.signal)

    def _build_index_sequence_similar_distance(self):
        # makes a sequence of segment indices based on maintaining
        #   a similar distances to original transitions
        sequence = find_sequence_similar_diagonal(
            self._distance_matrix, self._num_quilt_segments
            ) + 1  # add one to reflect segment index rather than transition index
        return sequence

    def _compute_num_quilt_segments(self):
        # TODO: treat len as min or max, or enforce consistency somehow?
        # TODO: add a compute_derivate_attributes (those that are computed from user-defined ones)
        # rounds down
        return int(np.ceil(self.len_quilt_samples / self.len_segment_samples))


def join_segments_psola(
        signal, ordered_initial_locations,
        window, max_shift,
        len_segment, len_overlap
        ):

    def overlap_add(segments, len_overlap, window):  # todo: put this function outside
        joined_segments = segments[0] * window
        for idx, chunk in enumerate(segments[1:]):
            pad = np.zeros(chunk.shape[0] - len_overlap)
            joined_segments = np.concatenate([joined_segments, pad])  # zero pad
            joined_segments[-chunk.shape[0]:] += chunk * window
        return joined_segments

    window_indices = np.arange(0, window.shape[0]) - len_overlap//2  # centered at half the first ramp
    first_loc = ordered_initial_locations[0]
    segments = [signal[first_loc + window_indices]] # window indices ensures correct length and centering

    for location in ordered_initial_locations[1:]:
        candidate_region_idx = np.arange(
            location - max_shift - len_overlap//2,
            location + max_shift + len_overlap//2
            )
        candidates = signal[candidate_region_idx]

        # get overlap region of last element in the growing list and window it for measurement
        sequence_tail = segments[-1][-len_overlap:] * np.hanning(len_overlap)
        # get optimal location of candidates to overlap with sequence tail
        optimal_location = find_optimal_location(sequence_tail, candidates)
        # grab chunk from signal based on optimal location and window with extra for overlapping
        choice_candidate = signal[candidate_region_idx[optimal_location] + window_indices]
        segments.append(choice_candidate)

    joined_segments = overlap_add(segments, len_overlap, window)
    return joined_segments


def find_optimal_location(vector, candidates):
    # maximise positive correlation
    # smaller vector is overlap region of the right side of the segment in the list
    # smaller vector is windowed with hann before computing cross correlation function
    # larger vector is the overlap candidates of the left side of the new segment to be added
    # mode should be "valid" (only correlations where vectors completely overlap)
    cross_correlations = np.correlate(vector, candidates, mode="valid")
    best_shift = np.argmax(cross_correlations)  # (positive) counting from tail side of candidates
    best_location = -vector.shape[0] - best_shift
    # transform to positive shift from start of candidates
    best_location = candidates.shape[0] + best_location
    return best_location


def find_sequence_similar_diagonal(distance_matrix, len_sequence):
    """
    Finds a sequence of indices with the closest element transitions to the ones
    appearing in the diagonal of the distance matrix.
    Expect a distance matrix representing the transitions between elements of 2 arrays.
    """
    # TODO: check that distance matrix indices can be interpreted as (right_border_nr, left_border_nr)
    indices = np.arange(0, distance_matrix.shape[0])
    choice = np.random.choice(indices, size=1)[0]  # choose randomly only first time
    index_sequence = [choice]

    for i in range(0, len_sequence-1):  # first item already determined randomly above
        element_nr = index_sequence[-1]
        original_distance = distance_matrix[element_nr, element_nr]

        candidate_distances = distance_matrix[element_nr, :].squeeze() # extract row of candidates
        np.put(candidate_distances, index_sequence, np.nan)  # replace used indices with nans

        # multiple minima are handled by argmin: returns the first one
        choice = np.nanargmin(np.abs(candidate_distances - original_distance))
        index_sequence.append(choice)

    # choice is (index that minimizes distance) + 1 because distance 0 is among 0 and 1
    return np.array(index_sequence)  # TODO: add 1 here to reflect index of segment rather than transition


def make_window(len_sides, len_middle):
    """Builds a window with cosine ramps to the sides and ones in the middle"""
    if len_sides % 2 == 1:
        raise Exception("Length sides should be even number of samples")
    hann = np.hanning(len_sides * 2)
    left_hann = hann[0:len_sides]
    middle = np.ones(len_middle)
    right_hann = hann[len_sides:]
    window = np.concatenate([left_hann, middle, right_hann])
    return window

def split_array(array, len_subarrays):
    """
    Splits an array into segments of desired length, along its last dimension.
    Discards remainder at the end if signal length is not divisible by segment length.
    Returned shape is (num_subarrays, array.shape)
    """

    remainder = array.shape[-1] % len_subarrays
    if remainder > 0:  # TODO: this may be redundant with leaving out overlap to the sides of signal
        warnings.warn(UserWarning(
            f"Signal length ({array.shape[-1]}) is not divisible "
            f"by segment length ({len_subarrays}). "
            f"Trailing samples discarded: {remainder}"
            ))
        array = array[..., 0:-remainder]  # discard excess samples in last dimension
    # calculate number of segments, given length of segments
    num_segments = array.shape[-1] // len_subarrays
    segments = np.split(array, indices_or_sections=num_segments, axis=-1)

    return np.array(segments)


def compute_distances(arrays, arrays_2=None):
    """Computes all pair-wise L2 distances between arrays of any size.
    First dimension indexes array number. The rest are the array shape.
    Returns a 2D distance matrix of shape (num_arrays, num_arrays)
    with num_arrays = arrays.shape[0].
    If arrays_2 is passed, pairwise distances are between the vectors in arrays and arrays_2.
    """
    # TODO: make it work with custom distance metric (a callable passed as argument)

    if arrays_2 is None:
        if arrays.ndim < 2 or (arrays.ndim > 1 and arrays.shape[0] < 2):
            raise ValueError(
                f"Less than 2 arrays to compare. Got shape = {arrays.shape}"
                "Arrays are indexed by the first dimension. Last dimensions index shape of array."
                )
        else:
            arrays_2 = arrays.copy()
    else:
        if not np.all(arrays.shape[1:] == arrays_2.shape[1:]):
            raise ValueError(
                f"Arrays in both sets should be the same shape. "
                f"Got {arrays.shape} and {arrays_2.shape}"
                )

    squared_differences = np.power(arrays[..., np.newaxis] - np.moveaxis(arrays_2, 0, -1), 2)
    dims2average = tuple(range(1, squared_differences.ndim-1))  # only dims of the representation array
    distance_matrix = np.sum(squared_differences, axis=dims2average)
    return distance_matrix


def identity_operation(array):
    return array.copy()

# TODO: function that checks whether desired quilt and segment length are possible values




