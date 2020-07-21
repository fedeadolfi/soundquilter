import numpy as np
import warnings

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
# TODO: test sorting part by fixing random seed and giving a source signal with ground truth order

# TODO: test_sound_quilter is a smoke test, not a unit test. If I run it alongside
#   all the other tests, it will give 100% coverage because it is running
#   a lot of lines in the module, but it is not testing all of them specifically

# TODO: function that chooses a segment order should be a generic "selector".
#   Then selection based on minimizing the difference in transition distance is
#       just a special case. This could accomodate selecting by, for example,
#       maximizing distance, or other selection criteria.

# TODO: in order to make feature computation generalizable, the border could be handled by
#   the feature transform (lambda x: spectrogram(x[0:border_length], srate)[2],
#   then features are just vectors of any length and distances are based on those.
#   Need to make the distance function accept any kind of custom distance and operate on vectors.
# TODO: if not...need to keep track or standardize what axis of the feature array corresponds to time
# TODO: len border samples applies to the feature representation which may have a different sampling rate
#   the concept of border may even be obsolete in a feature space with time-averaging
# TODO: or...to compute similarity between borders, first average the border over time,
#   and then compute distance using only the magnitude over frequencies vector?
# TODO: better option: change what is submitted to `project to feature space`

# TODO: estimate transition distance probability and sample from it, instead of minimizing difference in transition distance
#   This is saying "transition to a segment that will yield a distance sampled from a probability distribution X)
#   The transition probability matrix is estimating the probability of jumping a certain distance
#   This can be applied to any distance metric or feature kind.
#   E.g., to determine a next segment, draw from a probability distribution
#       over cosine similarity (angle). Choose a segment that moves
#       the angle as close as possible to that drawn from the distribution.

# TODO: you can provide the features or the callable
# TODO: features are being computed per signal chunk. Add ability to compute as a whole?
# TODO: consider adding dynamic range compression

# TODO: add ability to sample segments from signal *with replacement*
#   ...this way one can build quilts that are longer than the source signal
# TODO: add ability to reverse or shuffle within-segment samples
# TODO: add ability to order quilt segments randomly

# TODO: check that a custom transform is available
# TODO: if no custom transform, pass a spectrogram or define something with signal
# TODO: add ability to configure app with units of seconds
# TODO: what if the feature representation is computed with overlapping windows?
#   ...then it has to be computed over the whole signal first? (instead of the chunks)
# TODO: test attribute checks
# TODO: detrend each segment?
# TODO: check initial trimming of the signal is not redundant with buffer for overlap candidates
# TODO: function that checks wheather feature border len is less than the actual feature shape
# TODO: plot trajectories to see if there is a difference between ordering or not

class SoundQuilter():
    def __init__(self):
        # Attributes set by the user
        self.srate = None
        self.sample_with_replacement = False  # TODO: need to implement this functionality

        # self.len_segment_secs = None
        self.len_segment_samples = None

        self.len_overlap_samples = None

        # self.len_quilt_secs = None
        self.len_quilt_samples = None
        self._num_quilt_segments = None

        self.len_feature_border_samples = None  # extent to use for distance calculation
        # self.len_border_secs = None

        # Assigned via register_custom_transform().
        self._feature_transform = None

        # a callable that computes some kind of distance like L2
        self.distance_metric = None  # should work with broadcasting

        # for selecting via maximizing cross-correlation (PSOLA)
        # self.max_shift_secs = None
        self.max_shift_samples = None

        self.signal = None
        self._feature_repr = None

        # Attributes NOT set by the user
        self._window_segment = None

        self._original_signal_segments = None
        self._original_signal_locations = None

        self._original_feature_segments = None
        self._original_feature_locations = None

        self._distance_matrix = None

        self._ordered_segments_indices = None

        self._quilt = None

    def make_quilt(self):
        self.confirm_attributes_available()
        self.check_attribute_consistency()
        self._compute_populate_app_attributes()
        self.postprocess_quilt()
        return self._quilt

    def postprocess_quilt(self):
        trim_len = self.len_overlap_samples // 2
        len_fade = int(trim_len)
        self._quilt = fade_in_out(self._quilt[trim_len:-trim_len], len_fade)

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
        optional_attributes = ["len_feature_border_samples"]  # TODO: add more
        empty_attributes = []
        for attr_key, attr_val in vars(self).items():
            # only check existence of attributes assigned by the user
            if attr_key[0] != "_" and attr_key not in optional_attributes and attr_val is None:
                empty_attributes.append(attr_key)
        if len(empty_attributes) > 0:
            raise Exception(
                f"Some attributes are supposed to be set by the user and are empty\n"
                f"Please assign the following attributes:\n{empty_attributes}"
                )

    def check_attribute_consistency(self):
        # TODO: make these checks as separate functions for each thing being checked?
        error_strings = []
        warning_strings = []
        # Collect errors
        if self.signal.ndim > 1:
            error_strings.append(
                f"Signal should be a vector of shape=(num_samples,). "
                f"Got shape={self.signal.shape}."
                )
        # determine usable length and raise error if fewer than 2 segments fit
        remainder = (self.signal.shape[0] - 2 * self.len_overlap_samples) % self.len_segment_samples
        buffer = remainder + 2 * self.len_overlap_samples
        usable_length_signal = self.signal.shape[0] - buffer
        num_segments_fit = usable_length_signal / self.len_segment_samples
        if num_segments_fit < 2:
            error_strings.append(
                f"Either source signal is too short (allows fewer than 2 segments of chosen length)"
                f" or segment length is too long.\n"
                )
        if self.len_feature_border_samples is not None:
            if self.len_feature_border_samples > self.len_segment_samples//2:
                # TODO: but this should be in the feature representation sampling rate
                error_strings.append(
                    f"Length of border should be less than or equal to half of the segment length. "
                    f"Got border={self.len_feature_border_samples} and segment={self.len_segment_samples}.\n"
                    )
        if (
            (usable_length_signal < self.len_quilt_samples)
            and not self.sample_with_replacement
            ):
            error_strings.append(
                f"Requested quilt length ({self.len_quilt_samples}) is longer than "
                f"usable part of source signal ({usable_length_signal}). "
                f"This is only possible if sampling from source with replacement is enabled.\n"
                )
        if self.max_shift_samples > self.len_segment_samples//2:
            error_strings.append(
                f"Maximum shift used for generating overlap candidates ({self.max_shift_samples}) "
                f"cannot be greater than half of the segment length ({self.len_segment_samples}).\n"
                )
        if self.len_overlap_samples > self.len_segment_samples:
            error_strings.append(
                f"Overlap length ({self.len_overlap_samples}) cannot be greater"
                f"than segment length ({self.len_segment_samples}).\n"
                )
        # Collect warnings
        remainder_quilt_len = self.len_quilt_samples % self.len_segment_samples
        if remainder_quilt_len != 0:
            warning_strings.append(
                f"Length of quilt requested ({self.len_quilt_samples}) is not divisible by "
                f"length of segments ({self.len_segment_samples}). "
                f"Length of quilt will be {self.len_segment_samples - remainder_quilt_len} "
                f"samples longer than requested.\n"
                )

        # Warn and/or raise
        if len(warning_strings) > 0:
            warnings.warn(UserWarning(
                f"Found possibly undesired consequences of the configuration values: \n"
                f"".join(warning_strings)
                ))
        if len(error_strings) > 0:
            raise ValueError(
                f"Found {len(error_strings)} errors or inconsistencies in configuration values. " +
                f"See the following descriptions and adjust configuration accordingly:\n" +
                f"".join(error_strings)
                )

    def _compute_populate_app_attributes(self):
        self._window_segment = self._build_window_segment()
        self._num_quilt_segments = self._compute_num_quilt_segments()

        # TODO: decide how to handle feature computation for whole signal if at all
        # self._feature_repr = self._project_onto_feature_space()

        # make segments and locations leaving out a slice of the signal at the beginning and end
        (self._original_signal_locations,
         self._original_signal_segments) = self._make_locations_segments_with_buffer()
        self._original_feature_segments = self._make_features_from_signal_segments()

        self._distance_matrix = self._build_distance_matrix()
        self._ordered_segments_indices = self._build_index_sequence_similar_distance()
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
        # subselect signal leaving out a buffer at beginning and end
        # leave out first and last chunk to allow left-right shift
        # TODO: this preprocessing should be extracted, and number of segments should take it into account
        signal_middle, indices_middle = self._extract_usable_signal()

        # TODO: above we are discarding the edges, but split array will also discard stuff...
        # for locations, keep only the initial index
        segment_locations = split_array(indices_middle, self.len_segment_samples)[:, 0]
        signal_segments = split_array(signal_middle, self.len_segment_samples)
        return segment_locations, signal_segments

    def _extract_usable_signal(self):
        # leave a buffer at the edges of the signal and return middle
        # can we perfectly tile the signal with segments of the desired length?
        remainder = (self.signal.shape[0] - 2*self.len_overlap_samples) % self.len_segment_samples
        # distribute remainder over left and right edges
        if remainder % 2 == 0:
            edge_buffer_left = edge_buffer_right = self.len_overlap_samples + remainder//2
        else:
            edge_buffer_left = self.len_overlap_samples + remainder//2
            edge_buffer_right = self.len_overlap_samples + remainder//2 + 1

        middle_slice = slice(edge_buffer_left, -edge_buffer_right)
        signal_middle = self.signal[middle_slice]
        indices_middle = np.arange(edge_buffer_left, self.signal.shape[0] - edge_buffer_right)
        return signal_middle, indices_middle

    def _make_features_from_signal_segments(self):
        feature_segments = []
        for segment in self._original_signal_segments:
            features = self._feature_transform(segment)[np.newaxis, ...]
            feature_segments.append(features)
        return np.concatenate(feature_segments, axis=0)  # (num_subarrays, *feature.shape)

    def _build_distance_matrix(self):
        """
        Will compute the distance matrix independent of the input representation shape.
        (segment_nr, segment_nr) diagonal in distance matrix has original transition distance
        for segment `segment_nr`.
        """
        # need to first extract borders from arrays in order to compute distances based on those
        # Right border for last segment is not included because it doesn't have a "next segment"
        #   this works if number of segments is even and also if it is odd

        # TODO: last axis in feature array is assumed to be time. Need to enforce this.
        set_A = slice(0, -1, 1)  # segments from the first to next to last
        set_B = slice(1, self._original_feature_segments.shape[0], 1)  # from second to last

        if self.len_feature_border_samples is not None:
            right_border_idx = slice(
                -self.len_feature_border_samples,
                self._original_feature_segments.shape[-1]
                )
            left_border_idx = slice(0, self.len_feature_border_samples)
            feature_segments_A = self._original_feature_segments[set_A, ..., right_border_idx]
            feature_segments_B = self._original_feature_segments[set_B, ..., left_border_idx]
        else:
            feature_segments_A = self._original_feature_segments[set_A, ...]
            feature_segments_B = self._original_feature_segments[set_B, ...]

        return compute_distances(
            feature_segments_A, feature_segments_B,
            metric=self.distance_metric
            )

    def _project_onto_feature_space(self):
        return self._feature_transform(self.signal)

    def _build_index_sequence_similar_distance(self):
        # makes a sequence of segment indices based on maintaining
        #   a similar distances to original transitions
        num_transitions = self._num_quilt_segments - 1
        sequence = [i + 1 for i in find_sequence_similar_diagonal(
            self._distance_matrix, num_transitions
            )]  # add one to reflect segment index rather than transition index
        sequence.insert(0, sequence[0] - 1)  # add first element
        return np.array(sequence)

    def _compute_num_quilt_segments(self):
        # rounds up (a warning will be raised if it overflows requested segment length)
        # TODO: check if round or ceil is more appropriate
        return int(np.round(self.len_quilt_samples / self.len_segment_samples))


def overlap_add(segments, len_overlap, window):
    joined_segments = segments[0] * window  # init quilt
    for idx, chunk in enumerate(segments[1:]):
        pad = np.zeros(chunk.shape[0] - len_overlap)
        joined_segments = np.concatenate([joined_segments, pad])  # zero pad
        joined_segments[-chunk.shape[0]:] += chunk * window
    return joined_segments


def join_segments_psola(
        signal, ordered_initial_locations,
        window, max_shift,
        len_segment, len_overlap
        ):

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


def find_sequence_similar_diagonal(distance_matrix, num_transitions):
    """
    Finds a sequence of indices with the closest element transitions to the ones
    appearing in the diagonal of the distance matrix.
    Expect a distance matrix representing the transitions between elements of 2 arrays.
    """
    indices = np.arange(0, distance_matrix.shape[0])
    choice = np.random.choice(indices, size=1)[0]  # choose randomly only first time
    index_sequence = [choice]

    for i in range(0, num_transitions - 1):  # -1 because first item already determined randomly above
        element_nr = index_sequence[-1]
        original_distance = distance_matrix[element_nr, element_nr]

        candidate_distances = distance_matrix[element_nr, :].squeeze() # extract row of candidates
        np.put(candidate_distances, index_sequence, np.nan)  # replace used indices with nans

        # multiple minima are handled by argmin: returns the first one
        choice = np.nanargmin(np.abs(candidate_distances - original_distance))
        index_sequence.append(choice)

    # choice is (index that minimizes distance) + 1 because distance 0 is among 0 and 1
    return index_sequence


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


def split_array(array, len_subarrays, strict=True):
    """
    Splits an array into segments of desired length, along its last dimension.
    Discards remainder at the end if signal length is not divisible by segment length.
    Returned shape is (num_subarrays, array.shape)
    """

    remainder = array.shape[-1] % len_subarrays
    if remainder > 0 and strict:
        raise Exception(
            f"Signal length ({array.shape[-1]}) is not divisible "
            f"by segment length ({len_subarrays}). "
            f"Remainder: {remainder}"
            )
    elif not strict:
        array = array[..., 0:-remainder]  # discard excess samples in last dimension
        warnings.warn(UserWarning(
            f"Signal length ({array.shape[-1]}) is not divisible "
            f"by segment length ({len_subarrays}). "
            f"{remainder} samples have been discarded."
            ))
    # calculate number of segments, given length of segments
    num_segments = array.shape[-1] // len_subarrays
    segments = np.split(array, indices_or_sections=num_segments, axis=-1)

    return np.array(segments)


def compute_distances(arrays, arrays_2=None, metric="sqerror"):
    """Computes all pair-wise distances between arrays of any size.
    First dimension indexes array number. The rest are the array shape.
    Returns a 2D distance matrix of shape (num_arrays, num_arrays)
    with num_arrays = arrays.shape[0].
    If arrays_2 is passed, pairwise distances are between the vectors in arrays and arrays_2.
    """
    metrics_options = {
        "euclidean": euclidean_distance_matrix,
        "cosine": cosine_similarity_matrix,
        "sqerror": squared_error_matrix,
        }
    if type(metric) is str:
        if metric not in metrics_options.keys():
            raise NotImplementedError(
                f"Metric option '{metric}' not implemented. "
                f"Available metrics are: {metrics_options.keys()}.\n"
                f"You can also pass a custom metric as a callable."
                )
        else:
            fx = metrics_options[metric]
    elif callable(metric):
        fx = metric
    else:
        raise TypeError(
            f"`metric` should be of type string indicating distance measure "
            f"or a callable implementing a custom distance computation. "
            f"Got type {type(metric)}"
            )

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

    # Flatten feature arrays
    new_shape = (arrays.shape[0], -1)
    arrays, arrays_2 = arrays.reshape(new_shape), arrays_2.reshape(new_shape)

    distance_matrix = fx(arrays, arrays_2)
    return distance_matrix


def fade_in_out(vector, len_fade):
    win = np.hanning(len_fade * 2)
    left_indices = slice(0, len_fade)
    right_indices = slice(-len_fade, vector.shape[0])
    vector[left_indices] *= win[left_indices]
    vector[right_indices] *= win[right_indices]
    return vector


def identity_operation(array):
    return array.copy()


def cosine_similarity_matrix(x, y):
    return np.divide(
        np.dot(x, y.T),
        np.sqrt(np.dot(x, x.T)) * np.sqrt(np.dot(y, y.T))
        )


def euclidean_distance_matrix(x, y):
    return np.sqrt(np.sum((x[..., np.newaxis] - y.T) ** 2, axis=1))


def squared_error_matrix(x, y):
    return np.sum((x[..., np.newaxis] - y.T) ** 2, axis=1)



