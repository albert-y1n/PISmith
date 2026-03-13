from scipy.signal import savgol_filter
import numpy as np
from scipy.signal import find_peaks, peak_widths, savgol_filter

def group_consecutive_peaks(peaks, max_gap=20):
    if len(peaks) == 0:
        return []
    groups = []
    current_group = [peaks[0]]
    for i in range(1, len(peaks)):
        if peaks[i] - peaks[i - 1] <= max_gap:
            current_group.append(peaks[i])
        else:
            groups.append(current_group)
            current_group = [peaks[i]]
    groups.append(current_group)
    return groups

def find_indexes_above_threshold(signal, threshold=0.025):
    return [i for i, v in enumerate(signal) if v >= threshold]

def list_intersection(peak_groups, peak_indexes):
    group_labels = []
    for group in peak_groups:
        if set(group) & set(peak_indexes):
            group_labels.append(True)
        else:
            group_labels.append(False)
    return group_labels

def group_peaks(
    x,
    smooth_win=9,
    max_gap=10,
    threshold=0.01,

    prominence=0.0,
    distance=5,
    height=0.005,
    rel_height=0.95,
):  
    if len(x) < smooth_win:
        smooth_x = x
    else:
        smooth_x = list(savgol_filter(x, smooth_win, 2, mode="constant", cval=0.0))

    peaks, _ = find_peaks(smooth_x, prominence=prominence, distance=distance, height=height)

    peak_groups = group_consecutive_peaks(peaks, max_gap=max_gap)
    peak_indexes = find_indexes_above_threshold(smooth_x, threshold)
    
    group_labels = list_intersection(peak_groups, peak_indexes)

    remove_list = []
    top_values = []
    for i in range(len(group_labels)):
        if group_labels[i]:
            group_width = peak_widths(smooth_x, peak_groups[i], rel_height=rel_height)
            left_ips = int(group_width[-2][0])
            right_ips = int(group_width[-1][-1])

            top_value = max([smooth_x[j] for j in peak_groups[i]])
            top_values.append(top_value)
            remove_list.append((left_ips, right_ips))
    
    if len(top_values) > 0:
        top_idx = np.argmax(top_values)
        return_list = [remove_list[top_idx]]
    else:
        return_list = []
    
    return smooth_x, return_list


ALL_PROCESS_FUNCS = {
    "process_peaks": group_peaks,
}