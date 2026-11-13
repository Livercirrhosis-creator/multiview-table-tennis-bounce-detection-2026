"""Preserve reliable ch1; otherwise choose one fixed alternative channel.

Peak values are network scores, not calibrated correctness probabilities.
No smoothing, GT, source identity, event score or threshold search is used.
The final native frame without an original ch1 observation is excluded here.
"""
RELIABLE_THRESHOLD = .75


def replace_observations(video, observations, native_rows):
    native = {r["native_frame"]:r for r in native_rows}
    if len(native) != len(native_rows) or len(native) != video["frame_count"]:
        raise ValueError("Expected one three-channel record per native video frame")
    result, changes = [], []
    reliable_kept = unchanged_unreliable = padding_kept = 0
    for row in observations:
        if row["center_is_padding"]:
            result.append(dict(row))
            padding_kept += 1
            continue
        frame = row["native_frame"]
        record = native[frame]
        if abs(record["native_pts_seconds"] - row["native_pts_seconds"]) > 1e-8:
            raise ValueError("Native PTS alignment differs")
        channels = {s["channel"]:s["all_pixel"] for s in record["samples"]}
        previous = row["detection"]
        ch1 = channels.get(1)
        if (previous is None) != (ch1 is None) or (previous is not None and
                any(previous[k] != ch1[k] for k in ("x", "y", "conf"))):
            raise ValueError("Original observation differs from captured same-frame ch1")
        if previous is not None and previous["conf"] >= RELIABLE_THRESHOLD:
            result.append(dict(row))
            reliable_kept += 1
            continue
        choices = [(c, channels[c]) for c in (0, 2) if channels.get(c) is not None
                   and channels[c]["conf"] >= RELIABLE_THRESHOLD]
        if not choices:
            result.append(dict(row))
            unchanged_unreliable += 1
            continue
        chosen, point = max(choices, key=lambda item:(item[1]["conf"], -item[0]))
        replacement = {key:point[key] for key in ("x", "y", "conf")}
        result.append(dict(row, detection=replacement))
        changes.append(dict(native_frame=frame, native_pts_seconds=row["native_pts_seconds"],
            old=previous, new=replacement, chosen_channel=chosen,
            alternative_peaks={str(c):p["conf"] for c,p in choices}))
    expected = list(range(video["frame_count"] - 1))
    if [r["native_frame"] for r in result if not r["center_is_padding"]] != expected:
        raise ValueError("Original ch1 frame support must remain unchanged")
    return result, dict(video_id=video["clip_id"], replaced=len(changes), changes=changes,
        reliable_originals_preserved=reliable_kept, unchanged_unreliable=unchanged_unreliable,
        padding_records_preserved=padding_kept, final_native_frame_excluded=True,
        added_observation_rows=0, selected_sources={str(c):sum(r["chosen_channel"]==c for r in changes) for c in (0,2)})
