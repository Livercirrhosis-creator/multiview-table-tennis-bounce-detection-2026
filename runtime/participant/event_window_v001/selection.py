"""Window existence gates events; conditional location only chooses a real frame."""
import math

if "." in __package__:
    from ..event_decision_v001.selection import select_events
else:  # The local training entry imports inner participant packages directly.
    from event_decision_v001.selection import select_events


def select_window_events(candidates, windows, presence_probabilities, location_logits,
                         *, minimum_separation_seconds=.08):
    if len(windows) != len(presence_probabilities) or len(candidates) != len(location_logits):
        raise ValueError("Window presence or candidate location count differs")
    locations = [[float(v) for v in row] for row in location_logits]
    if any(len(row) != 2 or not all(math.isfinite(v) for v in row) for row in locations):
        raise ValueError("Expected finite two-class conditional location logits")
    winners = {}
    for window, raw in zip(windows, presence_probabilities):
        p = [float(v) for v in raw]
        if (len(p) != 3 or not all(math.isfinite(v) and 0 <= v <= 1 for v in p)
                or abs(sum(p) - 1) > 1e-5):
            raise ValueError("Expected finite normalized window presence probabilities")
        if p[0] <= max(p[1:]):
            continue
        members = window["member_rows"]
        if not members or any(i < 0 or i >= len(candidates) for i in members):
            raise ValueError("Window members must use this video's local candidate rows")
        chosen = min(members, key=lambda i: (-locations[i][0], candidates[i]["native_frame"]))
        candidate = candidates[chosen]
        reward = math.log(max(p[0], 1e-30)) - math.log(max(p[1], p[2], 1e-30))
        # Presence is independent of how its location mass is distributed.
        # The smallest native window center breaks an exact same-frame tie.
        key = (reward, -int(window["center_frame"]), -int(window["window_index"]))
        previous = winners.get(candidate["native_frame"])
        if previous is None or key > previous["key"]:
            winners[candidate["native_frame"]] = dict(
                candidate=candidate, probabilities=p, key=key,
                window_id=window["window_id"], window_center_frame=window["center_frame"])
    ordered = [winners[frame] for frame in sorted(winners)]
    selected = select_events([w["candidate"] for w in ordered],
                             [w["probabilities"] for w in ordered],
                             minimum_separation_seconds=minimum_separation_seconds)
    for event in selected:
        origin = winners[event["frame_id"]]
        event.update(window_id=origin["window_id"], window_center_frame=origin["window_center_frame"])
    return selected
