"""Final native-time event selection; no GT, source groups or early cooldown."""
from bisect import bisect_right
import math


def select_events(candidates, probabilities, *, minimum_separation_seconds=0.08):
    if len(candidates) != len(probabilities) or minimum_separation_seconds <= 0:
        raise ValueError("Candidate scores or separation disagree")
    eligible = []
    for candidate, raw in zip(candidates, probabilities):
        p = [float(x) for x in raw]
        if len(p) != 3 or not all(math.isfinite(x) and 0 <= x <= 1 for x in p) or abs(sum(p)-1) > 1e-5:
            raise ValueError("Expected finite normalized three-class probabilities")
        if p[0] <= max(p[1:]):
            continue
        reward = math.log(max(p[0], 1e-30)) - math.log(max(p[1], p[2], 1e-30))
        eligible.append((candidate, reward, p))
    eligible.sort(key=lambda v: (v[0]["native_pts_seconds"], v[0]["native_frame"]))
    times = [row[0]["native_pts_seconds"] for row in eligible]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Candidates must be unique native frame/time observations")
    best, predecessors, take = [0.0], [], []
    for i, (candidate, reward, _) in enumerate(eligible):
        predecessor = bisect_right(times, times[i] - minimum_separation_seconds + 1e-12, 0, i)
        gain = best[predecessor] + reward
        chosen = gain > best[-1] + 1e-12
        predecessors.append(predecessor)
        take.append(chosen)
        best.append(gain if chosen else best[-1])
    chosen = []
    i = len(eligible)
    while i:
        if take[i - 1]:
            candidate, reward, p = eligible[i - 1]
            observation = candidate["center_observation"]
            chosen.append(dict(frame_id=candidate["native_frame"], x=observation["x"], y=observation["y"],
                               pts_seconds=candidate["native_pts_seconds"], candidate_id=candidate["candidate_id"],
                               probabilities=p, reward=reward, sources=candidate["sources"]))
            i = predecessors[i - 1]
        else:
            i -= 1
    chosen.reverse()
    return chosen
