"""Possession from tracking geometry: control detection + hysteresis state machine + dead-ball logic.

    python starter/possession.py --team NAMATIM        # writes evaluation/results/NAMATIM.csv

Deterministic, numpy only, no training, no labelled data. All parameters below were fixed a priori
from football reasoning, NOT tuned on the public 5-minute ground truth (see RULES.md / FAQ.md).
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require, source_frames  # noqa: E402

CONTROL_R = 2.0        # m   player-to-ball distance that counts as control
CONTROL_MARGIN = 0.5   # m   nearest player must beat nearest opponent by this much
MAX_INTERP = 25        # fr  interpolate ball gaps up to 1 s
SWITCH_K = 10          # fr  0.4 s of opposing control needed to change possession
START_K = 3            # fr  control frames needed to start a possession after a dead ball
OUT_MARGIN = 0.5       # m   ball beyond a line by this much = out of play
RESTART_SPEED = 3.0    # m/s ball speed that marks a restart kick/throw
MIN_SPELL = 50         # fr  possessions shorter than 2 s are merged away
DEAD_MISSING = 75      # fr  ball missing >3 s right after being near a line -> dead
NEAR_LINE = 2.0        # m   "near a line"
OUT_MIN = 5            # fr  ball must be outside the lines this long to count
MIN_DEAD = 25          # fr  dead spells under 1 s inside one team's possession are noise

HALVES = ((11185, 90651), (110808, 190197))
SECOND_HALF_KICKOFF = 113742
HOME, AWAY, NONE = 0, 1, -1


def read_meta(meta_path):
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    home_id = team_ids[0]
    names = tuple(t.findtext("Name") for t in root.iter("Team"))
    slots = np.array([HOME if p.get("teamId") == home_id else AWAY for p in root.iter("Player")],
                     dtype=np.int8)
    return slots, names


def parse_atd(raw_path, slot_team, n, pitch):
    L, W = pitch
    bx = np.full(n, np.nan)
    by = np.full(n, np.nan)
    px_all = np.full((n, len(slot_team)), np.nan, dtype=np.float32)
    py_all = np.full((n, len(slot_team)), np.nan, dtype=np.float32)
    with open(raw_path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            i = int(parts[0]) - 1
            if i < 0 or i >= n:
                continue
            b = parts[2].split(",")
            if b[0][:1] != "N" and b[0] != "":
                bx[i] = float(b[0]) * L
                by[i] = float(b[1]) * W
            for s, g in enumerate(parts[1].split(";")):
                if s >= px_all.shape[1] or g[:1] == "N" or g == "":
                    continue
                c = g.split(",")
                px_all[i, s] = float(c[0]) * L
                py_all[i, s] = float(c[1]) * W
    return bx, by, px_all, py_all


def interpolate_gaps(x, max_gap):
    x = x.copy()
    idx = np.flatnonzero(~np.isnan(x))
    if len(idx) < 2:
        return x
    gaps = np.diff(idx)
    for k in np.flatnonzero((gaps > 1) & (gaps <= max_gap + 1)):
        a, b = idx[k], idx[k + 1]
        x[a:b + 1] = np.linspace(x[a], x[b], b - a + 1)
    return x


def nearest_by_team(bx, by, px, py, slot_team):
    dx = px - bx[:, None]
    dy = py - by[:, None]
    d = np.sqrt(dx * dx + dy * dy)
    d_home = np.nanmin(np.where(slot_team[None, :] == HOME, d, np.inf), axis=1)
    d_away = np.nanmin(np.where(slot_team[None, :] == AWAY, d, np.inf), axis=1)
    return d_home, d_away


def smart_tag_set_pieces(path, fps):
    out = []
    for inst in ET.parse(path).iter("instance"):
        if (inst.findtext("code") or "").strip().lower() == "set pieces":
            s = int(round(float(inst.findtext("start")) * fps))
            e = int(round(float(inst.findtext("end")) * fps))
            out.append((s, e))
    return out


def labels_team(c):
    return "H" if c == HOME else "A"


def build_labels(bx, by, d_home, d_away, pitch, set_pieces, n, fps):
    L, W = pitch
    present = ~np.isnan(bx)

    vx = np.gradient(np.where(present, bx, 0.0)) * fps
    vy = np.gradient(np.where(present, by, 0.0)) * fps
    speed = np.hypot(vx, vy)
    speed[~present] = np.nan
    speed[np.isnan(np.roll(bx, 1)) | np.isnan(np.roll(bx, -1))] = np.nan
    speed = np.convolve(np.nan_to_num(speed, nan=0.0), np.ones(3) / 3.0, mode="same")

    out_of_play = present & ((bx < -OUT_MARGIN) | (bx > L + OUT_MARGIN) |
                             (by < -OUT_MARGIN) | (by > W + OUT_MARGIN))
    run = np.zeros(n, dtype=int)
    for t in range(1, n):
        run[t] = run[t - 1] + 1 if out_of_play[t] else 0
    keep = np.zeros(n, dtype=bool)
    for t in np.flatnonzero(run >= OUT_MIN):
        keep[t - OUT_MIN + 1:t + 1] = True
    out_of_play = keep
    near_line = present & ((bx < NEAR_LINE) | (bx > L - NEAR_LINE) |
                           (by < NEAR_LINE) | (by > W - NEAR_LINE))
    inside = present & ~out_of_play

    ctrl = np.full(n, NONE, dtype=np.int8)
    ctrl[present & (d_home < CONTROL_R) & (d_home + CONTROL_MARGIN < d_away)] = HOME
    ctrl[present & (d_away < CONTROL_R) & (d_away + CONTROL_MARGIN < d_home)] = AWAY

    forced_dead = out_of_play.copy()
    forced_dead[HALVES[1][0]:SECOND_HALF_KICKOFF] = True
    for s, e in set_pieces:
        s, e = max(s, 0), min(e, n)
        if e <= s:
            continue
        fast = np.flatnonzero(speed[s:e] > RESTART_SPEED)
        kick = s + fast[0] if len(fast) else s + (e - s) // 2
        forced_dead[s:kick] = True
    i = 0
    while i < n:
        if not present[i]:
            j = i
            while j < n and not present[j]:
                j += 1
            if j - i > DEAD_MISSING and i > 0 and near_line[i - 1]:
                forced_dead[i:j] = True
            i = j
        else:
            i += 1

    labels = np.full(n, "X", dtype="<U1")
    for h0, h1 in HALVES:
        state, dead, streak, streak_start, pending = "D", True, 0, h0, None
        for t in range(h0, h1 + 1):
            c = ctrl[t]
            if forced_dead[t]:
                dead, state, streak = True, "D", 0
                labels[t] = "D"
                continue
            if dead:
                restart = inside[t] and (speed[t] > RESTART_SPEED or c != NONE)
                if c != NONE:
                    if streak == 0 or labels_team(c) != pending:
                        pending, streak, streak_start = labels_team(c), 0, t
                    streak += 1
                if restart and streak >= START_K:
                    dead, state = False, pending
                    labels[streak_start:t + 1] = state
                    streak = 0
                    continue
                labels[t] = "D"
                continue
            if c != NONE:
                if labels_team(c) == state:
                    streak = 0
                else:
                    if streak == 0:
                        streak_start = t
                    streak += 1
                    if streak >= SWITCH_K:
                        state = labels_team(c)
                        labels[streak_start:t] = state
                        streak = 0
            labels[t] = state
    return merge_short(labels)


def merge_short(labels):
    lab = labels.copy()
    n = len(lab)
    starts = np.flatnonzero(np.concatenate(([True], lab[1:] != lab[:-1])))
    ends = np.concatenate((starts[1:], [n]))
    for k in range(1, len(starts) - 1):
        s, e = starts[k], ends[k]
        short = (lab[s] in "HA" and e - s < MIN_SPELL) or (lab[s] == "D" and e - s < MIN_DEAD)
        if short:
            prev, nxt = lab[starts[k - 1]], lab[starts[k + 1]]
            if prev == nxt and prev in "HA":
                lab[s:e] = prev
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default="submission")
    ap.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results"))
    args = ap.parse_args()

    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, None)
    tag_path = require(["smart_tagging"], GAME, None)[0]
    n = source_frames("tactical", GAME)
    pitch, fps = GAMES[GAME]["pitch"], GAMES[GAME]["fps"]

    slot_team, names = read_meta(meta_path)
    print("teams: H = {}, A = {}".format(*names))
    print("reading tracking ...")
    bx, by, px, py = parse_atd(raw_path, slot_team, n, pitch)
    print("  ball present on {:.1f}% of frames".format(100 * np.mean(~np.isnan(bx))))
    bx, by = interpolate_gaps(bx, MAX_INTERP), interpolate_gaps(by, MAX_INTERP)
    print("  after <=1 s interpolation: {:.1f}%".format(100 * np.mean(~np.isnan(bx))))
    d_home, d_away = nearest_by_team(bx, by, px, py, slot_team)
    del px, py
    set_pieces = smart_tag_set_pieces(tag_path, fps)
    print("  {} set-piece tags".format(len(set_pieces)))
    labels = build_labels(bx, by, d_home, d_away, pitch, set_pieces, n, fps)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "{}.csv".format(args.team))
    with open(path, "w") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, l) for i, l in enumerate(labels))
    print("wrote {}  {}".format(path, {c: int(np.sum(labels == c)) for c in "HADX"}))


if __name__ == "__main__":
    main()
