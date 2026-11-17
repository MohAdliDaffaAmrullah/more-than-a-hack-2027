# NAMATIM — Ball Possession, More than a Hack 2027

## Approach
A deterministic, rule-based possession model built on the tactical ATD (`starter/possession.py`, numpy only).
1. **Per-frame control.** A team controls the ball when its nearest tracked player is within 2.0 m of the
   ball and at least 0.5 m closer than the nearest opponent. Otherwise the ball is loose.
2. **Hysteresis state machine.** Possession changes only after the other team accumulates 0.4 s (10 frames)
   of control with no control by the current team in between; the change is back-dated to the first of
   those frames. Loose balls and frames where the ball is not tracked keep the last team, following the
   challenge definition ("possession stays with the last team that had it").
3. **Dead ball.** D when the ball is outside the pitch lines for >= 5 frames; inside Smart Tagging set-piece
   windows until the restart kick (first frame with ball speed > 3 m/s); when the ball disappears for > 3 s
   just after being within 2 m of a line; and the published pre-kickoff block of the second half. Once dead,
   play restarts only when the ball is back inside the pitch and moving or controlled (3 frames).
4. **Spell cleaning.** Possessions shorter than 2 s and dead spells shorter than 1 s that sit inside one
   team's possession are merged into it, because the ranking metric scores whole possessions.

Frames outside the graded windows are written as X.

## Assets used
- Tactical ATD (players + ball): the core input.
- Tactical Smart Tagging: only the SET PIECES instances, as restart windows. Team tags were not used.
- Not used: broadcast video/ATD, homography, tactical video.

## Training data
None. No model is trained and no labelled data of any kind is used. All thresholds were fixed a priori
from football reasoning and were not tuned on the public 5-minute ground truth.

## Possession rules
- Loose balls, aerial duels and missing-ball periods: possession stays with the last controlling team.
- Challenges: a brief touch (< 0.4 s of control) by the opponent does not change possession.
- Restarts: after a dead ball, the first team to control the ball inside the pitch starts the new possession.

## Results on the public 5 minutes (sanity check only)
macroF1 0.692, spellF1 0.518, 17 possessions predicted vs 17 true (best baseline: 0.581 / 0.424 / 16).

## Limitations
- The ball is missing on ~41% of graded frames; transitions inside long gaps are detected late.
- Dead-ball detection depends on the ball being tracked out of the lines or on Smart Tagging,
  so throw-ins and fouls without a tag can be missed.
- With more time: ball-trajectory reconstruction through gaps, foul/stoppage detection from collective
  player speed, and using the broadcast ATD to fill tactical ball gaps.

## Reproduction
pip install -r requirements.txt
# unpack the data bundle into ./data (tactical/ is required)
python starter/possession.py --team NAMATIM --out .
