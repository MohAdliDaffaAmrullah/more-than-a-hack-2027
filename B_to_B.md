# Ball Possession Labelling: Technical Document

**More Than A Hack 2027 – Initial Challenge**

**Team:** B to B (Bandung to Barcelona)

**Members:**

- Muhammad Zeva Radityaputra
- Muhammad Mahar Yogaswara
- Moh. Adli Daffa Amrullah

**Code:** https://github.com/MohAdliDaffaAmrullah/more-than-a-hack-2027 (`starter/possession.py`)

---

## Challenge 1 Summary

The task is to label every video frame of one full LaLiga match (199,319 frames at 25 fps) with one of four labels: H (home team in possession, ball alive), A (away team in possession, ball alive), D (dead ball) or X (no answer). The output is a CSV with the header `frame,label` and one row per frame. Entries are ranked on spellF1, which measures how many of the match's real possessions are recovered as whole possessions. macroF1 is reported beside it and is the first tie-break. The repository baseline to beat scores a spellF1 of 0.2433.

The provided tracking, Smart Tagging and homography are automatic and uncorrected. They contain gaps, fragmented tracks and identity switches, and the ball is missing on 41% of graded frames.

## Approach

Our system is a deterministic, rule-based possession model built on the tactical automatic tracking data. It runs in four stages:

1. **Per-frame control.** For every frame the distance from the ball to the nearest tracked player of each team is computed in metres (normalised coordinates scaled to the 105.3 × 68.0 m pitch). Ball gaps of up to 1 s are filled by linear interpolation. A team *controls* the ball when its nearest player is within 2.0 m of the ball and at least 0.5 m closer than the nearest opponent. Otherwise the ball is loose.
2. **Dead-ball detection.** Frames are forced to D when the ball is out of play, inside a set-piece restart window from Smart Tagging, or in the published pre-kickoff block of the second half (details below).
3. **Hysteresis state machine.** Live play is labelled by a state machine that holds the current possessing team. Possession changes only after the other team accumulates 10 frames (0.4 s) of control with no control by the current team in between. The change is back-dated to the first of those frames, so the transition lands where the new team actually gained the ball rather than 0.4 s later.
4. **Spell cleaning.** Possessions shorter than 2 s and dead spells shorter than 1 s that sit inside one team's possession are merged back into that possession.

Reference point: the repository baseline assigns each frame to the team of the tracked player nearest to the ball, then takes a majority vote over a window. On the public 5 minutes, widening the window raises spellF1 from 0.0378 (raw) to 0.4242 (baseline_smooth125), while macroF1 stays between 0.58 and 0.62.

**Why this approach.** The baseline sweep shows that the ranking metric is decided by possession *structure*, not by per-frame accuracy: a wider window removes flicker but also blurs every transition and still treats every missing-ball frame as dead. A majority vote has no notion of "the last team that had the ball" and no notion of a stoppage. A state machine encodes both directly. It keeps possession through loose balls and tracking gaps, as the challenge definition requires, and it changes state only on sustained evidence. That fixes fragmentation without sacrificing transition timing. We chose a transparent rule-based model over a learned one because no labelled data from the provided game may be used for training or validation, and every rule can be explained and audited.

**Results on the public 5 minutes** (a pipeline sanity check only; 17 possessions are too few to rank systems):

| submission | macroF1 | spellF1 | possessions (true: 17) | home-share error |
|---|---|---|---|---|
| **ours (possession_v1)** | **0.6920** | **0.5176** | **17** | **0.013** |
| baseline_smooth125 | 0.5812 | 0.4242 | 16 | 0.052 |
| baseline_smooth50 | 0.6207 | 0.3256 | 26 | 0.048 |
| baseline_raw | 0.5972 | 0.0378 | 131 | 0.046 |

## Assets used

- **Automatic tracking with the ball:** used. The tactical ATD is the core input: ball position and every player's position with team membership. Player identities and track slots are not used, only team membership, so identity switches within a team do not affect the result. Ball speed is derived from the ball positions.
- **Smart Tagging:** used, partially. Only `SET PIECES` instances are used, as restart windows that mark the ball dead until the restart kick. Team tags and phase codes were not used, because phases are not possessions and every throw-in carries the team `N/A`.
- **Homography:** not used. All reasoning is done in pitch coordinates, which the ATD already provides.
- **Tactical and broadcast video:** not used. The broadcast ATD was also not used.
- **Public 5-minute ground truth:** used only to check that the pipeline runs and produces a sensible number of possessions. No parameter was tuned on it. All thresholds were fixed beforehand from football reasoning.
- **Tried and dropped:** the baseline approach of nearest player plus majority-vote smoothing. It was the starting point and was replaced by the state machine for the reasons given above.

**Training data:** none. The method has no trained component and uses no labelled data of any kind.

## Possession rules implemented

Every graded frame is H, A or D. A possession is a maximal run of one team with the ball alive, so H H D D H H counts as two possessions with a stoppage between them. A dead ball is a stoppage and never a possession. Possession changes only when a player of the other team has the ball under control. A single touch without control, a deflected pass that is not intercepted, and an aerial duel won without a controlled ball are not changes of possession. In loose-ball phases, the team that last had possession keeps it.

- **Definition of control:** the team's nearest player is within 2.0 m of the ball and at least 0.5 m closer than the nearest opponent. A change of possession needs 10 frames (0.4 s) of control by the other team with no control by the current team in between.
- **Loose balls and challenges:** frames where neither team controls the ball, or where the ball is not tracked, keep the last possessing team. An opponent touch shorter than 0.4 s (a block, deflection or contested header) does not change possession. If the current team regains control during the challenge, the opponent's count is reset.
- **Restarts:** kickoffs, corners, goal kicks and free kicks are taken from the Smart Tagging `SET PIECES` windows. The ball is dead from the start of the window until the first frame where the ball moves faster than 3 m/s (the restart kick). If no such frame exists, the first half of the window is marked dead. After any dead ball, a new possession starts when a team controls the ball for 3 frames inside the pitch while the ball is moving or controlled. That team is assigned the possession from the first of those frames. The team tag is not used to assign the restart, because it is often missing (`N/A`).
- **Dead ball detection:** D is assigned when (a) the ball is tracked more than 0.5 m outside a pitch line for at least 5 consecutive frames; (b) inside a set-piece window before the restart kick; (c) the ball disappears for more than 3 s immediately after being within 2 m of a line; (d) frames 110,808–113,741, the graded but pre-kickoff block of the second half published in the FAQ. Once dead, the ball stays dead until a restart is observed.
- **Use of X:** the system writes X only outside the two graded windows (before kickoff, halftime and after the final whistle), where it costs nothing. On graded frames it never abstains. Where the ball is missing inside a possession it repeats the current team, which is what the scorer would fill an X with anyway.

## Limitations

- **Missing ball.** The ball is absent on about 41% of graded frames, and only gaps up to 1 s are interpolated. A change of possession that happens inside a long gap is only detected when the ball reappears, so the transition is late or, if the ball reappears after a stoppage, missed.
- **Dead balls without evidence.** Fouls, offsides and throw-ins that are neither tagged nor show the ball clearly outside the lines can be labelled as live play. Conversely, a ball tracked just outside a line (for example an aerial ball near the touchline) can create a false stoppage.
- **Control versus rebound.** Proximity cannot fully separate a controlled reception from a rebound off a player who happens to be near the ball. The 0.4 s persistence filter removes most rebounds but can also delay a real interception.
- **Identity and duplicate tracks.** The model only uses team membership, but a duplicated or misassigned track near the ball can still create false control for the wrong team.
- **With more time:** reconstruct the ball trajectory through gaps using player motion (the team whose players converge on the ball's last position); detect stoppages from collective player behaviour (sudden drop in players' speeds, players walking back into shape); use the broadcast ATD to fill tactical ball gaps; and estimate control from ball–player relative velocity instead of distance alone.

## Reproduction

```bash
git clone https://github.com/MohAdliDaffaAmrullah/more-than-a-hack-2027.git
cd more-than-a-hack-2027
pip install -r requirements.txt            # numpy and pandas only
# unpack the data bundle into ./data (data/tactical/ is required)
python starter/possession.py --team B_to_B --out .
python evaluation/validate_submission.py B_to_B.csv
```

The script is deterministic and produces the same file on every run (about 1–2 minutes).
