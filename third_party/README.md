# Third-party baselines and data

## AGT (external tracker)

AGT is not part of this repository; its outputs are consumed as files.

1. Get AGT: TODO repository URL and commit.
2. Run it on the CocoaMOT detections with the settings used in the paper:
   - CocoaMOT-real: confidence threshold **0.05**, max track age **30** (verified: these outputs reproduce the
     paper's AGT rows bit-for-bit; other runs at conf 0.1–0.5 / age 5–30 were exploratory).
   - CocoaMOT-synthetic: TODO (conf / age of the evaluated run).
3. AGT writes `seq001.txt, seq006.txt, ...`. Rename them to the dataset sequence names and place them where the
   experiment runner expects external trackers:

   ```bash
   # e.g. for experiment "main", dataset "real"
   mkdir -p outputs/main/real/tracks/agt
   for f in agt_out/seq*.txt; do n=$(basename "$f" .txt); n=${n#seq}; cp "$f" "outputs/main/real/tracks/agt/seq_0$n.txt"; done
   ```

`run_experiment.py` stops with an explicit error if an `external_trackers` entry has no result files.

## AppleMOTS

Public dataset, not redistributed here.

1. Download AppleMOTS: TODO URL (and license).
2. Sequences used: testing split `0006, 0007, 0008, 0010, 0011, 0012`.
3. Arrange them as `data/applemot/<seq>/{img1/, gt/gt.txt, det/det.txt, seqinfo.ini}` (MOTChallenge layout,
   see the main README). The `gt.txt` / `det.txt` used in the paper are distributed with our data release:
   TODO confirm how det.txt was produced (detector / weights).
