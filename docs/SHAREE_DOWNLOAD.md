# Downloading the SHAREE validation data

`src/validate_sharee.py` needs the SHAREE database. **You only need the annotation
and header files, about 42 MB. Do not download the signal files, which are 3.7 GB
and are not used.**

SHAREE is open access on PhysioNet. No credentialing, no data use agreement, no login.

## One command

```bash
mkdir -p data/sharee && cd data/sharee
wget -r -np -nH --cut-dirs=2 -A '*.hea,*.qrs,info.txt' \
     https://archive.physionet.org/pn6/shareedb/
cd ../..
```

Flags: `-r` recursive, `-np` do not ascend to the parent directory, `-nH --cut-dirs=2`
strip the URL path so files land flat, `-A` accept only these extensions.

On Windows without wget, use WSL, or install wget, or use
`curl.exe` in a loop over the record list.

## Check it worked

```bash
ls data/sharee/*.qrs | wc -l     # expect 139
ls data/sharee/info.txt          # must exist
du -sh data/sharee               # expect roughly 40-50 MB, NOT gigabytes
```

If you see `.dat` files, delete them: `rm data/sharee/*.dat`

## Run it

```bash
pip install wfdb
python3 src/validate_sharee.py
```

Takes a minute or two. Writes `results/sharee_validation.json`,
`results/sharee_features.csv` and `results/sharee_validation.png`.

## What it is

Melillo P, Izzo R, Orrico A, et al. *Automatic prediction of cardiovascular and
cerebrovascular events using heart rate variability analysis.* PLoS ONE 2015;10(3):e0118504.

139 hypertensive patients, nominal 24-hour Holter at 128 Hz, 12-month follow-up.
17 patients had a vascular event: 11 myocardial infarction, 3 stroke, 3 syncope.

## What it validates, and what it does not

**Validates:** that the night/day RMSSD ratio, the exact feature CardioTwin's cuff-free
screen depends on, carries real prognostic information in real hypertensive patients,
over and above age, sex and clinic blood pressure.

**Does not validate:** non-dipper detection. SHAREE has no ambulatory blood pressure,
so the dipping phenotype is unobservable in it. That still requires paired ABPM and
wearable recordings, which remains the stated next step.

## If the result comes back null

With only 17 events this analysis is underpowered. A null result is a real finding and
should be reported, not dropped. The script prints a note if p > 0.05. Reporting
"we tested our core assumption on 139 real patients and it did not reach significance
at this sample size" is far stronger than reporting nothing from real data at all.
