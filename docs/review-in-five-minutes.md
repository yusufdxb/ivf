# Review IVF in five minutes

IVF checks whether two simulator experiment subjects satisfy a declared behavioral
acceptance contract. It verifies the comparison controls, runs typed oracles, localizes
divergence, and seals the evidence. It does not decide which backend is physically
correct.

Run these commands from the repository root after `uv sync --frozen`.

## 1. Inspect the contract

```bash
sed -n '1,240p' validation/examples/cartpole_physx_vs_newton_v1.yaml
```

The manifest names the two strict v1 captures, required controls, allowed backend
differences, unverifiable controls, signals, units, and decision thresholds.

## 2. Inspect the recorded verdict

```bash
sed -n '1,220p' artifacts/evidence/cartpole-v1-physx-vs-newton-20260801T050934Z-05005912/verdict.json
sed -n '1,260p' artifacts/evidence/cartpole-v1-physx-vs-newton-20260801T050934Z-05005912/validity.json
```

Look for `FAIL` and the two reason codes in `verdict.json`. In `validity.json`, look for
22 checks, no failed checks, and three explicitly unverifiable controls.

## 3. Verify every evidence file

```bash
uv run ivf reproduce artifacts/evidence/cartpole-v1-physx-vs-newton-20260801T050934Z-05005912 --verify-only
```

Expected: `integrity ok`, 13 files verified, and recorded verdict `FAIL`.

## 4. Open the static report

```bash
uv run ivf report artifacts/evidence/cartpole-v1-physx-vs-newton-20260801T050934Z-05005912 --print-verdict
python3 -m http.server 8000
```

Open
`http://127.0.0.1:8000/artifacts/evidence/cartpole-v1-physx-vs-newton-20260801T050934Z-05005912/report.html`,
then stop the server with Ctrl-C.

## 5. Inspect the first divergence

```bash
sed -n '1,3p' artifacts/evidence/cartpole-v1-physx-vs-newton-20260801T050934Z-05005912/divergence.jsonl
```

The first pole-rate tolerance violation is at step 13. The first pole-angle violation is
at step 29, and the first termination-event disagreement is at step 35.

One explicit limitation: binary asset identity, realized backend initial state, and
backend-internal state are unverifiable from the captured boundary. IVF records that
limit instead of treating those controls as matches.
