param(
  [string]$Python = ".\.venv312\Scripts\python.exe",
  [string]$LtspiceBin = "C:\Users\bchar\AppData\Local\Programs\ADI\LTspice\LTspice.exe",
  [int]$VariantsPerCircuit = 1200,
  [int]$MaxWorkers = 22,
  [int]$Seed = 42,
  [switch]$KeepRaw
)

$ErrorActionPreference = "Stop"

$Lab = "lab4_task2_part1_5"
$AscDir = "LTSpice_files/Lab4"
$OutRoot = "pipeline/out_one_lab/lab4_task2_part1_5_focus"
$LabDir = Join-Path $OutRoot $Lab

if (-not (Test-Path $Python)) {
  throw "Python not found at $Python"
}
if (-not (Test-Path $LtspiceBin)) {
  throw "LTspice not found at $LtspiceBin"
}

Write-Host "== Rebuild pipeline for $Lab =="
$runCmd = @(
  "pipeline_one_lab/run_one_lab_pipeline.py",
  "--lab", $Lab,
  "--asc-dir", $AscDir,
  "--ltspice-bin", $LtspiceBin,
  "--out-root", $OutRoot,
  "--variants-per-circuit", "$VariantsPerCircuit",
  "--seed", "$Seed",
  "--max-workers", "$MaxWorkers",
  "--timeout-sec", "240",
  "--val-ratio", "0.2",
  "--use-golden",
  "--weight-param-drift", "0.30",
  "--weight-missing-component", "0.12",
  "--weight-pin-open", "0.12",
  "--weight-swapped-nodes", "0.18",
  "--weight-short-between-nodes", "0.08",
  "--weight-resistor-value-swap", "0.20",
  "--weight-resistor-wrong-value", "0.15",
  "--vsource-min", "-5",
  "--vsource-max", "5",
  "--param-drift-vsource-prob", "0.45",
  "--no-param-drift-allow-resistor",
  "--measurement-noise-sigma", "0.0",
  "--measurement-noise-prob", "1.0",
  "--max-measurements", "24",
  "--max-deltas", "24",
  "--max-chars", "2200",
  "--measurement-stat-mode", "max_only",
  "--prefer-voltage-keys",
  "--map-resistor-param-drift",
  "--drop-noop-faults",
  "--canonicalize-output",
  "--input-mode", "delta_plus_measured",
  "--output-mode", "faulttype_diag_fix"
)
if ($KeepRaw) {
  $runCmd += "--keep-raw"
}
& $Python @runCmd
if ($LASTEXITCODE -ne 0) {
  throw "run_one_lab_pipeline.py failed"
}

Write-Host "== Rebuild finetune split with strict options =="
$splitCmd = @(
  "pipeline_one_lab/prepare_finetune_one_lab.py",
  "--lab", $Lab,
  "--out-root", $OutRoot,
  "--seed", "$Seed",
  "--val-ratio", "0.2",
  "--use-golden",
  "--max-measurements", "24",
  "--max-deltas", "24",
  "--max-chars", "2200",
  "--measurement-stat-mode", "max_only",
  "--prefer-voltage-keys",
  "--measurement-noise-sigma", "0.0",
  "--measurement-noise-prob", "1.0",
  "--input-mode", "delta_plus_measured",
  "--output-mode", "faulttype_diag_fix",
  "--voltage-only",
  "--ambiguity-policy", "majority",
  "--balance-classes",
  "--canonicalize-output",
  "--include-variant-id",
  "--map-resistor-param-drift",
  "--drop-noop-faults"
)
& $Python @splitCmd
if ($LASTEXITCODE -ne 0) {
  throw "prepare_finetune_one_lab.py failed"
}

Write-Host "== Fault mix summary =="
& $Python pipeline_one_lab/check_fault_mix.py --manifest "$LabDir/variant_manifest.jsonl"
if ($LASTEXITCODE -ne 0) {
  throw "check_fault_mix.py failed"
}

Write-Host "Done."
Write-Host "Lab folder: $LabDir"
Write-Host "Train:      $LabDir/finetune_small/train_instruct.jsonl"
Write-Host "Val:        $LabDir/finetune_small/val_instruct.jsonl"
