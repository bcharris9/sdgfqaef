param(
  [string]$Python = ".\.venv312\Scripts\python.exe",
  [string]$ModelName = "Qwen/Qwen2.5-1.5B-Instruct",
  [string]$RunTag = "v_next",
  [int]$MaxSamples = 100000,
  [int]$MaxNewTokens = 96
)

$ErrorActionPreference = "Stop"

$AdapterDir = "pipeline/out/qwen15b_lab4_task2_part1_5_lora_$RunTag"
$DataFile = "pipeline/out_one_lab/lab4_task2_part1_5_focus/lab4_task2_part1_5/finetune_small/val_instruct.jsonl"
$OutFile = "pipeline/out/qwen15b_lab4_task2_part1_5_eval_$RunTag.jsonl"
$ReportFile = "pipeline/out/qwen15b_lab4_task2_part1_5_eval_${RunTag}_report.json"

if (-not (Test-Path $Python)) {
  throw "Python not found at $Python"
}
if (-not (Test-Path $AdapterDir)) {
  throw "Adapter directory not found: $AdapterDir"
}
if (-not (Test-Path $DataFile)) {
  throw "Validation data not found: $DataFile"
}

Write-Host "== Evaluate adapter ($RunTag) =="
$evalCmd = @(
  "pipeline/test_lora_model.py",
  "--model-name", $ModelName,
  "--adapter-dir", $AdapterDir,
  "--data-file", $DataFile,
  "--out-file", $OutFile,
  "--report-file", $ReportFile,
  "--max-samples", "$MaxSamples",
  "--max-new-tokens", "$MaxNewTokens",
  "--temperature", "0",
  "--num-beams", "1",
  "--repetition-penalty", "1.0",
  "--no-repeat-ngram-size", "0",
  "--response-style", "faulttype_diag_fix",
  "--enforce-format"
)
& $Python @evalCmd
if ($LASTEXITCODE -ne 0) {
  throw "test_lora_model.py failed"
}

Write-Host "Done."
Write-Host "Predictions: $OutFile"
Write-Host "Report:      $ReportFile"

if (Test-Path $ReportFile) {
  $r = Get-Content $ReportFile | ConvertFrom-Json
  $classPct = "{0:N2}" -f [double]$r.class_match.pct
  $exactPct = "{0:N2}" -f [double]$r.exact_match.pct
  Write-Host "Summary: samples=$($r.samples) class_match=$classPct% exact_match=$exactPct%"
}
