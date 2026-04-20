param(
  [string]$Python = ".\.venv312\Scripts\python.exe",
  [string]$ModelName = "Qwen/Qwen2.5-1.5B-Instruct",
  [string]$RunTag = "v_next",
  [int]$TemplateIndex = 0,
  [switch]$PromptOnly,
  [switch]$NonInteractive,
  [string[]]$Measure = @(),
  [string[]]$Delta = @()
)

$ErrorActionPreference = "Stop"

$AdapterDir = "pipeline/out/qwen15b_lab4_task2_part1_5_lora_$RunTag"
$TemplateFile = "pipeline/out_one_lab/lab4_task2_part1_5_focus/lab4_task2_part1_5/finetune_small/train_instruct.jsonl"
$GoldenFile = "pipeline/out_one_lab/lab4_task2_part1_5_focus/lab4_task2_part1_5/golden/golden_measurements.json"
$SaveJson = "pipeline/out/qwen15b_lab4_task2_part1_5_manual_infer_$RunTag.json"

if (-not (Test-Path $Python)) {
  throw "Python not found at $Python"
}
if (-not (Test-Path $AdapterDir)) {
  throw "Adapter directory not found: $AdapterDir"
}
if (-not (Test-Path $TemplateFile)) {
  throw "Template file not found: $TemplateFile"
}
if (-not (Test-Path $GoldenFile)) {
  throw "Golden measurement file not found: $GoldenFile"
}

Write-Host "== Manual measurement inference ($RunTag) =="
$inferCmd = @(
  "pipeline/run_measurement_infer.py",
  "--model-name", $ModelName,
  "--adapter-dir", $AdapterDir,
  "--template-file", $TemplateFile,
  "--template-index", "$TemplateIndex",
  "--golden-file", $GoldenFile,
  "--max-new-tokens", "96",
  "--temperature", "0",
  "--num-beams", "1",
  "--repetition-penalty", "1.0",
  "--no-repeat-ngram-size", "0",
  "--enforce-format",
  "--save-json", $SaveJson
)

foreach ($m in $Measure) {
  $inferCmd += @("--measure", $m)
}
foreach ($d in $Delta) {
  $inferCmd += @("--delta", $d)
}
if ($NonInteractive) {
  $inferCmd += "--non-interactive"
}
if ($PromptOnly) {
  $inferCmd += "--prompt-only"
}

& $Python @inferCmd
if ($LASTEXITCODE -ne 0) {
  throw "run_measurement_infer.py failed"
}

Write-Host "Done."
Write-Host "Saved: $SaveJson"
