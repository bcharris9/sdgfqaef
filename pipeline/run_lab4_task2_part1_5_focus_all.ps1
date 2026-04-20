param(
  [string]$Python = ".\.venv312\Scripts\python.exe",
  [string]$LtspiceBin = "C:\Users\bchar\AppData\Local\Programs\ADI\LTspice\LTspice.exe",
  [string]$ModelName = "Qwen/Qwen2.5-1.5B-Instruct",
  [string]$RunTag = "v_next",
  [int]$VariantsPerCircuit = 1200,
  [int]$MaxWorkers = 22,
  [int]$MaxSamplesEval = 100000,
  [switch]$KeepRaw,
  [switch]$GradientCheckpointing
)

$ErrorActionPreference = "Stop"

Write-Host "== Stage 1/3: rebuild data =="
$rebuildArgs = @(
  "-ExecutionPolicy", "Bypass",
  "-File", ".\pipeline\run_lab4_task2_part1_5_focus_rebuild.ps1",
  "-Python", $Python,
  "-LtspiceBin", $LtspiceBin,
  "-VariantsPerCircuit", "$VariantsPerCircuit",
  "-MaxWorkers", "$MaxWorkers",
  "-Seed", "42"
)
if ($KeepRaw) {
  $rebuildArgs += "-KeepRaw"
}
& powershell @rebuildArgs
if ($LASTEXITCODE -ne 0) {
  throw "Rebuild stage failed"
}

Write-Host "== Stage 2/3: train adapter =="
$trainArgs = @(
  "-ExecutionPolicy", "Bypass",
  "-File", ".\pipeline\run_lab4_task2_part1_5_focus_train.ps1",
  "-Python", $Python,
  "-ModelName", $ModelName,
  "-RunTag", $RunTag,
  "-NumEpochs", "2",
  "-LearningRate", "8e-5",
  "-MaxLength", "384"
)
if ($GradientCheckpointing) {
  $trainArgs += "-GradientCheckpointing"
}
& powershell @trainArgs
if ($LASTEXITCODE -ne 0) {
  throw "Train stage failed"
}

Write-Host "== Stage 3/3: evaluate adapter =="
& powershell -ExecutionPolicy Bypass -File ".\pipeline\run_lab4_task2_part1_5_focus_eval.ps1" `
  -Python $Python `
  -ModelName $ModelName `
  -RunTag $RunTag `
  -MaxSamples $MaxSamplesEval `
  -MaxNewTokens 96
if ($LASTEXITCODE -ne 0) {
  throw "Eval stage failed"
}

Write-Host "Done."
Write-Host "Adapter: pipeline/out/qwen15b_lab4_task2_part1_5_lora_$RunTag"
Write-Host "Report:  pipeline/out/qwen15b_lab4_task2_part1_5_eval_${RunTag}_report.json"
