param(
  [string]$Python = ".\.venv312\Scripts\python.exe",
  [string]$ModelName = "Qwen/Qwen2.5-1.5B-Instruct",
  [string]$RunTag = "v_next",
  [double]$NumEpochs = 2.0,
  [double]$LearningRate = 8e-5,
  [int]$MaxLength = 384,
  [switch]$GradientCheckpointing
)

$ErrorActionPreference = "Stop"

$TrainFile = "pipeline/out_one_lab/lab4_task2_part1_5_focus/lab4_task2_part1_5/finetune_small/train_instruct.jsonl"
$ValFile = "pipeline/out_one_lab/lab4_task2_part1_5_focus/lab4_task2_part1_5/finetune_small/val_instruct.jsonl"
$OutDir = "pipeline/out/qwen15b_lab4_task2_part1_5_lora_$RunTag"

if (-not (Test-Path $Python)) {
  throw "Python not found at $Python"
}
if (-not (Test-Path $TrainFile)) {
  throw "Train file not found: $TrainFile"
}
if (-not (Test-Path $ValFile)) {
  throw "Val file not found: $ValFile"
}

Write-Host "== GPU check =="
& $Python -c "import torch; print('cuda:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
if ($LASTEXITCODE -ne 0) {
  throw "GPU check failed"
}

Write-Host "== Train adapter ($RunTag) =="
$trainCmd = @(
  "pipeline/train_lora.py",
  "--model-name", $ModelName,
  "--train-file", $TrainFile,
  "--val-file", $ValFile,
  "--output-dir", $OutDir,
  "--max-length", "$MaxLength",
  "--learning-rate", "$LearningRate",
  "--num-epochs", "$NumEpochs",
  "--train-batch-size", "1",
  "--eval-batch-size", "1",
  "--grad-accum", "16",
  "--logging-steps", "10",
  "--eval-steps", "100",
  "--save-steps", "100",
  "--response-style", "faulttype_diag_fix"
)
if ($GradientCheckpointing) {
  $trainCmd += "--gradient-checkpointing"
}
& $Python @trainCmd
if ($LASTEXITCODE -ne 0) {
  throw "train_lora.py failed"
}

Write-Host "Done."
Write-Host "Adapter: $OutDir"
