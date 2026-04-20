$ErrorActionPreference = "Stop"

$PY = ".\.venv312\Scripts\python.exe"
$MODEL = "google/gemma-3-4b-it"
$TRAIN = "pipeline/out_one_lab/lab4_task2_part1_set/merged_finetune/train_instruct.jsonl"
$VAL = "pipeline/out_one_lab/lab4_task2_part1_set/merged_finetune/val_instruct.jsonl"
$SMOKE_OUT = "pipeline/out/gemma3_4b_lab4_task2_part1_lora_smoke"
$FULL_OUT = "pipeline/out/gemma3_4b_lab4_task2_part1_lora"

Write-Host "== GPU check =="
& $PY -c "import torch; print('cuda:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
if ($LASTEXITCODE -ne 0) { throw "PyTorch GPU check failed" }

Write-Host "== Smoke train (fast iteration) =="
& $PY pipeline/train_lora.py `
  --model-name $MODEL `
  --train-file $TRAIN `
  --val-file $VAL `
  --output-dir $SMOKE_OUT `
  --max-length 128 `
  --train-batch-size 1 `
  --eval-batch-size 1 `
  --grad-accum 8 `
  --num-epochs 1 `
  --max-train-samples 512 `
  --max-val-samples 128 `
  --max-steps 40 `
  --logging-steps 1 `
  --eval-steps 20 `
  --save-steps 20 `
  --gradient-checkpointing
if ($LASTEXITCODE -ne 0) { throw "Smoke train failed" }

Write-Host "== Full train =="
& $PY pipeline/train_lora.py `
  --model-name $MODEL `
  --train-file $TRAIN `
  --val-file $VAL `
  --output-dir $FULL_OUT `
  --max-length 256 `
  --train-batch-size 1 `
  --eval-batch-size 1 `
  --grad-accum 16 `
  --num-epochs 1.5 `
  --logging-steps 10 `
  --eval-steps 200 `
  --save-steps 200 `
  --gradient-checkpointing
if ($LASTEXITCODE -ne 0) { throw "Full train failed" }

Write-Host "DONE"
Write-Host "Smoke adapter: $SMOKE_OUT"
Write-Host "Full adapter:  $FULL_OUT"
