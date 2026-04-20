$ErrorActionPreference = 'Stop'
$Py = (Resolve-Path '.\.venv312\Scripts\python.exe').Path
$LtspiceBin = 'C:\Users\bchar\AppData\Local\Programs\ADI\LTspice\LTspice.exe'
$AscDir = (Resolve-Path '.\LTSpice_files\Lab1').Path
$TrainRoot = (Resolve-Path '.\ltspice_llm_pipeline_vnext_20260303\pipeline\out_all_labs_rich_v1_train').Path
$EvalRoot = (Resolve-Path '.\ltspice_llm_pipeline_vnext_20260303\pipeline\out_all_labs_rich_v3_evalfresh_newseed').Path
$ApiDir = (Resolve-Path '.\circuit_debug_api').Path

$TrainLabs = @(
  @{ Lab='Lab1_1_0'; VariantSeed='1051' },
  @{ Lab='Lab1_2A_2_0'; VariantSeed='6096' },
  @{ Lab='Lab1_2A_5_0'; VariantSeed='11141' }
)
$EvalLabs = @(
  @{ Lab='Lab1_1_0'; VariantSeed='341786' },
  @{ Lab='Lab1_2A_2_0'; VariantSeed='346831' },
  @{ Lab='Lab1_2A_5_0'; VariantSeed='351876' }
)

function Run-Py([string[]]$CmdArgs) {
  Write-Host ("Running: " + ($CmdArgs -join ' '))
  & $Py @CmdArgs
  if ($LASTEXITCODE -ne 0) {
    throw "Command failed with exit code $LASTEXITCODE"
  }
}

function Reset-LabArtifacts([string]$LabDir, [string[]]$DirsToRemove, [string[]]$FilesToRemove) {
  foreach ($name in $DirsToRemove) {
    $p = Join-Path $LabDir $name
    if (Test-Path $p) { Remove-Item $p -Recurse -Force }
  }
  foreach ($name in $FilesToRemove) {
    $p = Join-Path $LabDir $name
    if (Test-Path $p) { Remove-Item $p -Force }
  }
}

function Rebuild-RawLab([string]$OutRoot, [string]$Lab, [string]$VariantSeed, [string]$VariantsPerCircuit) {
  $labDir = Join-Path $OutRoot $Lab
  $inputDir = Join-Path $labDir 'input_asc'
  New-Item -ItemType Directory -Force -Path $inputDir | Out-Null
  Copy-Item (Join-Path $AscDir ($Lab + '.asc')) (Join-Path $inputDir ($Lab + '.asc')) -Force
  Reset-LabArtifacts $labDir @('base_netlists','variants','sim_results','golden') @('sim_manifest.jsonl','training_dataset.jsonl','variant_manifest.jsonl')
  Run-Py @('ltspice_llm_pipeline_vnext_20260303/pipeline/generate_variants.py','--asc-dir',$inputDir,'--out-dir',$labDir,'--variants-per-circuit',$VariantsPerCircuit,'--seed',$VariantSeed,'--max-workers','8','--ltspice-bin',$LtspiceBin,'--weight-param-drift','0.3','--weight-missing-component','0.12','--weight-pin-open','0.12','--weight-swapped-nodes','0.18','--weight-short-between-nodes','0.08','--weight-resistor-value-swap','0.2','--weight-resistor-wrong-value','0.15','--vsource-min','-5.0','--vsource-max','5.0','--param-drift-vsource-prob','0.45','--no-param-drift-allow-resistor')
  Run-Py @('ltspice_llm_pipeline_vnext_20260303/pipeline_one_lab/run_ltspice_batch_one_lab.py','--lab',$Lab,'--out-root',$OutRoot,'--ltspice-bin',$LtspiceBin,'--max-workers','8','--timeout-sec','300')
  Run-Py @('ltspice_llm_pipeline_vnext_20260303/pipeline_one_lab/build_dataset_one_lab.py','--lab',$Lab,'--out-root',$OutRoot)
  Run-Py @('ltspice_llm_pipeline_vnext_20260303/pipeline_one_lab/build_golden_one_lab.py','--lab',$Lab,'--out-root',$OutRoot,'--ltspice-bin',$LtspiceBin,'--timeout-sec','300')
}

function Rebuild-TrainStudentlike([string]$Lab) {
  $labDir = Join-Path $TrainRoot $Lab
  Reset-LabArtifacts $labDir @('finetune_studentlike_cap64_v1') @()
  Run-Py @('ltspice_llm_pipeline_vnext_20260303/pipeline_one_lab/prepare_finetune_one_lab.py','--lab',$Lab,'--out-root',$TrainRoot,'--split-subdir','finetune_studentlike_cap64_v1','--seed','42','--val-ratio','0.2','--use-golden','--max-measurements','64','--max-chars','4000','--max-deltas','64','--measurement-stat-mode','rich','--input-mode','delta_plus_measured','--output-mode','faulttype_diag_fix','--canonicalize-output','--no-include-variant-id','--include-lab-id','--map-resistor-param-drift','--drop-noop-faults','--include-source-currents','--max-abs-current','100.0','--delta-rank-mode','magnitude','--measured-key-mode','match_deltas','--student-like-prompts','--student-train-copies','4','--student-val-copies','1','--student-scenario-mode','exhaustive','--student-min-sigfigs','3','--student-max-sigfigs','7','--student-min-stat-count','1','--student-max-stat-count','5','--student-full-profile-prob','0.08','--student-source-current-prob','0.22','--student-separate-current-stat-subsets','--student-include-no-current-scenario','--student-train-mixed-copies','12','--student-val-mixed-copies','4','--student-node-dropout-prob','0.2','--student-current-group-dropout-prob','0.35','--student-max-node-drop-fraction','0.6','--student-vary-sigfigs-per-value','--student-entry-jitter-sigma','0.003','--student-entry-jitter-prob','0.2','--student-zero-snap-threshold','1e-4','--student-max-prompts-per-source-row','64')
}

function Rebuild-EvalRich([string]$Lab, [string]$SplitSeed) {
  $labDir = Join-Path $EvalRoot $Lab
  Reset-LabArtifacts $labDir @('finetune_rich_v3eval') @()
  Run-Py @('ltspice_llm_pipeline_vnext_20260303/pipeline_one_lab/prepare_finetune_one_lab.py','--lab',$Lab,'--out-root',$EvalRoot,'--split-subdir','finetune_rich_v3eval','--seed',$SplitSeed,'--val-ratio','0.2','--use-golden','--max-measurements','36','--max-chars','3000','--max-deltas','36','--measurement-stat-mode','rich','--input-mode','delta_plus_measured','--output-mode','faulttype_diag_fix','--canonicalize-output','--no-include-variant-id','--include-lab-id','--map-resistor-param-drift','--drop-noop-faults','--no-include-source-currents','--max-abs-current','100.0','--delta-rank-mode','magnitude','--measured-key-mode','match_deltas')
}

Write-Host '=== REBUILD TRAIN ROOT ==='
foreach ($item in $TrainLabs) {
  Write-Host ("=== TRAIN " + $item.Lab + ' ===')
  Rebuild-RawLab $TrainRoot $item.Lab $item.VariantSeed '80'
  Rebuild-TrainStudentlike $item.Lab
}

Write-Host '=== REBUILD EVAL ROOT ==='
foreach ($item in $EvalLabs) {
  Write-Host ("=== EVAL " + $item.Lab + ' ===')
  Rebuild-RawLab $EvalRoot $item.Lab $item.VariantSeed '64'
  Rebuild-EvalRich $item.Lab $item.VariantSeed
}

Write-Host '=== MERGE TRAIN/EVAL SETS ==='
Run-Py @('ltspice_llm_pipeline_vnext_20260303/pipeline_one_lab/merge_finetune_sets.py','--out-root',$TrainRoot,'--dest-dir',(Join-Path $TrainRoot 'merged_finetune_studentlike_cap64_v1'),'--finetune-subdir','finetune_studentlike_cap64_v1','--seed','42')
Run-Py @('ltspice_llm_pipeline_vnext_20260303/pipeline_one_lab/merge_finetune_sets.py','--out-root',$EvalRoot,'--dest-dir',(Join-Path $EvalRoot 'merged_finetune_rich_v3eval'),'--finetune-subdir','finetune_rich_v3eval','--seed','42')
Run-Py @('ltspice_llm_pipeline_vnext_20260303/pipeline/merge_tagged_instruct_sets.py','--root-dir',$EvalRoot,'--source-subdir','finetune_rich_v3eval','--out-dir',(Join-Path $EvalRoot 'merged_finetune_rich_v3eval_tagged'),'--seed','42','--eval-chunks','6')

Write-Host '=== REBUILD TARGET BOOST DATASET ==='
$boostDir = Join-Path $TrainRoot 'merged_finetune_studentlike_cap64_v1_targetboost_v1'
New-Item -ItemType Directory -Force -Path $boostDir | Out-Null
Run-Py @('ltspice_llm_pipeline_vnext_20260303/pipeline/make_gapfix_oversample.py','--in-file',(Join-Path $TrainRoot 'merged_finetune_studentlike_cap64_v1\train_instruct.jsonl'),'--out-file',(Join-Path $boostDir 'train_instruct.jsonl'),'--report-file',(Join-Path $boostDir 'targeted_boost_report.json'),'--boost','short_between_nodes=8','--boost','resistor_value_swap=8','--boost','pin_open=4','--boost','swapped_nodes=3','--boost','resistor_wrong_value=2','--default-multiplier','1')
Copy-Item (Join-Path $TrainRoot 'merged_finetune_studentlike_cap64_v1\val_instruct.jsonl') (Join-Path $boostDir 'val_instruct.jsonl') -Force
Copy-Item (Join-Path $TrainRoot 'merged_finetune_studentlike_cap64_v1\merge_meta.json') (Join-Path $boostDir 'source_merge_meta.json') -Force
@{
  created_at_utc = [DateTime]::UtcNow.ToString('o')
  source_train_dataset = 'ltspice_llm_pipeline_vnext_20260303\\pipeline\\out_all_labs_rich_v1_train\\merged_finetune_studentlike_cap64_v1\\train_instruct.jsonl'
  source_val_dataset = 'ltspice_llm_pipeline_vnext_20260303\\pipeline\\out_all_labs_rich_v1_train\\merged_finetune_studentlike_cap64_v1\\val_instruct.jsonl'
  train_file = 'train_instruct.jsonl'
  val_file = 'val_instruct.jsonl'
  boost_report = 'targeted_boost_report.json'
} | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $boostDir 'dataset_manifest.json') -Encoding UTF8

Write-Host '=== REFRESH PACKAGED API GOLDENS/CATALOG/KNN ==='
$packagedGoldenRoot = Join-Path $ApiDir 'packaged_golden_root'
foreach ($item in $TrainLabs) {
  $lab = $item.Lab
  $dstLab = Join-Path $packagedGoldenRoot $lab
  New-Item -ItemType Directory -Force -Path $dstLab | Out-Null
  $dstGolden = Join-Path $dstLab 'golden'
  if (Test-Path $dstGolden) { Remove-Item $dstGolden -Recurse -Force }
  Copy-Item (Join-Path (Join-Path $TrainRoot $lab) 'golden') $dstGolden -Recurse -Force
}
Run-Py @('circuit_debug_api/build_runtime_assets.py')
Run-Py @('circuit_debug_api/build_hybrid_assets.py','--no-auto-pick-best','--adapter-dir',(Join-Path $ApiDir 'assets_hybrid\adapter'),'--knn-ref-file',(Join-Path $TrainRoot 'merged_finetune_studentlike_cap64_v1\train_instruct.jsonl'),'--catalog-file',(Join-Path $ApiDir 'assets\circuit_catalog.json'))

Write-Host 'REBUILD_COMPLETE'
