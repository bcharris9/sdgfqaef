$ErrorActionPreference = 'Stop'
$ltspice = 'C:\Users\bchar\AppData\Local\Programs\ADI\LTspice\LTspice.exe'
$root = Join-Path (Get-Location) 'LTSpice_files'
$tmp = Join-Path (Get-Location) '_tmp_ltspice_validate'
if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
New-Item -ItemType Directory -Path $tmp | Out-Null
$report = @()
Get-ChildItem $root -Recurse -Filter *.asc | Sort-Object FullName | ForEach-Object {
  $asc = $_.FullName
  $name = $_.BaseName
  $net = Join-Path $tmp ($name + '.net')
  $null = & $ltspice -netlist $asc 2>$null
  $srcNet = [System.IO.Path]::ChangeExtension($asc, '.net')
  $exists = Test-Path $srcNet
  $nonGround = 0
  $status = 'ok'
  if ($exists) {
    Move-Item $srcNet $net -Force
    $lines = Get-Content $net
    $nodes = New-Object System.Collections.Generic.HashSet[string]
    foreach ($line in $lines) {
      $s = $line.Trim()
      if (-not $s -or $s.StartsWith('*') -or $s.StartsWith('.') -or $s.StartsWith(';')) { continue }
      $t = $s -split '\s+'
      if ($t.Length -lt 3) { continue }
      if ($t[1] -ne '0') { [void]$nodes.Add($t[1]) }
      if ($t[2] -ne '0') { [void]$nodes.Add($t[2]) }
    }
    $nonGround = $nodes.Count
    if ($nonGround -eq 0) { $status = 'ground_only' }
  } else {
    $status = 'netlist_export_failed'
  }
  $report += [pscustomobject]@{
    asc = $asc
    stem = $name
    status = $status
    non_ground_nodes = $nonGround
  }
}
$report | ConvertTo-Json -Depth 4 | Set-Content (Join-Path $tmp 'report.json')
$report | Group-Object status | Sort-Object Name | ForEach-Object { '{0}`t{1}' -f $_.Name, $_.Count }
