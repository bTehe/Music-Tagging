param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

Write-Host "1) Setup check"
& $Python scripts/check_setup.py

Write-Host "2) Build MTAT manifest + split audit"
& $Python scripts/build_mtat_manifest.py `
  --mtat-root MTAT `
  --annotations MTAT/annotations_final.csv `
  --output-dir artifacts/manifests

Write-Host "3) Train MTAT log-mel short-chunk CNN baseline"
& $Python scripts/train_mtat_cnn.py `
  --manifest artifacts/manifests/mtat_top50_manifest.csv `
  --tags artifacts/manifests/mtat_top50_tags.json `
  --output-dir artifacts/runs/mtat_cnn_baseline

Write-Host "4) Evaluate track pooling + edge cases"
& $Python scripts/evaluate_mtat.py `
  --manifest artifacts/manifests/mtat_top50_manifest.csv `
  --tags artifacts/manifests/mtat_top50_tags.json `
  --checkpoint artifacts/runs/mtat_cnn_baseline/best.pt `
  --output-dir artifacts/reports/mtat_eval

Write-Host "5) Raw waveform ablation"
& $Python scripts/train_mtat_waveform.py `
  --manifest artifacts/manifests/mtat_top50_manifest.csv `
  --tags artifacts/manifests/mtat_top50_tags.json `
  --output-dir artifacts/runs/mtat_waveform

Write-Host "6) Pretrained encoder branch"
& $Python scripts/train_mtat_pretrained.py `
  --manifest artifacts/manifests/mtat_top50_manifest.csv `
  --tags artifacts/manifests/mtat_top50_tags.json `
  --backend panns `
  --output-dir artifacts/runs/mtat_pretrained

Write-Host "7) GTZAN transfer"
& $Python scripts/train_transfer_gtzan.py `
  --gtzan-root GTZAN `
  --mtat-checkpoint artifacts/runs/mtat_cnn_baseline/best.pt `
  --output-dir artifacts/runs/gtzan_transfer

Write-Host "Done"
