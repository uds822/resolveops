param(
    [Parameter(Mandatory = $true)][string]$ProjectId,
    [Parameter(Mandatory = $true)][string]$Region,
    [Parameter(Mandatory = $true)][string]$Repository,
    [Parameter(Mandatory = $true)][string]$ServiceAccount,
    [string]$JobName = "resolveops-docling-pilot"
)

$ErrorActionPreference = "Stop"
if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    throw "Install and authenticate the Google Cloud CLI before deploying."
}

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$imageTag = "pilot-" + (Get-Date).ToUniversalTime().ToString("yyyyMMddHHmmss")
$image = "${Region}-docker.pkg.dev/${ProjectId}/${Repository}/resolveops-docling:${imageTag}"

Push-Location $projectRoot
try {
    & gcloud builds submit . --tag $image --region $Region --project $ProjectId
    if ($LASTEXITCODE -ne 0) { throw "Cloud Build failed." }

    & gcloud run jobs deploy $JobName `
        --image $image `
        --region $Region `
        --project $ProjectId `
        --service-account $ServiceAccount `
        --cpu 2 `
        --memory 8Gi `
        --task-timeout 3600s `
        --max-retries 1 `
        --tasks 1 `
        --parallelism 1 `
        --set-env-vars MAX_PAGES=10
    if ($LASTEXITCODE -ne 0) { throw "Cloud Run Job deployment failed." }
}
finally {
    Pop-Location
}

Write-Output "Deployed $JobName with image $image"
Write-Output "Execute it with bounded INPUT_URI, OUTPUT_URI, PAGE_START, and PAGE_END overrides."
