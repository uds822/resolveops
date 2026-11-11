# Docling PDF pilot on Google Cloud

This deploys the **parser only** as a CPU Cloud Run Job. Each execution reads a
PDF from Cloud Storage, parses at most 10 selected pages, and writes one JSON
object containing the canonical `Document`, validation result, and timing and
memory metrics. It does not create chunks or index content.

## Prerequisites

1. Install and sign in to the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install).
   Choose a billed project and a region. Enable Cloud Run, Cloud Build,
   Artifact Registry, and Cloud Storage APIs in that project.
2. Create a Docker Artifact Registry repository in the chosen region. Create a
   dedicated service account for the job. Grant it `roles/storage.objectViewer`
   on the input bucket and `roles/storage.objectCreator` on the output bucket.
   Grant the account running the deployment permission to build images, deploy
   Cloud Run Jobs, use the job service account, and push to the repository.
3. Upload a sample PDF to the input bucket. Keep the original PDF and output
   object names stable for each run. Use a new output name for a new parser
   version because the worker will not overwrite an existing output.

## Build and deploy

Run from PowerShell at the repository root after all local parsing checks pass:

```powershell
./scripts/deploy_pdf_job.ps1 `
  -ProjectId "YOUR_PROJECT_ID" `
  -Region "YOUR_REGION" `
  -Repository "YOUR_DOCKER_REPOSITORY" `
  -ServiceAccount "YOUR_WORKER_ACCOUNT@YOUR_PROJECT_ID.iam.gserviceaccount.com"
```

The script builds the image with Cloud Build and deploys one Cloud Run Job with
2 vCPU, 8 GiB memory, a one hour task timeout, one task, one parallel worker,
and one retry. It does not start a run. The source upload excludes local PDFs,
virtual environments, tests, and generated files through `.gcloudignore`.

## Execute one page

The initial smoke sample is DynamoDB PDF page 58. Upload the PDF, then execute:

```powershell
gcloud storage cp data/raw/aws/pdf/dynamodb-dg.pdf gs://YOUR_INPUT_BUCKET/dynamodb-dg.pdf

gcloud run jobs execute resolveops-docling-pilot `
  --project YOUR_PROJECT_ID `
  --region YOUR_REGION `
  --update-env-vars "INPUT_URI=gs://YOUR_INPUT_BUCKET/dynamodb-dg.pdf,OUTPUT_URI=gs://YOUR_OUTPUT_BUCKET/parsed/dynamodb-p58-v1.json,PAGE_START=58,PAGE_END=58" `
  --wait

gcloud storage cp gs://YOUR_OUTPUT_BUCKET/parsed/dynamodb-p58-v1.json ./dynamodb-p58-v1.json
```

Inspect `validation.state`, `validation.review_reasons`, table cells and page
numbers in `document.blocks`, plus `metrics.elapsed_seconds` and
`metrics.peak_rss_mib`. The job exits successfully when extraction produces a
review flag so that the flagged JSON is retained for inspection. A conversion
exception fails the task. Cloud Run and Cloud Build charges apply when you run
the build and job; use the measured execution time before scaling the sample.

The worker parses selected digital PDF pages with OCR disabled. Scanned PDFs
will need a separately measured OCR setting. A bounded page range limits model
work but the worker still downloads the whole source PDF into temporary disk.

References: [Cloud Run Jobs](https://docs.cloud.google.com/run/docs/create-jobs),
[job execution overrides](https://docs.cloud.google.com/sdk/gcloud/reference/run/jobs/execute),
and [Docling model prefetching](https://docling-project.github.io/docling/usage/advanced_options/).
