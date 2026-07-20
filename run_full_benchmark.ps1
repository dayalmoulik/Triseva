# TriSeva Full Benchmark Automation Script
Write-Host "============================================================" -ForegroundColor Teal
Write-Host "STARTING TRISEVA FULL OPTIMIZED BENCHMARK RUN (500 QUESTIONS)" -ForegroundColor Teal
Write-Host "============================================================" -ForegroundColor Teal

$PYTHON_PATH = "C:\Users\Moulik\anaconda3\envs\triseva\python.exe"

# 1. Run Response Collector
Write-Host "`n[STEP 1/5] Running Response Collector (eval_runner.py)..." -ForegroundColor Yellow
& $PYTHON_PATH evaluation/eval_runner.py --size 500

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Step 1 failed. Aborting." -ForegroundColor Red
    Exit 1
}

# 2. Prepare progress logs
Write-Host "`n[STEP 2/5] Preparing progress logs..." -ForegroundColor Yellow
if (Test-Path "evaluation/results/answered_questions_500.jsonl") {
    Copy-Item "evaluation/results/answered_questions_500.jsonl" "evaluation/results/eval_500_progress.jsonl" -Force
    Write-Host "✅ Prepared eval_500_progress.jsonl successfully." -ForegroundColor Green
} else {
    Write-Host "❌ evaluation/results/answered_questions_500.jsonl not found." -ForegroundColor Red
    Exit 1
}

# 3. Run Advanced Ragas and BERTScore Evaluations
Write-Host "`n[STEP 3/5] Running Advanced Ragas and BERTScore Evaluations..." -ForegroundColor Yellow
& $PYTHON_PATH evaluation/run_advanced_evals.py --provider sarvam

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Step 3 failed. Aborting." -ForegroundColor Red
    Exit 1
}

# 4. Run Custom Hinglish Rubrics and Trajectory Evaluations
Write-Host "`n[STEP 4/5] Running Custom Rubrics & Trajectory Auditing (500 samples)..." -ForegroundColor Yellow
& $PYTHON_PATH evaluation/run_custom_rubrics.py --sample_size 500

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Step 4 failed. Aborting." -ForegroundColor Red
    Exit 1
}

# 5. Regenerate Word/Markdown Deliverables
Write-Host "`n[STEP 5/5] Regenerating Word & Markdown Deliverables..." -ForegroundColor Yellow
& $PYTHON_PATH C:\Users\Moulik\.gemini\antigravity\brain\781ff800-1961-4b2d-8dba-e996c58407cc\scratch\create_docx.py
& $PYTHON_PATH evaluation/create_feedback_docx.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Step 5 failed. Aborting." -ForegroundColor Red
    Exit 1
}

Write-Host "`n============================================================" -ForegroundColor Green
Write-Host "🎉 ALL STEPS COMPLETED SUCCESSFULLY! DELIVERABLES UPDATED!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
