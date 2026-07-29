import os
import sys
sys.path.insert(0, os.path.abspath("."))

import time
import json
from main import ask

MIDTERM_TEST_SUITE = [
    # ── Healthcare Usecases ───────────────────────────────────────────────────
    {
        "id": "HEALTH-01",
        "domain": "health",
        "category": "Clinical Lab Test (Hemoglobin & Anemia)",
        "query": "My haemoglobin is 10.2 g/dL. Is this normal?",
        "expected_keywords": ["normal", "13.5", "12.0", "low", "anemia"]
    },
    {
        "id": "HEALTH-02",
        "domain": "health",
        "category": "Ayushman Bharat Scheme Rules",
        "query": "Is my family eligible for Ayushman Bharat PM-JAY and what cover is provided?",
        "expected_keywords": ["5 lakh", "family", "secondary", "tertiary"]
    },
    
    # ── Legal & Welfare Usecases ──────────────────────────────────────────────
    {
        "id": "LEGAL-01",
        "domain": "legal",
        "category": "PM-KISAN Landholding Eligibility",
        "query": "Am I eligible for PM Kisan if I own 1.5 hectares of land?",
        "expected_keywords": ["eligible", "2 hectares", "6000"]
    },
    {
        "id": "LEGAL-02",
        "domain": "legal",
        "category": "Nagaland Article 371A & FOCUS Scheme",
        "query": "Nagaland me kheti ki jameen, Jhum kheti aur FOCUS scheme ke under kisaano ko kya chhoot aur help di gayi hai?",
        "expected_keywords": ["article 371a", "jhum", "focus"]
    },
    {
        "id": "LEGAL-03",
        "domain": "legal",
        "category": "MGNREGA Unemployment Allowance",
        "query": "If Gram Panchayat does not provide work within 15 days under MGNREGA, what allowance is paid?",
        "expected_keywords": ["15 days", "unemployment allowance"]
    },
    {
        "id": "LEGAL-04",
        "domain": "legal",
        "category": "Pradhan Mantri Awas Yojana (PMAY)",
        "query": "What is the income limit and interest subsidy for EWS under PMAY?",
        "expected_keywords": ["3.0", "6.5%"]
    },

    # ── Agriculture Usecases ──────────────────────────────────────────────────
    {
        "id": "AGRI-01",
        "domain": "agriculture",
        "category": "Crop Advisory (Wheat Yellowing / Chlorosis)",
        "query": "Meri gehun ki fasal me pila rang ho gaya hai kya upay karein?",
        "expected_keywords": ["urea", "nitrogen", "zinc", "yellow rust", "kvk"]
    },
    {
        "id": "AGRI-02",
        "domain": "agriculture",
        "category": "PM Fasal Bima Yojana (Crop Loss Claim)",
        "query": "What is the claim intimation time limit for crop loss under PM Fasal Bima Yojana?",
        "expected_keywords": ["72 hours", "pmfby"]
    },
    {
        "id": "AGRI-03",
        "domain": "agriculture",
        "category": "PM-KUSUM Solar Pump Subsidy",
        "query": "What is the subsidy percentage available for solar agriculture pumps under PM KUSUM?",
        "expected_keywords": ["60%", "subsidy"]
    },
    {
        "id": "AGRI-04",
        "domain": "agriculture",
        "category": "Soil Health Card NPK Balance",
        "query": "What is the recommended NPK ratio for cereal crops under Soil Health Card guidelines?",
        "expected_keywords": ["4:2:1", "npk"]
    }
]

def run_midterm_benchmark():
    print("=" * 80)
    print("TRISEVA — MIDTERM REPORT COMPREHENSIVE USECASE EVALUATION BENCHMARK")
    print("=" * 80)

    results_summary = []
    
    for test in MIDTERM_TEST_SUITE:
        print(f"\n[{test['id']}] Category: {test['category']}")
        print(f"Query: '{test['query']}'")
        print("-" * 60)
        
        start_t = time.time()
        res = ask(test["query"])
        latency = round(time.time() - start_t, 2)
        
        pred_domain = res.get("domain")
        answer_text = res.get("answer") or ""
        score = res.get("score") or 0.0
        disclaimer = res.get("disclaimer") or ""
        sources = res.get("sources") or []
        
        domain_match = (pred_domain == test["domain"])
        kw_found = [kw for kw in test["expected_keywords"] if kw.lower() in answer_text.lower()]
        kw_pass = len(kw_found) > 0
        
        status = "PASSED 🏆" if (domain_match and (score >= 0.7 or kw_pass)) else "FAILED ❌"
        
        print(f"Predicted Domain: {pred_domain} (Expected: {test['domain']}) -> Match: {domain_match}")
        print(f"Dual-Judge Score: {score}")
        print(f"Latency: {latency}s")
        print(f"Keywords Matched: {kw_found} / {test['expected_keywords']}")
        print(f"Status: {status}")
        print(f"Answer Snippet:\n{answer_text[:250]}...")
        
        results_summary.append({
            "id": test["id"],
            "category": test["category"],
            "domain": test["domain"],
            "pred_domain": pred_domain,
            "score": score,
            "latency": latency,
            "status": status,
            "answer": answer_text,
            "disclaimer": disclaimer,
            "sources": sources
        })

    print("\n" + "=" * 80)
    print("MIDTERM BENCHMARK SUMMARY TABLE")
    print("=" * 80)
    print(f"{'ID':<10} | {'Domain':<12} | {'Score':<6} | {'Latency':<8} | {'Status':<10}")
    print("-" * 55)
    for r in results_summary:
        print(f"{r['id']:<10} | {r['pred_domain']:<12} | {r['score']:<6} | {r['latency']:<7}s | {r['status']:<10}")

if __name__ == "__main__":
    run_midterm_benchmark()
