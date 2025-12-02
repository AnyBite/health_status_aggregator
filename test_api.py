"""
Comprehensive API Test Script for Health Status Aggregator
Run this script to test all endpoints and workflows.

Usage: python test_api.py
"""

import requests
import json
from typing import Any

BASE_URL = "http://127.0.0.1:8000"

def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def print_result(name: str, success: bool, response: Any = None) -> None:
    """Print test result."""
    status = "✓ PASS" if success else "✗ FAIL"
    print(f"\n{status} - {name}")
    if response:
        if isinstance(response, dict):
            print(json.dumps(response, indent=2, default=str)[:500])
        else:
            print(str(response)[:500])

def test_health_check() -> bool:
    """Test 1: Health check endpoint."""
    try:
        response = requests.get(f"{BASE_URL}/")
        success = response.status_code == 200
        print_result("Health Check", success, response.json())
        return success
    except Exception as e:
        print_result("Health Check", False, str(e))
        return False

def test_single_record_analysis() -> dict | None:
    """Test 2: Analyze a single health record."""
    # Using correct schema: record_id (int), account_id, project_id, predefined_health, free_text_details
    test_cases = [
        {
            "name": "Critical Case (major issues)",
            "data": {
                "record_id": 9001,
                "account_id": "ACC001",
                "project_id": "PRJ001",
                "predefined_health": "Critical",
                "free_text_details": "Major budget overrun. Timeline slipped significantly. Project at risk of cancellation. Critical resource resignation, no backup."
            }
        },
        {
            "name": "Good Case (healthy project)",
            "data": {
                "record_id": 9002,
                "account_id": "ACC002",
                "project_id": "PRJ002",
                "predefined_health": "Good",
                "free_text_details": "On schedule and under budget. Everything is progressing well. Team morale is high, deliverables exceeding expectations."
            }
        },
        {
            "name": "Warning Case (minor issues)",
            "data": {
                "record_id": 9003,
                "account_id": "ACC003",
                "project_id": "PRJ003",
                "predefined_health": "Warning",
                "free_text_details": "Slight delays due to vendor issues. Budget is close to the limit. Minor scope creep detected, monitoring closely."
            }
        },
        {
            "name": "Ambiguous Case (mixed signals - should trigger RAG)",
            "data": {
                "record_id": 9004,
                "account_id": "ACC004",
                "project_id": "PRJ004",
                "predefined_health": "Good",
                "free_text_details": "Delayed vendor input despite client praise. Mostly healthy but timeline risk. Excellent feedback from client, internal issues though."
            }
        }
    ]
    
    results = []
    for test in test_cases:
        try:
            print(f"\n  Testing: {test['name']}")
            response = requests.post(f"{BASE_URL}/analyze/", json=test["data"])
            success = response.status_code == 200
            result = response.json() if success else response.text
            print_result(test["name"], success, result)
            if success:
                results.append(result)
        except Exception as e:
            print_result(test["name"], False, str(e))
    
    return results[0] if results else None

def test_batch_analysis() -> list | None:
    """Test 3: Batch analyze multiple records."""
    batch_data = {
        "records": [
            {
                "record_id": 9101,
                "account_id": "ACC010",
                "project_id": "PRJ010",
                "predefined_health": "Critical",
                "free_text_details": "Budget exceeded by 40%, no additional funding. Key deliverable failed acceptance testing."
            },
            {
                "record_id": 9102,
                "account_id": "ACC010",
                "project_id": "PRJ011",
                "predefined_health": "Good",
                "free_text_details": "All milestones achieved on time. Client satisfaction rating: 9/10. Resources optimally allocated."
            },
            {
                "record_id": 9103,
                "account_id": "ACC010",
                "project_id": "PRJ012",
                "predefined_health": "Warning",
                "free_text_details": "Third-party dependency delayed by 2 weeks. Budget variance at 10%, corrective action planned."
            }
        ]
    }
    
    try:
        response = requests.post(f"{BASE_URL}/analyze/batch/", json=batch_data)
        success = response.status_code == 200
        result = response.json() if success else response.text
        print_result("Batch Analysis (3 records)", success, result)
        return result.get("results", []) if success else None
    except Exception as e:
        print_result("Batch Analysis", False, str(e))
        return None

def test_get_pending_reviews() -> list | None:
    """Test 4: Get pending reviews (low confidence cases)."""
    try:
        response = requests.get(f"{BASE_URL}/reviews/pending/")
        success = response.status_code == 200
        result = response.json() if success else response.text
        print_result("Get Pending Reviews", success, result)
        return result.get("reviews", []) if success else None
    except Exception as e:
        print_result("Get Pending Reviews", False, str(e))
        return None

def test_get_review_details(review_id: int) -> dict | None:
    """Test 5: Get details of a specific review."""
    try:
        response = requests.get(f"{BASE_URL}/reviews/{review_id}/")
        success = response.status_code == 200
        result = response.json() if success else response.text
        print_result(f"Get Review Details (ID: {review_id})", success, result)
        return result if success else None
    except Exception as e:
        print_result("Get Review Details", False, str(e))
        return None

def test_submit_feedback(record_id: int) -> bool:
    """Test 6: Submit human feedback for a record."""
    # Using correct schema: record_id (int), human_decision, reviewer_notes
    feedback_cases = [
        {
            "name": "Correction feedback",
            "data": {
                "record_id": record_id,
                "human_decision": "Critical",
                "reviewer_notes": "LLM underestimated severity. Project history indicates higher risk."
            }
        },
        {
            "name": "Confirmation feedback",
            "data": {
                "record_id": record_id + 1,
                "human_decision": "Warning",
                "reviewer_notes": "Initial assessment was correct. Confirmed after manual review."
            }
        }
    ]
    
    success_count = 0
    for case in feedback_cases:
        try:
            print(f"\n  Testing: {case['name']}")
            response = requests.post(f"{BASE_URL}/feedback/", json=case["data"])
            success = response.status_code == 200
            result = response.json() if success else response.text
            print_result(case["name"], success, result)
            if success:
                success_count += 1
        except Exception as e:
            print_result(case["name"], False, str(e))
    
    return success_count > 0

def test_data_validation() -> dict | None:
    """Test 7: Validate and load dataset."""
    try:
        # Use query parameter instead of JSON body
        response = requests.post(
            f"{BASE_URL}/data/validate/",
            params={"file_path": "app/data/health_dataset.json"}
        )
        success = response.status_code == 200
        result = response.json() if success else response.text
        print_result("Data Validation", success, result)
        return result if success else None
    except Exception as e:
        print_result("Data Validation", False, str(e))
        return None

def test_aggregation_by_account() -> dict | None:
    """Test 8: Get aggregated health metrics by account."""
    try:
        response = requests.get(f"{BASE_URL}/aggregations/by-account/")
        success = response.status_code == 200
        result = response.json() if success else response.text
        print_result("Aggregation by Account", success, result)
        return result if success else None
    except Exception as e:
        print_result("Aggregation by Account", False, str(e))
        return None

def test_generate_report() -> dict | None:
    """Test 9: Generate comprehensive report."""
    report_configs = [
        {
            "name": "Summary Report",
            "data": {
                "report_type": "summary",
                "include_metrics": True,
                "include_recommendations": True,
                "account_filter": None
            }
        },
        {
            "name": "Account-specific Report",
            "data": {
                "report_type": "detailed",
                "include_metrics": True,
                "include_recommendations": True,
                "account_filter": "ACC001"
            }
        }
    ]
    
    result = None
    for config in report_configs:
        try:
            print(f"\n  Testing: {config['name']}")
            response = requests.post(f"{BASE_URL}/reports/generate/", json=config["data"])
            success = response.status_code == 200
            resp_data = response.json() if success else response.text
            print_result(config["name"], success, resp_data)
            if success:
                result = resp_data
        except Exception as e:
            print_result(config["name"], False, str(e))
    
    return result

def run_full_workflow_test():
    """Run a complete workflow simulating real usage."""
    print_header("FULL WORKFLOW TEST")
    print("\nThis test simulates the complete workflow:")
    print("1. Analyze records → 2. Review low-confidence → 3. Submit feedback → 4. Generate report")
    
    # Step 1: Analyze a challenging record (ambiguous - should trigger RAG/review)
    print("\n--- Step 1: Analyze a challenging record ---")
    challenging_record = {
        "record_id": 9999,
        "account_id": "ACC099",
        "project_id": "PRJ099",
        "predefined_health": "Good",  # Predefined says Good
        "free_text_details": "Project financials good, schedule slipping. Client happy but scope significantly expanded. Meeting deadlines but at cost of team burnout."  # But text suggests issues
    }
    
    try:
        response = requests.post(f"{BASE_URL}/analyze/", json=challenging_record)
        if response.status_code == 200:
            analysis_result = response.json()
            print(f"Analysis Result:")
            print(f"  - Recommended: {analysis_result['result']['recommended_health']}")
            print(f"  - Confidence: {analysis_result['result']['confidence']:.2f}")
            print(f"  - Mismatched: {analysis_result['result']['mismatched']}")
            print(f"  - Requires Human Review: {analysis_result['result']['requires_human_review']}")
            print(f"  - Review Created: {analysis_result['review_created']}")
        else:
            print(f"Error: {response.text}")
            analysis_result = None
    except Exception as e:
        print(f"Error: {e}")
        analysis_result = None
    
    # Step 2: Check pending reviews
    print("\n--- Step 2: Check for pending reviews ---")
    try:
        response = requests.get(f"{BASE_URL}/reviews/pending/")
        if response.status_code == 200:
            pending = response.json()
            print(f"Pending Reviews Count: {pending.get('count', 0)}")
            if pending.get('reviews'):
                for review in pending['reviews'][:3]:  # Show first 3
                    print(f"  - Record {review.get('record_id')}: {review.get('reason_for_review', 'N/A')[:50]}...")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Step 3: Submit feedback
    print("\n--- Step 3: Submit human feedback ---")
    feedback_data = {
        "record_id": 9999,
        "human_decision": "Warning",
        "reviewer_notes": "Given the scope expansion and team burnout mentioned, this should be flagged as Warning despite good financials."
    }
    
    try:
        response = requests.post(f"{BASE_URL}/feedback/", json=feedback_data)
        success = response.status_code == 200
        print(f"Feedback Submitted: {success}")
        if success:
            print(f"  Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Step 4: Generate final report
    print("\n--- Step 4: Generate report ---")
    report_request = {
        "report_type": "summary",
        "include_metrics": True,
        "include_recommendations": True
    }
    
    try:
        response = requests.post(f"{BASE_URL}/reports/generate/", json=report_request)
        if response.status_code == 200:
            report = response.json()
            metrics = report.get('metrics', {})
            print(f"Report Generated Successfully!")
            print(f"  - Total Records: {metrics.get('total_records', 'N/A')}")
            print(f"  - Accuracy: {metrics.get('accuracy', {}).get('accuracy', 'N/A')}")
            print(f"  - Macro F1: {metrics.get('accuracy', {}).get('macro_f1', 'N/A')}")
            print(f"  - Mismatches Flagged: {metrics.get('flagged_mismatches', 'N/A')}")
            print(f"  - Human Reviews Required: {metrics.get('human_reviews_required', 'N/A')}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n--- Workflow Complete ---")

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("  HEALTH STATUS AGGREGATOR - API TEST SUITE")
    print("=" * 60)
    print(f"\nBase URL: {BASE_URL}")
    print("Starting tests...\n")
    
    results = {
        "passed": 0,
        "failed": 0
    }
    
    # Test 1: Health Check
    print_header("TEST 1: Health Check")
    if test_health_check():
        results["passed"] += 1
    else:
        results["failed"] += 1
    
    # Test 2: Single Record Analysis
    print_header("TEST 2: Single Record Analysis")
    analysis_result = test_single_record_analysis()
    if analysis_result:
        results["passed"] += 1
    else:
        results["failed"] += 1
    
    # Test 3: Batch Analysis
    print_header("TEST 3: Batch Analysis")
    batch_results = test_batch_analysis()
    if batch_results is not None:
        results["passed"] += 1
    else:
        results["failed"] += 1
    
    # Test 4: Get Pending Reviews
    print_header("TEST 4: Pending Reviews")
    pending_reviews = test_get_pending_reviews()
    if pending_reviews is not None:
        results["passed"] += 1
        # Test 5: Get Review Details (if there are pending reviews)
        if pending_reviews and len(pending_reviews) > 0:
            print_header("TEST 5: Review Details")
            review_id = pending_reviews[0].get("record_id", 9001)
            if test_get_review_details(review_id):
                results["passed"] += 1
            else:
                results["failed"] += 1
    else:
        results["failed"] += 1
    
    # Test 6: Submit Feedback
    print_header("TEST 6: Submit Feedback")
    if test_submit_feedback(9001):
        results["passed"] += 1
    else:
        results["failed"] += 1
    
    # Test 7: Data Validation
    print_header("TEST 7: Data Validation")
    if test_data_validation():
        results["passed"] += 1
    else:
        results["failed"] += 1
    
    # Test 8: Aggregation
    print_header("TEST 8: Aggregation by Account")
    if test_aggregation_by_account():
        results["passed"] += 1
    else:
        results["failed"] += 1
    
    # Test 9: Report Generation
    print_header("TEST 9: Report Generation")
    if test_generate_report():
        results["passed"] += 1
    else:
        results["failed"] += 1
    
    # Full Workflow Test
    run_full_workflow_test()
    
    # Summary
    print("\n" + "=" * 60)
    print("  TEST SUMMARY")
    print("=" * 60)
    print(f"\n  ✓ Passed: {results['passed']}")
    print(f"  ✗ Failed: {results['failed']}")
    print(f"  Total:   {results['passed'] + results['failed']}")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
