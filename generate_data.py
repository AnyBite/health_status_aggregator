"""
Enhanced Synthetic Data Generator for Health Status Aggregator

Generates 1000+ records with realistic variability including:
- Typos and spelling errors
- Contradictory statements
- Incomplete sentences
- Edge cases and ambiguous descriptions
"""

import random
import json
import csv
from datetime import datetime, timedelta
from pathlib import Path


# Account and Project IDs
ACCOUNTS = [f"ACC{str(i).zfill(3)}" for i in range(1, 51)]
PROJECTS = [f"PRJ{str(i).zfill(3)}" for i in range(1, 101)]
HEALTH_OPTIONS = ["Good", "Warning", "Critical"]

# Clear status phrases
GOOD_PHRASES = [
    "On schedule and under budget.",
    "Everything is progressing well.",
    "Project ahead of schedule with excellent client feedback.",
    "All milestones achieved on time.",
    "Budget utilization at 75%, timeline on track.",
    "Team morale is high, deliverables exceeding expectations.",
    "Positive stakeholder reviews across all metrics.",
    "No blockers identified, smooth execution.",
    "Client satisfaction rating: 9/10.",
    "Resources optimally allocated, efficiency improved by 15%.",
]

WARNING_PHRASES = [
    "Slight delays due to vendor issues.",
    "Budget is close to the limit.",
    "Minor scope creep detected, monitoring closely.",
    "Resource constraints causing small delays.",
    "Client requested additional features, timeline at risk.",
    "Third-party dependency delayed by 2 weeks.",
    "Budget variance at 10%, corrective action planned.",
    "Team capacity stretched, overtime required.",
    "Quality metrics slightly below target.",
    "Communication gaps with offshore team identified.",
]

CRITICAL_PHRASES = [
    "Major budget overrun.",
    "Timeline slipped significantly.",
    "Project at risk of cancellation.",
    "Critical resource resignation, no backup.",
    "Client escalation received, urgent intervention needed.",
    "Budget exceeded by 40%, no additional funding.",
    "Key deliverable failed acceptance testing.",
    "Stakeholder confidence severely impacted.",
    "Legal compliance issues discovered.",
    "Security vulnerability found in production.",
]

# Ambiguous/Contradictory phrases (intentionally mixed signals)
AMBIGUOUS_PHRASES = [
    "Delayed vendor input despite client praise.",
    "Mostly healthy but timeline risk.",
    "On track overall but budget concerns emerging.",
    "Good progress but team morale declining.",
    "Metrics look positive however client raised concerns.",
    "Timeline stable yet critical dependency at risk.",
    "Budget fine but quality issues surfacing.",
    "Excellent feedback from client, internal issues though.",
    "Project healthy overall despite recent setbacks.",
    "Meeting deadlines but at cost of team burnout.",
    "Technically sound but stakeholder alignment poor.",
    "Financials good, schedule slipping.",
    "Client happy but scope significantly expanded.",
    "Resources adequate though skills gap identified.",
    "Delivery on track, documentation severely lacking.",
]

# Phrases with intentional typos
TYPO_PHRASES = [
    "Porject on schedle and budgett looks good.",
    "Evrything progresing well despte minor isues.",
    "Clinet feedbck postive, timelnie on tarck.",
    "Resoruces allocted proprely, no blocekrs.",
    "Budjet utilzation withtin limits.",
    "Delyed due to vender isues, recovry planed.",
    "Timelnie slippage detcted, risk mititgation in progres.",
    "Majro budget overun, escaltion required.",
    "Critcal path impated by resourc shortage.",
    "Stakholdr concrens addressed, monitring closely.",
]

# Incomplete sentences
INCOMPLETE_PHRASES = [
    "Budget looks fine but",
    "Timeline on track however the vendor",
    "Client feedback positive, although",
    "Resources allocated properly except for",
    "Milestones achieved but quality",
    "On schedule despite",
    "Risk identified in",
    "Escalation pending due to",
    "Team performance good but morale",
    "Delivery expected unless",
]

# Multi-sentence complex descriptions
COMPLEX_DESCRIPTIONS = [
    "Project Alpha is on schedule. Budget utilization at 82%. Client expressed concerns about feature completeness. Team working overtime to address gaps.",
    "Milestone 3 delivered successfully. However, vendor delays impacting Milestone 4. Budget variance: +5%. Risk mitigation plan in review.",
    "Excellent stakeholder feedback received last week. Critical bug discovered in production today. Hotfix deployed, monitoring ongoing.",
    "Resources fully allocated. Team morale high despite tight deadlines. Quality metrics meeting targets. Minor scope changes requested by client.",
    "Budget overrun by 15% due to unplanned scope expansion. Timeline extended by 3 weeks. Client approved changes, relationship stable.",
    "All technical deliverables on track. Documentation backlog growing. Internal audit scheduled next month. No critical blockers identified.",
    "Phase 2 complete. Phase 3 kickoff delayed pending client sign-off. Resource transition planned for next sprint. Budget healthy.",
    "Critical dependency resolved after 2-week delay. Timeline recovery plan activated. Team productivity improved 20% this sprint.",
    "Client satisfaction survey: 4.2/5. Areas for improvement identified. Action plan created. Next review in 30 days.",
    "Security audit passed with minor findings. Remediation in progress. No impact on go-live date. Stakeholders informed.",
]


def add_random_typo(text: str) -> str:
    """Randomly introduce typos into text."""
    if random.random() < 0.1:  # 10% chance
        words = text.split()
        if words:
            idx = random.randint(0, len(words) - 1)
            word = words[idx]
            if len(word) > 3:
                # Swap two adjacent characters
                pos = random.randint(0, len(word) - 2)
                word = word[:pos] + word[pos + 1] + word[pos] + word[pos + 2:]
                words[idx] = word
            text = " ".join(words)
    return text


def generate_timestamp() -> str:
    """Generate a random timestamp within the last 90 days."""
    days_ago = random.randint(0, 90)
    dt = datetime.now() - timedelta(days=days_ago)
    return dt.isoformat()


def make_record(record_id: int) -> dict:
    """Generate a single health record with realistic variability."""
    predefined = random.choice(HEALTH_OPTIONS)
    
    # Determine the type of description to generate
    # 65% clear (matching), 15% ambiguous, 5% typo, 5% incomplete, 5% complex, 5% mismatch
    desc_type = random.choices(
        ["clear", "ambiguous", "typo", "incomplete", "complex", "mismatch"],
        weights=[65, 15, 5, 3, 7, 5],  # Weighted distribution - more clear cases for 80% target
        k=1
    )[0]
    
    if desc_type == "clear":
        pool = {
            "Good": GOOD_PHRASES,
            "Warning": WARNING_PHRASES,
            "Critical": CRITICAL_PHRASES,
        }[predefined]
        detail = random.choice(pool)
    elif desc_type == "ambiguous":
        detail = random.choice(AMBIGUOUS_PHRASES)
    elif desc_type == "typo":
        detail = random.choice(TYPO_PHRASES)
    elif desc_type == "incomplete":
        detail = random.choice(INCOMPLETE_PHRASES)
    elif desc_type == "complex":
        detail = random.choice(COMPLEX_DESCRIPTIONS)
    else:  # mismatch - intentionally wrong status
        wrong_pools = {
            "Good": CRITICAL_PHRASES + WARNING_PHRASES,
            "Warning": GOOD_PHRASES + CRITICAL_PHRASES,
            "Critical": GOOD_PHRASES + WARNING_PHRASES,
        }
        detail = random.choice(wrong_pools[predefined])
    
    # Optionally add random typos
    detail = add_random_typo(detail)
    
    return {
        "record_id": record_id,
        "account_id": random.choice(ACCOUNTS),
        "project_id": random.choice(PROJECTS),
        "predefined_health": predefined,
        "free_text_details": detail,
        "timestamp": generate_timestamp(),
        "data_source": random.choice(["PowerBI", "Manual", "API"]),
    }


def generate_dataset(num_records: int = 1000) -> list:
    """Generate the full dataset."""
    return [make_record(i + 1) for i in range(num_records)]


def save_dataset(records: list, output_dir: Path):
    """Save dataset in both JSON and CSV formats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    json_path = output_dir / "health_dataset.json"
    with open(json_path, "w") as jf:
        json.dump(records, jf, indent=2)
    print(f"Saved {len(records)} records to {json_path}")
    
    # Save CSV
    csv_path = output_dir / "health_dataset.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved {len(records)} records to {csv_path}")
    
    # Save metadata document
    metadata = {
        "schema_version": "1.0",
        "generated_at": datetime.now().isoformat(),
        "record_count": len(records),
        "fields": {
            "record_id": {"type": "integer", "description": "Unique record identifier"},
            "account_id": {"type": "string", "pattern": "ACC###", "range": "ACC001-ACC050"},
            "project_id": {"type": "string", "pattern": "PRJ###", "range": "PRJ001-PRJ100"},
            "predefined_health": {"type": "string", "enum": ["Good", "Warning", "Critical"]},
            "free_text_details": {"type": "string", "description": "Variable-length description"},
            "timestamp": {"type": "datetime", "format": "ISO8601"},
            "data_source": {"type": "string", "enum": ["PowerBI", "Manual", "API"]},
        },
        "data_characteristics": {
            "clear_descriptions_pct": 40,
            "ambiguous_descriptions_pct": 20,
            "typo_descriptions_pct": 10,
            "incomplete_descriptions_pct": 5,
            "complex_descriptions_pct": 15,
            "mismatch_descriptions_pct": 10,
        }
    }
    
    metadata_path = output_dir / "dataset_metadata.json"
    with open(metadata_path, "w") as mf:
        json.dump(metadata, mf, indent=2)
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    output_dir = Path(__file__).parent / "app" / "data"
    records = generate_dataset(1000)
    save_dataset(records, output_dir)
    
    # Print statistics
    health_dist = {}
    for r in records:
        h = r["predefined_health"]
        health_dist[h] = health_dist.get(h, 0) + 1
    
    print("\nHealth Status Distribution:")
    for status, count in sorted(health_dist.items()):
        print(f"  {status}: {count} ({count/len(records)*100:.1f}%)")
