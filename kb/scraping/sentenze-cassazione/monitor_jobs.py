"""
monitor_jobs.py
---------------
Python script to monitor and report on all running Cloud Run Jobs.
"""

import subprocess
import json
import time
from datetime import datetime
from typing import List, Dict


def get_job_executions(project_id: str, region: str, job_name: str) -> List[Dict]:
    """Get all executions for a job."""
    try:
        cmd = [
            "gcloud", "run", "jobs", "executions", "list",
            "--job", job_name,
            "--region", region,
            "--project", project_id,
            "--format", "json"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return json.loads(result.stdout)
        return []
    except Exception as e:
        print(f"Error getting executions for {job_name}: {e}")
        return []


def get_execution_logs(project_id: str, job_name: str, limit: int = 10) -> List[str]:
    """Get recent logs for a job."""
    try:
        cmd = [
            "gcloud", "logging", "read",
            f'resource.labels.job_name="{job_name}"',
            "--project", project_id,
            "--limit", str(limit),
            "--format", "value(textPayload)"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip().split('\n')
        return []
    except Exception as e:
        print(f"Error getting logs for {job_name}: {e}")
        return []


def monitor_jobs(project_id: str, years: List[int], regions: List[str], chunks_per_year: int):
    """Monitor all jobs and display status."""

    # Map years to regions (round-robin)
    job_configs = []
    for i, year in enumerate(years):
        region = regions[i % len(regions)]
        for chunk in range(chunks_per_year):
            job_configs.append({
                'year': year,
                'region': region,
                'chunk': chunk,
                'job_name': f'italgiure-scraper-{year}-c{chunk:02d}'
            })
    
    print("=" * 100)
    print(f"📊 Monitoring Cloud Run Jobs - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)
    print()
    
    # Get status for each job
    for config in job_configs:
        year = config['year']
        region = config['region']
        chunk = config['chunk']
        job_name = config['job_name']

        print(f"📅 Year {year} chunk {chunk:02d} ({region}):")
        
        # Get latest execution
        executions = get_job_executions(project_id, region, job_name)
        
        if not executions:
            print(f"   ℹ️  No executions found")
            print()
            continue
        
        # Get most recent execution
        latest = executions[0]
        status = latest.get('status', {})
        
        # Extract status info
        conditions = status.get('conditions', [])
        completion_time = status.get('completionTime')
        start_time = status.get('startTime')
        succeeded = status.get('succeededCount', 0)
        failed = status.get('failedCount', 0)
        
        # Determine overall status
        if conditions:
            condition_type = conditions[0].get('type', 'Unknown')
            
            if condition_type == 'Completed':
                if succeeded > 0:
                    status_icon = "✅"
                    status_text = "COMPLETED"
                else:
                    status_icon = "❌"
                    status_text = "FAILED"
            elif condition_type == 'Running':
                status_icon = "🔄"
                status_text = "RUNNING"
            else:
                status_icon = "⏳"
                status_text = condition_type
        else:
            status_icon = "❓"
            status_text = "UNKNOWN"
        
        print(f"   Status: {status_icon} {status_text}")
        
        if start_time:
            print(f"   Started: {start_time}")
        
        if completion_time:
            print(f"   Completed: {completion_time}")
        
        if succeeded > 0 or failed > 0:
            print(f"   Tasks: {succeeded} succeeded, {failed} failed")
        
        # Get recent logs with stats
        logs = get_execution_logs(project_id, job_name, limit=50)
        
        # Parse logs for statistics
        for log in logs:
            if "Final Statistics:" in log or "Successful:" in log or "Success rate:" in log:
                print(f"   {log.strip()}")
        
        print()
    
    print("=" * 100)


def main():
    """Main entry point."""
    import os
    from datetime import datetime

    PROJECT_ID = os.getenv('GCP_PROJECT_ID', 'your-project-id')
    CURRENT_YEAR = datetime.now().year
    START_YEAR = CURRENT_YEAR - 5
    DOCUMENTS_PER_SHARD = int(os.getenv('DOCUMENTS_PER_SHARD', '1200'))
    ESTIMATED_DOCS_PER_YEAR = int(os.getenv('ESTIMATED_DOCS_PER_YEAR', '50000'))
    CHUNKS_PER_YEAR = (ESTIMATED_DOCS_PER_YEAR + DOCUMENTS_PER_SHARD - 1) // DOCUMENTS_PER_SHARD

    YEARS = list(range(START_YEAR, CURRENT_YEAR + 1))
    REGIONS = [
        "europe-west8",
        "europe-west1",
        "europe-west3",
        "europe-west4",
        "europe-west6",
        "europe-west9"
    ]

    try:
        # Monitor once
        monitor_jobs(PROJECT_ID, YEARS, REGIONS, CHUNKS_PER_YEAR)

        # Ask if user wants to continue monitoring
        print("Monitor continuously? (Ctrl+C to stop)")
        input("Press Enter to start continuous monitoring...")

        while True:
            print("\033[2J\033[H")  # Clear screen
            monitor_jobs(PROJECT_ID, YEARS, REGIONS, CHUNKS_PER_YEAR)
            time.sleep(30)  # Update every 30 seconds

    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped")


if __name__ == "__main__":
    main()
