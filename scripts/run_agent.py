#!/usr/bin/env python3
"""
CLI Entry Point - Run the Agentic Insight Analyst from command line.
"""

import argparse
import json
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent import InsightAgent
from src.config import get_config


console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="Agentic Insight Analyst - Synthesize actionable insights from data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_agent.py --goal "Summarize key pain points" --data data/sample/sample_survey.csv
  python scripts/run_agent.py --goal "Identify top 3 issues" --data survey.csv --verbose
        """
    )
    
    parser.add_argument(
        "--goal", "-g",
        required=True,
        help="Natural language analysis goal"
    )
    
    parser.add_argument(
        "--data", "-d",
        required=True,
        help="Path to data file (CSV)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output file for JSON results (optional)"
    )
    
    parser.add_argument(
        "--max-iterations",
        type=int,
        help="Override maximum iterations"
    )
    
    args = parser.parse_args()
    
    # Validate data path
    data_path = Path(args.data)
    if not data_path.exists():
        console.print(f"[red]Error: Data file not found: {args.data}[/red]")
        sys.exit(1)
    
    # Display header
    console.print(Panel.fit(
        "[bold blue]Agentic Insight Analyst[/bold blue]\n"
        "Synthesizing actionable insights with controlled autonomy",
        border_style="blue"
    ))
    
    console.print(f"\n[bold]Goal:[/bold] {args.goal}")
    console.print(f"[bold]Data:[/bold] {args.data}\n")
    
    # Run agent
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Running analysis...", total=None)
        
        try:
            agent = InsightAgent()
            result = agent.run(goal=args.goal, data_path=str(data_path))
        except Exception as e:
            progress.stop()
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)
    
    # Display results
    if result.success:
        console.print("\n[green]✓ Analysis Complete[/green]\n")
        
        # Executive Summary
        console.print(Panel(
            result.final_output.get("executive_summary", "No summary available"),
            title="Executive Summary",
            border_style="green"
        ))
        
        # Key Findings
        findings_table = Table(title="Key Findings", show_header=True)
        findings_table.add_column("Finding", style="cyan")
        
        findings = result.final_output.get("key_findings", [])
        for finding in findings[:5]:
            if isinstance(finding, dict):
                findings_table.add_row(finding.get("finding", str(finding)))
            else:
                findings_table.add_row(str(finding))
        
        console.print(findings_table)
        
        # Recommendations
        rec_table = Table(title="Recommendations", show_header=True)
        rec_table.add_column("Recommendation", style="yellow")
        rec_table.add_column("Priority", style="red")
        
        recommendations = result.final_output.get("recommendations", [])
        for rec in recommendations[:5]:
            if isinstance(rec, dict):
                rec_table.add_row(
                    rec.get("recommendation", str(rec))[:80],
                    rec.get("priority", "medium")
                )
            else:
                rec_table.add_row(str(rec)[:80], "medium")
        
        console.print(rec_table)
        
        # Insights discovered
        if result.insights:
            console.print("\n[bold]Insights Discovered:[/bold]")
            for insight in result.insights:
                console.print(f"  • {insight}")
        
        # Statistics
        console.print(f"\n[dim]Completed in {result.iterations} iterations[/dim]")
        
        # Verbose output
        if args.verbose:
            console.print("\n[bold]Reasoning Trace:[/bold]")
            for i, step in enumerate(result.reasoning_trace, 1):
                console.print(f"  {i}. {step}")
        
        # Save to file if requested
        if args.output:
            output_data = {
                "goal": result.goal,
                "success": result.success,
                "final_output": result.final_output,
                "insights": result.insights,
                "iterations": result.iterations,
                "reasoning_trace": result.reasoning_trace
            }
            
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            
            console.print(f"\n[dim]Results saved to: {args.output}[/dim]")
    
    else:
        console.print("[red]✗ Analysis Failed[/red]")
        console.print(f"Error: {result.final_output.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
