#!/usr/bin/env python3
"""
Master pipeline runner for AI Phishing Detector.
Runs the complete workflow: dataset generation -> feature extraction -> model training -> evaluation
"""

import sys
import subprocess
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
NOTEBOOKS_DIR = ROOT / "notebooks"

# Steps in the pipeline
PIPELINE_STEPS = [
    {
        'name': 'Dataset Generation',
        'script': NOTEBOOKS_DIR / 'generate_enhanced_dataset.py',
        'description': 'Generate enhanced phishing dataset with 5000+ URLs',
        'required_output': ROOT / 'data' / 'phishing_enhanced.csv'
    },
    {
        'name': 'Feature Extraction',
        'script': NOTEBOOKS_DIR / 'advanced_features.py',
        'description': 'Extract 37 advanced features from URLs',
        'required_output': ROOT / 'data' / 'phishing_features.csv'
    },
    {
        'name': 'Model Training & Comparison',
        'script': NOTEBOOKS_DIR / 'compare_models.py',
        'description': 'Train and compare multiple ML models',
        'required_output': ROOT / 'models' / 'best_model.pkl'
    },
    {
        'name': 'Model Evaluation',
        'script': NOTEBOOKS_DIR / 'evaluate_model.py',
        'description': 'Final evaluation with detailed metrics',
        'required_output': ROOT / 'docs' / 'confusion_matrix.png'
    }
]


def print_banner(text, char='='):
    """Print a formatted banner"""
    width = 70
    print(f"\n{char * width}")
    print(f"{text.center(width)}")
    print(f"{char * width}\n")


def run_step(step_num, total_steps, step_info):
    """Run a single pipeline step"""
    print_banner(f"STEP {step_num}/{total_steps}: {step_info['name']}", char='=')
    print(f"📝 Description: {step_info['description']}")
    print(f"📂 Script: {step_info['script']}")
    
    if not step_info['script'].exists():
        print(f"❌ ERROR: Script not found at {step_info['script']}")
        return False
    
    # Check if output already exists
    if step_info['required_output'].exists():
        print(f"⚠️  Output already exists: {step_info['required_output']}")
        response = input("   Skip this step? (y/n): ").strip().lower()
        if response == 'y':
            print("⏭️  Skipping...")
            return True
    
    print(f"\n🚀 Running...")
    start_time = time.time()
    
    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, str(step_info['script'])],
            cwd=ROOT,
            capture_output=True,
            text=True
        )
        
        elapsed_time = time.time() - start_time
        
        # Print output
        if result.stdout:
            print(result.stdout)
        
        if result.returncode == 0:
            print(f"✅ SUCCESS! (completed in {elapsed_time:.2f}s)")
            
            # Verify output was created
            if not step_info['required_output'].exists():
                print(f"⚠️  WARNING: Expected output not found: {step_info['required_output']}")
            
            return True
        else:
            print(f"❌ FAILED with return code {result.returncode}")
            if result.stderr:
                print(f"Error output:\n{result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def main():
    """Run the complete pipeline"""
    print_banner("AI PHISHING DETECTOR - MASTER PIPELINE", char='█')
    print("This script will run the complete machine learning pipeline:")
    print("  1. Generate enhanced dataset (5000+ URLs)")
    print("  2. Extract advanced features (37 features)")
    print("  3. Train and compare multiple models")
    print("  4. Evaluate best model with detailed metrics")
    print("\n⏱️  Estimated time: 3-5 minutes\n")
    
    response = input("Ready to start? (y/n): ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        return
    
    # Run all steps
    total_steps = len(PIPELINE_STEPS)
    failed_steps = []
    
    for idx, step in enumerate(PIPELINE_STEPS, 1):
        success = run_step(idx, total_steps, step)
        
        if not success:
            failed_steps.append(step['name'])
            print(f"\n⚠️  Step failed. Continue anyway? (y/n): ", end='')
            response = input().strip().lower()
            if response != 'y':
                print("\n❌ Pipeline aborted.")
                return
    
    # Summary
    print_banner("PIPELINE COMPLETE", char='█')
    
    if not failed_steps:
        print("✅ All steps completed successfully!\n")
        print("📊 Generated outputs:")
        print(f"   • Enhanced dataset:     {ROOT / 'data' / 'phishing_enhanced.csv'}")
        print(f"   • Feature dataset:      {ROOT / 'data' / 'phishing_features.csv'}")
        print(f"   • Best model:           {ROOT / 'models' / 'best_model.pkl'}")
        print(f"   • Comparison plots:     {ROOT / 'docs' / 'model_comparison.png'}")
        print(f"   • Confusion matrices:   {ROOT / 'docs' / 'confusion_matrices.png'}")
        print(f"   • Evaluation results:   {ROOT / 'docs' / 'model_comparison_results.csv'}")
        
        print("\n🚀 Next steps:")
        print("   1. Review the generated visualizations in docs/")
        print("   2. Test the best model with: python app/main.py")
        print("   3. (Optional) Run hyperparameter tuning for better performance")
        
    else:
        print(f"⚠️  Pipeline completed with {len(failed_steps)} failed step(s):")
        for step in failed_steps:
            print(f"   • {step}")
        print("\nPlease check the error messages above and fix any issues.")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user.")
        sys.exit(1)
