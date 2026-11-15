nextflow.enable.dsl=2

process RUN_QUANTUM_PIPELINE {
    // 1. THIS FIXES THE AWS BATCH ERROR
    container 'python:3.10-slim'
    
    publishDir "${params.outdir}/results", mode: 'copy'

    output:
    path "*.json", emit: json_results
    path "dashboard_bundle/*", emit: dashboard_assets, optional: true

    script:
    """
    pip install --quiet scikit-learn qiskit 
    
    # 2. THIS FIXES THE SCRIPT ERROR - CHANGE "main.py" TO YOUR ACTUAL FILENAME
    python main.py
    """
}

workflow {
    RUN_QUANTUM_PIPELINE()
}