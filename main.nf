// main.nf
nextflow.enable.dsl=2

process RUN_QUANTUM_PIPELINE {
    publishDir "${params.outdir}/results", mode: 'copy'

    output:
    path "*.json", emit: json_results
    path "dashboard_bundle/*", emit: dashboard_assets, optional: true

    script:
    """
    pip install --quiet scikit-learn qiskit 
    
    // REPLACE THIS with your actual python script name from the repo
    python your_main_script.py 
    """
}

workflow {
    RUN_QUANTUM_PIPELINE()
}