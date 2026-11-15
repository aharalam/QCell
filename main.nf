nextflow.enable.dsl=2

params.outdir  = 'results'        // use an s3:// path when launching on AWS Batch
params.engine  = 'quantum_rings'  // 'reference' runs without a Quantum Rings account
params.n_cells = 200

process RUN_QUANTUM_PIPELINE {
    container 'python:3.10-slim'
    cpus 6
    // More memory on each retry, in case the container was killed for running out of it
    memory { 16.GB * task.attempt }
    time '3h'
    // Retry when AWS Batch stops the task (Spot reclaim, out of memory), but stop at once
    // on a real script error (Python exits 1, bad arguments exit 2) so it isn't re-run for nothing
    errorStrategy { task.exitStatus in [1, 2] ? 'terminate' : 'retry' }
    maxRetries 2
    // Create both secrets in Seqera Platform before launching.
    // Remove these two lines when running with --engine reference.
    secret 'QUANTUM_RINGS_API_KEY'
    secret 'QUANTUM_RINGS_ACCOUNT'
    publishDir params.outdir, mode: 'copy'

    input:
    path src

    output:
    path "out/*/dashboard_bundle/*", emit: dashboard_bundle

    script:
    """
    pip install --quiet numpy scikit-learn scanpy umap-learn qiskit matplotlib pylatexenc \\
        QuantumRingsLib quantumrings-toolkit-qiskit

    # -u: write output as it happens, so a killed task still shows how far it got
    python -u ${src}/runner.py \\
        --engine ${params.engine} \\
        --n-cells ${params.n_cells} \\
        --workers ${task.cpus} \\
        --out out \\
        --run-id ${workflow.runName}
    """
}

workflow {
    // Stage the whole source folder: runner.py imports the modules next to it.
    RUN_QUANTUM_PIPELINE(file("${projectDir}/modeling/src", checkIfExists: true))
}
