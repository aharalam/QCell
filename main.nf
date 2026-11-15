nextflow.enable.dsl=2

params.outdir  = 'results'        // use an s3:// path when launching on AWS Batch
params.mode    = 'full'           // 'smoke': reference simulator, small sample, no secrets, a few minutes
params.engine  = 'quantum_rings'  // full mode only; 'reference' runs without a Quantum Rings account
params.n_cells = 200              // full mode only
// QuantumRingsLib needs Python 3.11+; on 3.10 pip silently installs a 2024 release.
// Not the -slim image: Nextflow's task wrapper needs `ps` (procps) to collect task
// metrics, and slim images don't include it, so the task fails before the script runs.
params.container = 'python:3.12-trixie'

// Shared setup: print what is being installed so a failure shows where it stopped.
// requireQuantumRings: false in smoke mode, which runs on the reference simulator, so a
// Quantum Rings import failure is reported as a warning there instead of stopping the run.
def setupScript(boolean requireQuantumRings) {
    def onImportFailure = requireQuantumRings ? 'exit 1' : 'true'
    """
    echo "== \$(python --version 2>&1) on \$(uname -m), \$(getconf _NPROCESSORS_ONLN) CPUs"
    echo "== Installing packages"
    pip install --no-cache-dir --progress-bar off --root-user-action=ignore \\
        numpy scikit-learn scanpy umap-learn qiskit matplotlib pylatexenc \\
        QuantumRingsLib quantumrings-toolkit-qiskit
    echo "== Installed versions"
    pip list 2>/dev/null | grep -i -E '^(numpy|scikit-learn|scanpy|umap-learn|qiskit|quantumrings)' || true
    if python -c "import QuantumRingsLib, quantumrings.toolkit.qiskit"; then
        echo "== Quantum Rings imports OK"
    else
        echo "== WARNING: Quantum Rings failed to import on \$(uname -m) (exit \$?)" >&2
        ${onImportFailure}
    fi
    """
}

// Quick end-to-end check of the container, packages, data download and bundle export.
// Uses no secrets, so it also runs when the secrets themselves are the problem.
process SMOKE_TEST {
    container params.container
    cpus 2
    memory '8 GB'
    time '45m'
    publishDir "${params.outdir}/smoke", mode: 'copy'

    input:
    path src

    output:
    path "out/*/dashboard_bundle/*", emit: dashboard_bundle

    script:
    """
    ${setupScript(false)}
    echo "== Running smoke benchmark"
    python -u ${src}/runner.py \\
        --engine reference \\
        --n-cells 80 \\
        --outer-folds 2 \\
        --inner-folds 2 \\
        --out out \\
        --run-id smoke-${workflow.runName}
    """
}

process RUN_QUANTUM_PIPELINE {
    container params.container
    cpus 6
    // More memory on each retry, in case the container was killed for running out of it
    memory { 16.GB * task.attempt }
    time '3h'
    // Retry when AWS Batch stops the task (Spot reclaim, out of memory), but stop at once
    // on a real script error (Python exits 1, bad arguments exit 2) so it isn't re-run for nothing
    errorStrategy { task.exitStatus in [1, 2] ? 'terminate' : 'retry' }
    maxRetries 2
    // Create both secrets in Seqera Platform before launching. The AWS Batch execution role
    // needs secretsmanager:GetSecretValue, or the container never starts.
    secret 'QUANTUM_RINGS_API_KEY'
    secret 'QUANTUM_RINGS_ACCOUNT'
    publishDir params.outdir, mode: 'copy'

    input:
    path src

    output:
    path "out/*/dashboard_bundle/*", emit: dashboard_bundle

    script:
    """
    for name in QUANTUM_RINGS_API_KEY QUANTUM_RINGS_ACCOUNT; do
        if [ -z "\${!name:-}" ]; then echo "== Secret \$name is not set" >&2; exit 1; fi
    done
    echo "== Quantum Rings secrets are set"
    ${setupScript(true)}
    echo "== Running benchmark"
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
    def src = file("${projectDir}/modeling/src", checkIfExists: true)
    if (params.mode == 'smoke') {
        SMOKE_TEST(src)
    } else if (params.mode == 'full') {
        RUN_QUANTUM_PIPELINE(src)
    } else {
        error "Unknown --mode '${params.mode}'. Use 'smoke' or 'full'."
    }
}
