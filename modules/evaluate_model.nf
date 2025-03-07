process EVALUATE_MODEL {
    publishDir "${params.outdir}/evaluation", mode: 'copy'
    
    input:
    path model
    path test_data
    val model_type
    val device
    
    output:
    path 'evaluation_metrics.json'
    path 'confusion_matrix.png'
    path 'roc_curve.png'
    
    script:
    """
    python ${baseDir}/bin/evaluate.py \
        --model ${model} \
        --test_data ${test_data} \
        --model_type ${model_type} \
        --device ${device} \
        --output_dir ./
    """
}