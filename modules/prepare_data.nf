process PREPARE_DATA {
    publishDir "${params.outdir}/data", mode: 'copy'
    
    input:
    val data_dir
    
    output:
    path 'train_data.pt', emit: train_data
    path 'val_data.pt', emit: val_data
    path 'test_data.pt', emit: test_data
    
    script:
    """
    python ${baseDir}/bin/data.py \
        --data_dir ${data_dir} \
        --output_dir ./ \
        --cache_rate ${params.cache_rate} \
        --val_frac ${params.val_frac} \
        --test_frac ${params.test_frac} \
        --seed ${params.seed} \
        --download
    """
}