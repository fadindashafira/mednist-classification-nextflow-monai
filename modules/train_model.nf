process TRAIN_MODEL {
    publishDir "${params.outdir}/model", mode: 'copy'
    
    input:
    path train_data
    path val_data
    val model_type
    val epochs
    val batch_size
    val learning_rate
    val device
    
    output:
    path 'model.pt', emit: model
    path 'training_metrics.json', emit: metrics
    
    script:
    """
    python ${baseDir}/bin/train.py \
        --train_data ${train_data} \
        --val_data ${val_data} \
        --model_type ${model_type} \
        --epochs ${epochs} \
        --batch_size ${batch_size} \
        --learning_rate ${learning_rate} \
        --device ${device} \
        --output_dir ./
    """
}