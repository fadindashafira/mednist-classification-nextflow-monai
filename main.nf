#!/usr/bin/env nextflow

nextflow.enable.dsl=2

// Log info
log.info """\
         M E D N I S T - C L A S S I F I C A T I O N    
         ===================================
         Dataset       : ${params.data_dir}
         Output dir    : ${params.outdir}
         Model type    : ${params.model_type}
         Epochs        : ${params.epochs}
         Batch size    : ${params.batch_size}
         Learning rate : ${params.learning_rate}
         Device        : ${params.device}
         """
         .stripIndent()

// Import modules
include { PREPARE_DATA } from './modules/prepare_data'
include { TRAIN_MODEL } from './modules/train_model'
include { EVALUATE_MODEL } from './modules/evaluate_model'

// Main workflow
workflow {
    // Prepare data
    PREPARE_DATA( 
        Channel.value(params.data_dir)
    )
    
    // Train model
    TRAIN_MODEL(
        PREPARE_DATA.out.train_data,
        PREPARE_DATA.out.val_data,
        Channel.value(params.model_type),
        Channel.value(params.epochs),
        Channel.value(params.batch_size),
        Channel.value(params.learning_rate),
        Channel.value(params.device)
    )
    
    // Evaluate model
    EVALUATE_MODEL(
        TRAIN_MODEL.out.model,
        PREPARE_DATA.out.test_data,
        Channel.value(params.model_type),
        Channel.value(params.device)
    )
}

// Module definitions
workflow.onComplete {
    log.info "Pipeline completed at: $workflow.complete"
    log.info "Execution status: ${ workflow.success ? 'OK' : 'failed' }"
    log.info "Execution duration: $workflow.duration"
}