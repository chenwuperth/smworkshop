.PHONY: clean help draw

# these three could be env variables external to Makefile
INPUT_FNAME?=abalone.csv
INPUT_S3_BUCKET?=sagemaker-ap-southeast-2-454979696062
INPUT_S3_PREFIX?=inference-pipeline-scikit-linearlearner/train

# these variables are hard-coded as per the source code structure
MODULE=examples.07
MODULE_TRAIN=$(MODULE).train
MODULE_INFER=$(MODULE).inference
INPUT_LOCAL_DIR=examples/07/train

# these variables are derived from the above variables
TRANSFORMER_OUTPUT_SUFFIX=.out
INPUT_S3_DIR=$(INPUT_S3_BUCKET)/$(INPUT_S3_PREFIX)
INPUT_LOCAL_FILE=$(INPUT_LOCAL_DIR)/$(INPUT_FNAME)

#TODO - define our own functions to grep parameters - https://coderwall.com/p/cezf6g/define-your-own-function-in-a-makefile
#TODO - test file size, and break if it is zero - 
	# https://stackoverflow.com/questions/10673189/checking-file-size-in-makefile-stopping-if-file-is-too-short
	# test -n "$$(find filename -a -size +NNNc)"
#TODO - proper error handling (remove the zero-sized file before exiting), also exit with error code
	#https://stackoverflow.com/questions/44061611/makefile-error-exit-with-a-message

## raw_data_on_s3:		upload local dataset to S3 if any more recent changes
raw_data_on_s3: $(INPUT_LOCAL_FILE)
	aws s3 sync $(INPUT_LOCAL_DIR) s3://$(INPUT_S3_DIR) --exclude='*' --include='/*.csv' > $@

## trained_preproc_model: 	train the data pre-processing model using sagemaker scikit learn framework
trained_preproc_model: raw_data_on_s3
	$(eval raw_input=$(shell cat $<  | rev | cut -d' ' -f1 | rev))
	python -m $(MODULE_TRAIN).train_preproc_job --train-s3-path $(raw_input) > $@

## preprocessed_data:		inference using the pre-processor model, and produces featurized data in the "output-dir" (S3 locations)
preprocessed_data: trained_preproc_model raw_data_on_s3
	$(eval preproc_model=$(shell cat $< | tail -1 | rev | cut -d' ' -f1 | rev))
	$(eval raw_input=$(shell cat $(word 2,$^) | rev | cut -d' ' -f1 | rev))
	python -m $(MODULE_INFER).infer_preproc_job --input-file $(raw_input) --model-file $(preproc_model) --output-dir s3://$(INPUT_S3_DIR) > $@

## trained_ml_model:		train the ML model  using featurized data from the last step, output the trained ML model in S3
trained_ml_model: preprocessed_data
	$(eval preproc_data=$(shell cat $< | tail -1 | rev | cut -d' ' -f1 | rev))
	python -m $(MODULE_TRAIN).train_job --train-s3-path $(preproc_data)/$(INPUT_FNAME)$(TRANSFORMER_OUTPUT_SUFFIX) > $@

## infer_pipeline_endpoint:	create an inference pipeline by chaining the pre-processor model and the ML model into a single endpoint
infer_pipeline_endpoint: trained_ml_model trained_preproc_model
	$(eval ml_model=$(shell cat $< | tail -1 | rev | cut -d' ' -f1 | rev))
	$(eval preproc_model=$(shell cat $(word 2,$^) | tail -1 | rev | cut -d' ' -f1 | rev))
	python -m $(MODULE_INFER).inference_job --infer-mode ep --model-file $(ml_model) --preproc-model $(preproc_model) > $@

## invoke_endpoint:		invoke the endpoint 
invoke_endpoint: infer_pipeline_endpoint
	$(eval ep_url=$(shell cat $< | tail -1 | rev | cut -d' ' -f1 | rev))
	python -m $(MODULE_INFER).invoke_endpoint --end-point $(ep_url) > $@

## delete_endpoint:		delete the endpoint
delete_endpoint: invoke_endpoint
	$(eval ep_url=$(shell cat $< | head -1 | rev | cut -d' ' -f1 | rev))
	python -m $(MODULE_INFER).invoke_endpoint --end-point $(ep_url) --delete > $@

clean :
	rm raw_data_on_s3
	rm trained_preproc_model
	rm preprocessed_data
	rm trained_ml_model
	rm infer_pipeline_endpoint
	rm invoke_endpoint
	rm delete_endpoint

help : $(firstword $(MAKEFILE_LIST))
	@sed -n 's/^##//p' $<

draw :
	makefile2dot -v --direction TB
