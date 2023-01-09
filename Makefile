# Usage:
# make install			# downloads miniconda and initializes conda environment
# make install_mimic	# downloads required mimic files from physionet (physionet credentialed account required)
# make server	  		# starts mlflow server at port 5000
# make run  			# executes main.py within the conda environment \
				  			example: make run ARGS="--experimentconfig_sequence_type huawei_logs"
# make run_mimic 		# executes main.py within the conda environment for all knowledge types on mimic dataset
# make run_huawei		# executes main.py within the conda environment for all knowledge types on huawei dataset

CONDA_ENV_NAME = lena
CONDA_URL = https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
CONDA_SH = Miniconda3-latest-Linux-x86_64.sh
CONDA_DIR = /home/vincenzo/anaconda3

DATA_DIR = data
KNOWLEDGE_TYPES = simple gram

install:
ifneq (,$(wildcard ${CONDA_DIR}))
	@echo "Remove old install files"
	@rm -Rf ${CONDA_DIR}
endif
	@echo "Downloading miniconda..."
	@mkdir ${CONDA_DIR}
	@cd .tmp && wget -nc ${CONDA_URL} > /dev/null
	@chmod +x ./${CONDA_DIR}/${CONDA_SH}
	@./${CONDA_DIR}/${CONDA_SH} -b -u -p ./${CONDA_DIR}/miniconda3/ > /dev/null
	@echo "Initializing conda environment..."
	@./${CONDA_DIR}/miniconda3/bin/conda env create -q --force -f environment.yml > /dev/null
	@echo "Finished!"

install_mimic:
	@cd ${DATA_DIR}
	@wget -N -c --user ${PHYSIONET_USER} --ask-password https://physionet.org/files/mimiciii/1.4/ADMISSIONS.csv.gz
	@wget -N -c --user ${PHYSIONET_USER} --ask-password https://physionet.org/files/mimiciii/1.4/DIAGNOSES_ICD.csv.gz
	@gzip -d ADMISSIONS.csv.gz
	@gzip -d DIAGNOSES_ICD.csv.gz

server:
	@echo "Starting MLFlow UI at port 5000"
	PATH="${PATH}:$(shell pwd)/${CONDA_DIR}/miniconda3/envs/${CONDA_ENV_NAME}/bin" ; \
	${CONDA_DIR}/envs/${CONDA_ENV_NAME}/bin/mlflow server --gunicorn-opts -t180

notebook:
	@echo "Starting Jupyter Notebook at port 8888"
	PATH="${PATH}:$(shell pwd)/${CONDA_DIR}/miniconda3/envs/${CONDA_ENV_NAME}/bin" ; \
	./${CONDA_DIR}/miniconda3/envs/${CONDA_ENV_NAME}/bin/jupyter notebook notebooks/ --no-browser 

run: 
	./${CONDA_DIR}/miniconda3/envs/${CONDA_ENV_NAME}/bin/python main.py ${ARGS}

run_huawei:
	for knowledge_type in ${KNOWLEDGE_TYPES} ; do \
		echo "Starting experiment for huawei_logs with knowledge type " $$knowledge_type "....." ; \
		${CONDA_DIR}/envs/${CONDA_ENV_NAME}/bin/python main.py \
			--experimentconfig_model_type $$knowledge_type \
			--huaweipreprocessorconfig_min_causality 0.01 \
			--huaweipreprocessorconfig_relevant_log_column attention_log_cluster_template \
			--no-modelconfig_base_feature_embeddings_trainable \
			--no-modelconfig_base_hidden_embeddings_trainable \
		  --sequenceconfig_x_sequence_column_name fine_log_cluster_template \
		  --sequenceconfig_y_sequence_column_name attributes \
		  --sequenceconfig_max_window_size 10 \
		  --sequenceconfig_min_window_size 10 \
			--experimentconfig_multilabel_classification \
			--sequenceconfig_flatten_y \
			--sequenceconfig_flatten_x \
			${ARGS} ; \
	done ; \



run_attention:
	for knowledge_type in ${KNOWLEDGE_TYPES} ; do \
		echo "Starting experiment for huawei_logs with knowledge type " $$knowledge_type "....." ; \
		${CONDA_DIR}/envs/${CONDA_ENV_NAME}/bin/python main.py \
			--experimentconfig_model_type $$knowledge_type \
			--huaweipreprocessorconfig_min_causality 0.01 \
			--huaweipreprocessorconfig_relevant_log_column log_cluster_template\
			--no-modelconfig_base_feature_embeddings_trainable \
			--no-modelconfig_base_hidden_embeddings_trainable \
			--sequenceconfig_x_sequence_column_name log_cluster_template\
			--textualpapermodelconfig_num_filters 100 \
			--sequenceconfig_y_sequence_column_name attributes \
			--sequenceconfig_max_window_size 10 \
			--sequenceconfig_min_window_size 10 \
			--experimentconfig_multilabel_classification \
			--sequenceconfig_flatten_y \
			--sequenceconfig_flatten_x \
			${ARGS} ; \
	done ; \
