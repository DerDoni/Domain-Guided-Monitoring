# Usage:
# make install			# downloads miniconda and initializes conda environment
# make install_mimic	# downloads required mimic files from physionet (physionet credentialed account required)
# make server	  		# starts mlflow server at port 5000
# make run  			# executes main.py within the conda environment \
				  			example: make run ARGS="--experimentconfig_sequence_type huawei_logs"
# make run_huawei		# executes main.py within the conda environment for all knowledge types on huawei dataset

CONDA_ENV_NAME = lena
CONDA_URL = https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
CONDA_SH = Miniconda3-latest-Linux-x86_64.sh
CONDA_DIR = ~/miniconda3

DATA_DIR = data
KNOWLEDGE_TYPES = simple simple simple simple simple
COLUMN_NAME = fine_log_cluster_template medium_log_cluster_template coarse_log_cluster_template
ALL_COLUMNS = fine_log_cluster_template_drain medium_log_cluster_template_drain coarse_log_cluster_template_drain fine_log_cluster_template_spell medium_log_cluster_template_spell coarse_log_cluster_template_spell fine_log_cluster_template_nulog medium_log_cluster_template_nulog coarse_log_cluster_template_nulog
HUAWEI_LOGS = logs_aggregated_concurrent_2000.csv logs_aggregated_concurrent_20000.csv logs_aggregated_concurrent_200000.csv
ALGOS = spell nulog drain
NULOG_SPELL = fine_log_cluster_template_spell medium_log_cluster_template_spell coarse_log_cluster_template_spell fine_log_cluster_template_nulog medium_log_cluster_template_nulog coarse_log_cluster_template_nulog

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

server:
	@echo "Starting MLFlow UI at port 5000"
	PATH="${PATH}:$(shell pwd)/${CONDA_DIR}/miniconda3/envs/${CONDA_ENV_NAME}/bin" ; \
	${CONDA_DIR}/envs/${CONDA_ENV_NAME}/bin/mlflow server --gunicorn-opts -t180

notebook:
	@echo "Starting Jupyter Notebook at port 8888"
	PATH="${PATH}:$(shell pwd)/${CONDA_DIR}/miniconda3/envs/${CONDA_ENV_NAME}/bin" ; \
	${CONDA_DIR}/envs/${CONDA_ENV_NAME}/bin/jupyter notebook notebooks/ --no-browser 

run: 
	./${CONDA_DIR}/miniconda3/envs/${CONDA_ENV_NAME}/bin/python main.py ${ARGS}

run_attention:
	for knowledge_type in ${KNOWLEDGE_TYPES} ; do \
	echo "Starting experiment for huawei_logs with knowledge type " $$knowledge_type "....." ; \
		${CONDA_DIR}/envs/${CONDA_ENV_NAME}/bin/python main.py \
			--experimentconfig_model_type $$knowledge_type \
			--huaweipreprocessorconfig_min_causality 0.01 \
			--huaweipreprocessorconfig_relevant_log_column fine_log_cluster_template \
			--no-modelconfig_base_feature_embeddings_trainable \
			--no-modelconfig_base_hidden_embeddings_trainable \
			--sequenceconfig_y_sequence_column_name attributes \
			--sequenceconfig_max_window_size 10 \
			--sequenceconfig_min_window_size 10 \
			--experimentconfig_multilabel_classification \
			--sequenceconfig_flatten_y \
			--sequenceconfig_flatten_x \
			--huaweipreprocessorconfig_log_parser drain \
			${ARGS} ; \
	done ; \



run_huawei:
	for knowledge_type in ${KNOWLEDGE_TYPES} ; do \
		for col_name in ${COLUMN_NAME} ; do \
			${CONDA_DIR}/envs/${CONDA_ENV_NAME}/bin/python main.py \
				--experimentconfig_sequence_type huawei_logs \
				--experimentconfig_model_type $$knowledge_type \
				--huaweipreprocessorconfig_min_causality 0.01 \
				--experimentconfig_batch_size 128 \
				--no-modelconfig_base_feature_embeddings_trainable \
				--no-modelconfig_base_hidden_embeddings_trainable \
				--sequenceconfig_y_sequence_column_name attributes \
				--sequenceconfig_x_sequence_column_name $$col_name \
				--huaweipreprocessorconfig_relevant_log_column $$col_name \
				--sequenceconfig_max_window_size 10 \
				--sequenceconfig_min_window_size 10 \
				--experimentconfig_multilabel_classification \
				--sequenceconfig_flatten_y \
				--modelconfig_rnn_type gru \
				--modelconfig_rnn_dim 200 \
				--modelconfig_embedding_dim 300 \
				--modelconfig_attention_dim 100 \
				--huaweipreprocessorconfig_log_parser drain \
				${ARGS} ; \
		done ; \
	done ; \


run_tbird:
	for knowledge_type in ${KNOWLEDGE_TYPES} ; do \
		for col_name in ${ALL_COLUMNS} ; do \
			${CONDA_DIR}/envs/${CONDA_ENV_NAME}/bin/python main.py \
				--experimentconfig_sequence_type tbird_logs \
				--experimentconfig_model_type $$knowledge_type \
				--experimentconfig_batch_size 128 \
				--no-modelconfig_base_feature_embeddings_trainable \
				--no-modelconfig_base_hidden_embeddings_trainable \
				--sequenceconfig_y_sequence_column_name attributes \
				--sequenceconfig_x_sequence_column_name $$col_name \
				--thunderbirdpreprocessorconfig_relevant_log_column $$col_name \
				--sequenceconfig_max_window_size 10 \
				--sequenceconfig_min_window_size 10 \
				--experimentconfig_multilabel_classification \
				--sequenceconfig_flatten_y \
				--modelconfig_rnn_type gru \
				--modelconfig_rnn_dim 200 \
				--modelconfig_embedding_dim 300 \
				--modelconfig_attention_dim 100 \
				--thunderbirdpreprocessorconfig_log_parser all \
				${ARGS} ; \
		done ; \
	done ; \

run_tbird_all:
	for knowledge_type in ${KNOWLEDGE_TYPES} ; do \
		for col_name in ${NULOG_SPELL} ; do \
			${CONDA_DIR}/envs/${CONDA_ENV_NAME}/bin/python main.py \
				--experimentconfig_sequence_type tbird_logs \
				--experimentconfig_model_type $$knowledge_type \
				--experimentconfig_batch_size 32 \
				--no-modelconfig_base_feature_embeddings_trainable \
				--no-modelconfig_base_hidden_embeddings_trainable \
				--sequenceconfig_y_sequence_column_name attributes \
				--sequenceconfig_x_sequence_column_name $$col_name \
				--thunderbirdpreprocessorconfig_relevant_log_column $$col_name \
				--sequenceconfig_max_window_size 10 \
				--sequenceconfig_min_window_size 10 \
				--experimentconfig_multilabel_classification \
				--sequenceconfig_flatten_y \
				--modelconfig_rnn_type gru \
				--modelconfig_rnn_dim 200 \
				--modelconfig_embedding_dim 300 \
				--modelconfig_attention_dim 100 \
				--thunderbirdpreprocessorconfig_log_parser all \
				${ARGS} ; \
		done ; \
	done ; \


run_tbird_attention:
	for knowledge_type in ${KNOWLEDGE_TYPES} ; do \
	echo "Starting experiment for huawei_logs with knowledge type " $$knowledge_type "....." ; \
		${CONDA_DIR}/envs/${CONDA_ENV_NAME}/bin/python main.py \
			--experimentconfig_sequence_type tbird_logs \
			--experimentconfig_model_type $$knowledge_type \
			--no-modelconfig_base_feature_embeddings_trainable \
			--no-modelconfig_base_hidden_embeddings_trainable \
			--sequenceconfig_y_sequence_column_name attributes \
			--sequenceconfig_max_window_size 10 \
			--sequenceconfig_min_window_size 10 \
			--experimentconfig_multilabel_classification \
			--sequenceconfig_flatten_y \
			--sequenceconfig_flatten_x \
			--thunderbirdpreprocessorconfig_log_parser all \
			--thunderbirdpreprocessorconfig_relevant_log_column fine_log_cluster_template_spell \
			${ARGS} ; \
	done ; \

run_bgl:
	for algo in ${ALGOS} ; do \
		for knowledge_type in ${KNOWLEDGE_TYPES} ; do \
			for col_name in ${COLUMN_NAME} ; do \
				${CONDA_DIR}/envs/${CONDA_ENV_NAME}/bin/python main.py \
					--experimentconfig_sequence_type bgl \
					--experimentconfig_model_type $$knowledge_type \
					--experimentconfig_batch_size 128 \
					--no-modelconfig_base_feature_embeddings_trainable \
					--no-modelconfig_base_hidden_embeddings_trainable \
					--sequenceconfig_y_sequence_column_name attributes \
					--sequenceconfig_x_sequence_column_name $$col_name \
					--bglpreprocessorconfig_relevant_log_column $$col_name \
					--sequenceconfig_max_window_size 10 \
					--sequenceconfig_min_window_size 10 \
					--experimentconfig_multilabel_classification \
					--sequenceconfig_flatten_y \
					--modelconfig_rnn_type gru \
					--modelconfig_rnn_dim 200 \
					--modelconfig_embedding_dim 300 \
					--modelconfig_attention_dim 100 \
					--bglpreprocessorconfig_log_parser $$algo \
					${ARGS} ; \
			done ; \
		done ; \
	done ; \

run_bgl_all:
	for knowledge_type in ${KNOWLEDGE_TYPES} ; do \
		for col_name in ${ALL_COLUMNS} ; do \
			${CONDA_DIR}/envs/${CONDA_ENV_NAME}/bin/python main.py \
				--experimentconfig_sequence_type bgl \
				--experimentconfig_model_type $$knowledge_type \
				--experimentconfig_batch_size 32 \
				--no-modelconfig_base_feature_embeddings_trainable \
				--no-modelconfig_base_hidden_embeddings_trainable \
				--sequenceconfig_y_sequence_column_name attributes \
				--sequenceconfig_x_sequence_column_name $$col_name \
				--bglpreprocessorconfig_relevant_log_column $$col_name \
				--sequenceconfig_max_window_size 10 \
				--sequenceconfig_min_window_size 10 \
				--experimentconfig_multilabel_classification \
				--sequenceconfig_flatten_y \
				--modelconfig_rnn_type gru \
				--modelconfig_rnn_dim 200 \
				--modelconfig_embedding_dim 300 \
				--modelconfig_attention_dim 100 \
				--bglpreprocessorconfig_log_parser all \
				${ARGS} ; \
		done ; \
	done ; \

run_bgl_attention:
	for knowledge_type in ${KNOWLEDGE_TYPES} ; do \
	echo "Starting experiment for huawei_logs with knowledge type " $$knowledge_type "....." ; \
		${CONDA_DIR}/envs/${CONDA_ENV_NAME}/bin/python main.py \
			--experimentconfig_sequence_type bgl \
			--experimentconfig_model_type $$knowledge_type \
			--no-modelconfig_base_feature_embeddings_trainable \
			--no-modelconfig_base_hidden_embeddings_trainable \
			--sequenceconfig_y_sequence_column_name attributes \
			--sequenceconfig_max_window_size 10 \
			--sequenceconfig_min_window_size 10 \
			--experimentconfig_multilabel_classification \
			--sequenceconfig_flatten_y \
			--sequenceconfig_flatten_x \
			--bglpreprocessorconfig_log_parser all \
			--bglpreprocessorconfig_relevant_log_column fine_log_cluster_template\
			${ARGS} ; \
	done ; \




run_hdfs:
	for knowledge_type in ${KNOWLEDGE_TYPES} ; do \
		for col_name in ${NULOG_SPELL} ; do \
			${CONDA_DIR}/envs/${CONDA_ENV_NAME}/bin/python main.py \
				--experimentconfig_sequence_type hdfs \
				--experimentconfig_model_type $$knowledge_type \
				--experimentconfig_batch_size 128 \
				--no-modelconfig_base_feature_embeddings_trainable \
				--no-modelconfig_base_hidden_embeddings_trainable \
				--sequenceconfig_y_sequence_column_name attributes \
				--sequenceconfig_x_sequence_column_name $$col_name \
				--hdfspreprocessorconfig_relevant_log_column $$col_name \
				--sequenceconfig_max_window_size 10 \
				--sequenceconfig_min_window_size 10 \
				--experimentconfig_multilabel_classification \
				--sequenceconfig_flatten_y \
				--modelconfig_rnn_type gru \
				--modelconfig_rnn_dim 200 \
				--modelconfig_embedding_dim 300 \
				--modelconfig_attention_dim 100 \
				--hdfspreprocessorconfig_log_parser all \
				${ARGS} ; \
		done ; \
	done ; \

run_hdfs_attention:
	for knowledge_type in ${KNOWLEDGE_TYPES} ; do \
	echo "Starting experiment for hdfs_logs with knowledge type " $$knowledge_type "....." ; \
		${CONDA_DIR}/envs/${CONDA_ENV_NAME}/bin/python main.py \
			--experimentconfig_sequence_type hdfs \
			--experimentconfig_model_type $$knowledge_type \
			--no-modelconfig_base_feature_embeddings_trainable \
			--no-modelconfig_base_hidden_embeddings_trainable \
			--sequenceconfig_y_sequence_column_name attributes \
			--sequenceconfig_max_window_size 10 \
			--sequenceconfig_min_window_size 10 \
			--experimentconfig_multilabel_classification \
			--sequenceconfig_flatten_y \
			--sequenceconfig_flatten_x \
			--hdfspreprocessorconfig_log_parser all \
			--hdfspreprocessorconfig_relevant_log_column fine_log_cluster_template_spell \
			${ARGS} ; \
	done ; \



run_timestamps:
	for knowledge_type in ${KNOWLEDGE_TYPES} ; do \
		for col_name in ${COLUMN_NAME} ; do \
			for log in ${HUAWEI_LOGS} ; do \
				${CONDA_DIR}/envs/${CONDA_ENV_NAME}/bin/python main.py \
					--experimentconfig_sequence_type huawei_logs \
					--experimentconfig_model_type $$knowledge_type \
					--experimentconfig_batch_size 128 \
					--no-modelconfig_base_feature_embeddings_trainable \
					--no-modelconfig_base_hidden_embeddings_trainable \
					--sequenceconfig_y_sequence_column_name attributes \
					--sequenceconfig_x_sequence_column_name $$col_name \
					--huaweipreprocessorconfig_relevant_log_column $$col_name \
					--sequenceconfig_max_window_size 10 \
					--sequenceconfig_min_window_size 10 \
					--experimentconfig_multilabel_classification \
					--sequenceconfig_flatten_y \
					--modelconfig_rnn_type gru \
					--modelconfig_rnn_dim 200 \
					--modelconfig_embedding_dim 300 \
					--modelconfig_attention_dim 100 \
					--huaweipreprocessorconfig_log_parser drain \
					--huaweipreprocessorconfig_aggregated_log_file data/$$log \
					--no-huaweipreprocessorconfig_remove_dates_from_payload  \
					${ARGS} ; \
			done ; \
		done ; \
	done ; \


run_timestamps_attention:
	for knowledge_type in ${KNOWLEDGE_TYPES} ; do \
		for log in ${HUAWEI_LOGS} ; do \
			echo "Starting experiment for huawei_logs with knowledge type " $$knowledge_type "....." ; \
			${CONDA_DIR}/envs/${CONDA_ENV_NAME}/bin/python main.py \
				--experimentconfig_model_type $$knowledge_type \
				--huaweipreprocessorconfig_min_causality 0.01 \
				--huaweipreprocessorconfig_relevant_log_column fine_log_cluster_template \
				--no-modelconfig_base_feature_embeddings_trainable \
				--no-modelconfig_base_hidden_embeddings_trainable \
				--sequenceconfig_y_sequence_column_name attributes \
				--sequenceconfig_max_window_size 10 \
				--sequenceconfig_min_window_size 10 \
				--experimentconfig_multilabel_classification \
				--sequenceconfig_flatten_y \
				--sequenceconfig_flatten_x \
				--huaweipreprocessorconfig_aggregated_log_file data/$$log \
				--no-huaweipreprocessorconfig_remove_dates_from_payload  \
				--huaweipreprocessorconfig_log_parser drain \
				${ARGS} ; \
		done ; \
	done ; \

run_anomaly_detection:
	for algo in ${ALGOS} ; do \
		for knowledge_type in ${KNOWLEDGE_TYPES} ; do \
			for col_name in ${COLUMN_NAME} ; do \
				${CONDA_DIR}/envs/${CONDA_ENV_NAME}/bin/python main.py \
					--experimentconfig_sequence_type bgl \
					--experimentconfig_model_type $$knowledge_type \
					--experimentconfig_batch_size 128 \
					--no-modelconfig_base_feature_embeddings_trainable \
					--no-modelconfig_base_hidden_embeddings_trainable \
					--sequenceconfig_y_sequence_column_name Label \
					--sequenceconfig_x_sequence_column_name $$col_name \
					--hdfspreprocessorconfig_relevant_log_column $$col_name \
					--sequenceconfig_max_window_size 10 \
					--sequenceconfig_min_window_size 10 \
					--experimentconfig_multilabel_classification \
					--sequenceconfig_flatten_y \
					--modelconfig_rnn_type gru \
					--modelconfig_rnn_dim 200 \
					--modelconfig_embedding_dim 300 \
					--modelconfig_attention_dim 100 \
					--hdfspreprocessorconfig_log_parser $$algo \
					${ARGS} ; \
			done ; \
		done ; \
	done ; \
