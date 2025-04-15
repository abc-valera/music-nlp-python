start:
	PYTHONPYCACHEPREFIX=local/__pycache__ python src/main.py

archive:
	PYTHONPYCACHEPREFIX=local/__pycache__ python archive/main.py

download_maestro_dataset:
	wget https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip
	unzip -o -d ${DATASET_FOLDER_PATH} maestro-v3.0.0-midi.zip

new_env:
	python -m venv local/venv


.PHONY: start archive download_maestro_dataset new_env	