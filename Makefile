.PHONY: run train train-demo build-demo-data install

run:
	APOGEE_DATASET=demo-data python3 -m streamlit run app/Home.py

train:
	APOGEE_DATASET=full-data python3 model/train.py

train-demo:
	APOGEE_DATASET=demo-data python3 model/train.py

build-demo-data:
	python3 model/build_demo_data.py

install:
	pip3 install -r requirements.txt
