install:
	pip install -r requirements.txt

format:
	black .
	flake8 .

prepare:
	python main.py --prepare

train:
	python main.py --train

evaluate:
	python main.py --evaluate

clean:
	rm -rf __pycache__

all: install format prepare train evaluate

