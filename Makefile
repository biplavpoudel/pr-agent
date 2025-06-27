install:
	pip install --upgrade pip && \
		pip install -r requirements.txt

lint:
	pylint --disable=R,C,W *.py

format:
	pyink *.py

docformatter:
	docformatter --in-place tools.py

all:
	install lint format