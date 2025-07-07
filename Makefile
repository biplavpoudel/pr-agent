install:
	uv pip install --upgrade pip && \
		uv pip install -r requirements.txt && \
		uv add -r requirements.txt

lint:
	pylint --disable=R,C,W *.py ./agent/*.py

format:
	pyink ./agent/*.py *.py

all:
	install lint format