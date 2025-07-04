install:
	uv pip install --upgrade pip && \
		uv pip install -r requirements.txt

lint:
	pylint --disable=R,C,W *.py

format:
	pyink *.py

docformatter:
	docformatter --in-place *.py

all:
	install lint format