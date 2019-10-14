all: test

clean: FORCE
	git clean -dfx -e brmp.egg-info

install: FORCE
	pip install -e .

format: FORCE
	isort -rc .

lint: FORCE
	flake8

test: FORCE
	pytest -vs tests --tb=short

test-all: FORCE
	RUN_SLOW=1 pytest -vs tests --tb=short

FORCE:
