all: test

clean: FORCE
	git clean -dfx -e brmp.egg-info

docs: FORCE
	$(MAKE) -C docs html

install: FORCE
	pip install -e .

format: FORCE
	isort -rc .

lint: FORCE
	flake8

test: lint FORCE
	pytest -vs tests --tb=short

FORCE:
