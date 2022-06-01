# Copyright (c) 2021-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

MAX_LINE_LENGTH = 79

build-pip:
	python -m build

build-clean:
	rm -r dist
	rm -r build
	rm -r prob_forest.egg-info 

black:
	black -l ${MAX_LINE_LENGTH} prob_forest test

black-check:
	black --check -l ${MAX_LINE_LENGTH} prob_forest test

isort:
	isort -l $(MAX_LINE_LENGTH) prob_forest test

isort-check:
	isort -c -l $(MAX_LINE_LENGTH) prob_forest test

mypy:
	mypy prob_forest test --disallow-untyped-defs

flake8:
	flake8 --max-line-length=$(MAX_LINE_LENGTH) prob_forest test

quality: black-check isort-check flake8 mypy

unit-test:
	pytest --disable-pytest-warnings --durations=0 ./test/unit

tests: unit-test
