.PHONY: lint test run

run:
	python render_shorts.py --input sample.mp4 --outdir outputs --max_shorts 3 --force_cpu

test:
	pytest -q

fmt:
	ruff check --fix .
	black .
