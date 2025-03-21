.PHONY: test test-unit test-integration test-all test-safe

test-unit:
	pytest tests/preprocessing tests/models tests/novelty -v

test-integration:
	pytest tests/test_full_pipeline.py tests/evaluation -v

test-all:
	pytest tests -v

test-safe:
	pytest tests/preprocessing tests/models tests/novelty -v

test: test-unit
