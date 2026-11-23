PY = .venv/bin/python -I
all:
	$(PY) tests/test_core.py
	for s in fig_pipeline fig_alignment fig_rtk fig_ekf fig_ekf_paper fig_noise fig_control fig_logs fig_gifs export_payload results; do $(PY) src/$$s.py || exit 1; done
