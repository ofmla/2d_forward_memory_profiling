serial.solver: generate_shot_data.py
	mprof run --include-children python generate_shot_data.py --serial --solver
	
serial.operator: generate_shot_data.py
	mprof run --include-children python generate_shot_data.py --serial --no-solver
	
dask.solver: generate_shot_data.py
	mprof run --include-children python generate_shot_data.py --no-serial --solver
	
dask.operator:
	mprof run --include-children python generate_shot_data.py --no-serial --no-solver
