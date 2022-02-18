shots.marmousi: generate_shot_data.py
	python generate_shot_data.py marmousi

shots.marmousi2: generate_shot_data.py
	python generate_shot_data.py marmousi2
	$(eval marmousi2_shots=$(shell ls ./marmousi2/shots/))
	@echo generated shots $(marmousi2_shots)

shots.overthrust: generate_shot_data.py
	python generate_shot_data.py overthrust

fwi.marmousi2: inversion.py
	python inversion.py marmousi2 LBFGS
	
lsrtm.marmousi: inversion.py
	python inversion.py marmousi LBFGS
	python inversion.py marmousi PSTD
	python inversion.py marmousi PNLCG

lsrtm.overthrust: inversion.py
	python inversion.py overthrust LBFGS
	python inversion.py overthrust PSTD
	python inversion.py overthrust PNLCG
