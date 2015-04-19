# DSGD_MF
You can run the .py file with the following command:
spark-submit dsgd_mf.py <num_factors> <num_workers> <num_iterations> <beta_value> <lambda_value> <inputV_filepath> <outputW_filepath> <outputH_filepath>

For example,
spark-submit dsgd_mf.py 100 10 50 0.8 1.0 test.csv w.csv h.csv
