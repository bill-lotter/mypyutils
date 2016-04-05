import os

def get_next_num(base_str):
	r = 0
	while(os.path.isdir(base_str + str(r))):
		r = r+1
	return r
