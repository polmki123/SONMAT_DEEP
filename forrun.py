import os 
path = './200/'
for filename in os.listdir(path):
	os.rename(path+filename, path+filename[:-3] + str('png'))