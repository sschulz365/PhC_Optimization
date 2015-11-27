# runExperiment tests
import subprocess

for i in range (0,10):
    subprocess.call(['python', 'runExperiment.py']) # Just run the program
    print subprocess.check_output(['python', 'runExperiment.py']) # Also gets you the stdout
    print i
