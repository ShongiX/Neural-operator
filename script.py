import os

for number in range(10):
    os.system('start cmd /K python generate.py --number ' + str(number))
