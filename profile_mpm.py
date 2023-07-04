import os

# profile mpm
os.system("python -m cProfile -o mpm.profile .\MPM\main.py")
os.system("snakeviz mpm.profile")
