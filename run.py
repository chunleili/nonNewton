import os

# # profile mpm
# os.system("python -m cProfile -o mpm.profile .\MPM\main.py")
# os.system("snakeviz mpm.profile")

# run mpm
os.system("python .\MPM\main.py --save_results 1 --enable_plot 1 --num_steps=10")
