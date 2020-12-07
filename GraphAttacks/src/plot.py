import matplotlib.pyplot as plt
import sys
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--file', default=False)
parser.add_argument('--title', default="MAP vs #flips")
parser.add_argument('--xlabel', default="#flips")
parser.add_argument('--ylabel', default="MAP")

av=parser.parse_args()

file=sys.argv[1]
# print(file)
budgets=[]
values=[]
# values_names=[]
with open(av.file,"r") as rfile:
	budgets=list(map(int,rfile.readline().strip().split(" ")))
	plot_labels=str(rfile.readline().strip()).split(" ")
	for _ in range(len(plot_labels)):
		values.append(list(map(float,rfile.readline().strip().split(" "))))

print(budgets)
# plot_labels
for i in range(len(values)):
	plt.plot(budgets,values[i],label=plot_labels[i])
title=av.title
xlabel=av.xlabel
ylabel=av.ylabel
plt.title(title)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.legend()
plt.show()


