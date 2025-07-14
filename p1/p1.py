from scipy.stats import norm, ttest_rel, t
import numpy as np

# 2a
f = open('./data.txt', 'r')
A = []
B = []

for line in f:
    line = line.strip().split()
    A.append(float(line[0]))
    B.append(float(line[1]))

a = np.array(A)
b = np.array(B)

mean_a = np.mean(a)
mean_b = np.mean(b)

std_a = np.std(a,ddof=1)
std_b = np.std(b,ddof=1)

n = len(a)
alpha = 0.05
tci_a = t.interval(1-alpha, n-1, mean_a, std_a/np.sqrt(n))
tci_b = t.interval(1-alpha, n-1, mean_b, std_b/np.sqrt(n))

print('Algorithm A CI:')
print(tci_a)
print()
print('Algorithm B CI:')
print(tci_b)
print()

# diff = a - b

# diff_tci = norm.interval(0.95, np.mean(diff), np.std(diff)/np.sqrt(12))
# print('Difference between A and B CI:')
# print(diff_tci)

# 2b
print(ttest_rel(a, b).confidence_interval)
