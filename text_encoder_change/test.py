n=12
li = [10 ** 4] * (n + 1)
li[0] = 0
for i in range(n + 1):
    tem = 10 ** 4 + 1
    for j in range(i + 1):
        if j == 0:
            continue
        dif = i - j * j
        if i < j * j:
            break
        tem = min(li[dif], tem)
    li[i] = tem + 1
print(li)