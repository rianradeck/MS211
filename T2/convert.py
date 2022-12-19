f = open('in.in', 'r', encoding="utf-8")

ans = []
try:
    for s in f:
        l = s.split()
        if len(l) > 0:
            ans.append(l)
except:
    print("EOF")

for i in range(len(ans)):
    for j in range(len(ans[i])):
        ans[i][j] = float(ans[i][j])

print(ans)