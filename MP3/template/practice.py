dict={'a':(0,1),'b':(1,2)}
keys=list(dict.keys())
values=list(dict.values())
parents=[]
for i in values:
    parents.append(i[0])

ind=parents.index(0)
print(keys[ind])