# Train:Test = 800:200 으로 랜덤분할
import random

v = open('Violence.txt', 'r')
nv = open('NonViolence.txt', 'r')

train = open('trainlist.txt', 'w')
test = open('testlist.txt', 'w')

vlist = v.readlines()
random.shuffle(vlist)
nvlist = nv.readlines()
random.shuffle(nvlist)

for filename in vlist[0:800]:
    train.write(filename)

for filename in vlist[800:1000]:
    test.write(filename)

for filename in nvlist[0:800]:
    train.write(filename)

for filename in nvlist[800:1000]:
    test.write(filename)

v.close()
nv.close()

train.close()
test.close()