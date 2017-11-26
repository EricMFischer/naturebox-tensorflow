import random
fid = open("members.csv", "r")
li = fid.readlines()
fid.close()

random.shuffle(li)

fid = open("members_shuffled.csv", "w")
fid.writelines(li)
fid.close()
