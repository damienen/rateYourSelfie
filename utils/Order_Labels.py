import re
cMale = []
cFemale = []

with open("../Images/All_labels.txt") as fp:
        line = fp.readline()
        while line:
                if line[0] == 'C':
                        if line[1] == 'F':
                                cFemale.append(line)
                        else:
                                cMale.append(line)
                line = fp.readline()

def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)

sort_nicely(cFemale)
sort_nicely(cMale)


cFemale.extend(cMale)

print(cFemale)

newString = ""

for str in cFemale:
    newString += str

print(newString)

outFIle = open("../Images/All_labels.txt", "w")
outFIle.write(newString)
outFIle.close()
