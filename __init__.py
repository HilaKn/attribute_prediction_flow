from collections import defaultdict

dict = defaultdict(lambda: defaultdict(int))

dict["noun"] = 5
test = dict["unknown"]

print "{}".format(test)