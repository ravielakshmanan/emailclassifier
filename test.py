import string

s = "l*ots! o(f. p@u)n[c}t]u[a'ti\"on#$^?/"
table = str.maketrans({key: None for key in string.punctuation})
str = s.translate(table)
print(str)

new_set = set(['Jane', 'Marvin', 'Janice', 'John', 'Jack', 'Jane'])
print(new_set)

a = {}

a[("a", "s")] = 2
a[("b", "s")] = 3
a[("c", "h")] = 6
a[("a", "h")] = 4
a[("d", "h")] = 4

word_label_frequency_list = a.keys()
word_count_per_label_map = {}

for item in word_label_frequency_list:
    word, label = item[0], item[1]
    count = word_count_per_label_map.get(label, 0)
    word_count_per_label_map[label] = count + 1

print(word_count_per_label_map)

c = ["been", "b", "c"]
d = ["been", "b", "d"]

new_list = list(set(c).difference(d))
print(new_list)

e = {}
count_map = {}

e["s"]= "a"
e["s"]= "b"
e["s"]= "c"
e["h"]= "d"

print(e)

for item in e.keys():
    count = count_map.get(item, 0)
    count_map[item] = count + 1

print(count_map)

x = reduce(lambda x,y: x+y, range(49999951,50000000))

