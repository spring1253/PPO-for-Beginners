in_txt = 'log_original.txt'

i = 0

with open(in_txt, 'r') as file:
    for line in file:
        i += 1
        if i % 900 == 898:
            print((i//900+1)*100, line.split(": ")[1], end='')