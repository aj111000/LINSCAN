import csv

def import_test():
    with open('eq_test.dat', 'r', newline='\n') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ')
        output = []
        for row in spamreader:
            row = list(filter(lambda x: x != '', row))
            output.append(list(map(float, row)))

    return output


def import_hs():
    with open('eq_hs.utm', 'r', newline='\n') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ')
        output = []
        for row in spamreader:
            row = list(filter(lambda x: x != '', row))
            output.append(list(map(float, row))[:2])

    return output