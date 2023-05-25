with open(input("Enter a file name: "), 'r') as file:
    line = file.readline()
    listPoints = []
    while file and line:
        line = line.strip()
        line_sep = line.split(" ")
        digits = [int(x) for x in line_sep if x.isdigit()]
        listPoints.append((digits[1], digits[2]))
        print(listPoints)
        line = file.readline()
