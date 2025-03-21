
#!/usr/bin/python3

# strings: Alpher numeric characters  a-z, 0-9, special characters

city =  "seoul"

#strings can be converted either to Upper or Lower case

print(city.upper())
number = 78

print(type(number))

name = "WILTON"

print(name.lower())

print(type(name))

# float,decimal point numbers

weight = 64.7812
print(weight)
print("%f" %weight)

print("%.1f" %weight)      #1 decimal place
print("%.2f" %weight)      #2 decimal place

# using .format to print floats

print ("{:.3f}".format(weight))

# using f string to print floats

print ("%f"%weight)

# boolean

is_married = True
is_late = False

print(is_married)
print(type(is_married))