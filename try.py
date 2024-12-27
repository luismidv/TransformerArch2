frase = "luis tiene un perro"
for word in frase:
    print(word)


list = ['luismi','luismi','luismi','luismi','luismi','luismi','luismi','luismi','luismi', ]
import re



counter = 0

for word in list :
    list[counter] = regTokenize(word)

print(list)

