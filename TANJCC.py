# !/usr/bin/env python2.7
# -*- coding: utf-8 -*-
import os, sys


"""
	Archivo con las funciones usadas en la asignatura
	Teoría Algebráica de Números y Criptografía
"""

from sympy import Symbol
from sympy import gcdex

hexa_to_dec_dict = {str(x): x for x in range(0,10)}
hexa_to_dec_dict.update({'a': 10, 'b':11, 'c':12, 'd':13, 'e':14, 'f':15})

dec_to_hex_dict = {x: str(x) for x in range(0,10)}
dec_to_hex_dict.update({ 10:'a', 11:'b', 12:'c', 13:'d', 14:'e', 15:'f'})

def elemento( hexastring, i):
	return hexa_to_dec_dict[hexastring[i]]

def hextodec_it(hexastring):
	result = 0
	for i in range (len(hexastring)):
		result += elemento(hexastring, len(hexastring)-1-i) * pow(16, i)

	return result

def hextodec_rec(hexastring):
	if ( len(hexastring) == 1):
		return hexa_to_dec_dict[hexastring]
	else:
		return elemento(hexastring, len(hexastring)-1) + 16*hextodec_rec(hexastring[:len(hexastring)-1])

def dectohex_rec(entero):
	if ( entero < 16 ):
		return dec_to_hex_dict[entero]
	else:
		return dectohex_rec(int(entero / 16)) + dec_to_hex_dict[int(entero % 16)]

def lista_palabras(mensaje):
	word_list = []
	word = ''
	for letter in mensaje:
		if(letter == ' '):
			word_list.append(word)
			word = ''
		else:
			word += letter

	word_list.append(word)

	return word_list

def print_list(list):
	for word in lista:
		print word

def print_list_as_string(lista):
	string = ''
	for word in lista:
		string += word

	print string

def str_to_hexalist(mensaje):
	hexalist = []
	word_list = lista_palabras(mensaje)
	for word in word_list:
		hexalist.append(word.encode('hex'))

	return hexalist

def hexa_to_str(hexa):

	if( len(hexa)%2 == 1):
		hexa = '0' + hexa

	return hexa.decode('hex')

def hexalist_to_str(lista):
	string = ''
	for i in range(0, len(lista)-1):
		string += hexa_to_str(lista[i])
		string += ' '

	string += hexa_to_str(lista[len(lista)-1])
	return string

def str_to_declist(mensaje):
	declist = []
	hexalist = str_to_hexalist(mensaje)
	for hexa in hexalist:
		declist.append(hextodec_rec(hexa))

	return declist

def cod_afin_number(x,a,b,n):
	return (a*x+b)%n

def cod_afin(mensaje,a,b,n):
	assert gcdex(a,n)[2] == 1, 'Las claves a y n deben ser primos relativos'

	number_list = []
	decdoded_list = str_to_declist(mensaje)
	for number in decdoded_list:
		number_list.append(cod_afin_number(number,a,b,n))

	return number_list

# coded_list = cod_afin('Hola, ¿qué tal?',4892055594575155744537,1231241,465675465116607065549*44211790234832169331)
# print coded_list

def deco_afin_number(number, a, b, n):
	return (gcdex(a,n)[0] * (number - b)) % n

def deco_afin(lista,a,b,n):
	assert gcdex(a,n)[2] == 1, 'Las claves a y n deben ser primos relativos'

	decoded_list = []
	for number in lista:
		decoded_number = deco_afin_number(number,a,b,n)
		decoded_list.append(dectohex_rec(decoded_number))

	return hexalist_to_str(decoded_list)

# deco_afin(coded_list,4892055594575155744537,1231241,465675465116607065549*44211790234832169331)

mensaje_cifrado = [734461933241186429117476L,
 3435209764L,
 279048452422856320008528932L,
 1033071336087238299690020L,
 974633772068L,
 57111833174811684L,
 15623803945172673572L,
 39003172L,
 60467473942520868L,
 3668513721263858590756L,
 3400737828L,
 17334517580104451101254330950692L,
 38741028L,
 60467473942520868L,
 3779622034147486278692L,
 39265316L,
 60467473942520868L,
 264466043694263907773391908L,
 39527460L,
 60467473942520868L,
 1011702180066720474247122611544859684L,
 422555755556L,
 60467473942520868L,
 3596316486929846969380L,
 43721764L,
 150370742641700L,
 63301989280982052L,
 974633772068L,
 3702727716L,
 16776694611586262052L,
 58761092191167524L,
 931014845476L,
 4072327277505430889508L,
 854174540836L,
 65516448682550308L,
 279010673496068621976282148L,
 974633772068L,
 16195758769579435044L,
 974633772068L,
 240419345292453720492024868L,
 245091883754532L,
 218660688569380L,
 3435471908L,
 61056768755442724L,
 3534955556L,
 58760954251125796L,
 75164731049239781261359916068L]
n = 553612260071847767819357303824754235825777

counting = dict((i,mensaje_cifrado.count(i)) for i in mensaje_cifrado)
coded_kilos = max(counting, key=counting.get)
decoded_kilos = hextodec_rec("kilos".encode('hex'))

coded_email = 268352937076140713164752723958686406250826
decoded_email = hextodec_rec("jacinto@gmail.com".encode('hex'))

a = deco_afin_number(coded_kilos - coded_email, decoded_kilos - decoded_email, 0, n)
b = (coded_kilos - a * decoded_kilos) % n

print deco_afin(mensaje_cifrado,a,b,n)
