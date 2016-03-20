# !/usr/bin/env python2.7
# -*- coding: utf-8 -*-
import os, sys

"""
	Archivo con las funciones usadas en la asignatura
	Teoría Algebráica de Números y Criptografía
"""
"""
	Funciones Práctica factor base a partir de la línea 268
"""
from sympy import *
from random import randint
from sympy.ntheory import jacobi_symbol
from itertools import combinations
from math import floor, sqrt

"""
	Prácticas iniciales
"""

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

"""
	Práctica Codificación Afín
"""

def cod_afin_number(x,a,b,n):
	return (a*x+b)%n

def cod_afin(mensaje,a,b,n):
	assert gcdex(a,n)[2] == 1, 'Las claves a y n deben ser primos relativos'

	number_list = []
	decdoded_list = str_to_declist(mensaje)
	for number in decdoded_list:
		number_list.append(cod_afin_number(number,a,b,n))

	return number_list

def deco_afin_number(number, a, b, n):
	return (gcdex(a,n)[0] * (number - b)) % n

def deco_afin(lista,a,b,n):
	assert gcdex(a,n)[2] == 1, 'Las claves a y n deben ser primos relativos'
	decoded_list = []
	for number in lista:
		decoded_number = deco_afin_number(number,a,b,n)
		decoded_list.append(dectohex_rec(decoded_number))

	return hexalist_to_str(decoded_list)

def solve_congr(a,b,n):
	if(gcdex(a,n)[2] == 1):
		return (gcdex(a,n)[0] * b) % n
	else:
		a_prime = a / gcdex(a,n)[2]
		b_prime = b / gcdex(a,n)[2]
		n_prime = n / gcdex(a,n)[2]
		return (gcdex(a_prime,n_prime)[0] * b_prime ) % n_prime

"""
 	Práctica RSA
"""


def decode_vigenere(board, coded_string, key):
	decoded_string = ''
	first_col = [row[0] for row in board]
	for letter in range(len(coded_string)):
		row = first_col.index(key[letter%(len(key))])
		decoded_string += board[0][board[row].index(coded_string[letter])]

	print decoded_string

def code_vigenere(board, string, key):
	coded_string = ''
	first_col = [row[0] for row in board]
	for letter in range(len(string)):
		row = first_col.index(string, [letter%(len(key))])
		col = board[0].index(string, [letter%(len(key))])
		coded_string += board[row][col]

	print coded_string

def deco_rsa(lista,e,p,q):
	u = gcdex(e,(p-1)*(q-1))[0] % ((p-1)*(q-1))
	decoded_list = []
	n = p*q
	for number in lista:
		decoded_number = pow(number,int(u),n)
		decoded_list.append(dectohex_rec(decoded_number))

	return hexalist_to_str(decoded_list)

"""
	Práctica Primaridad
"""

def psp(n):
	b = randint(2, n-1)
	d = gcdex(b, n)[2]
	if( d != 1):
		print "gcd(",b,",",n,")=", d , " es divisor de", n
		return [b, False]
	else:
		v = pow(b, n-1, n)
		return [b, v == 1]

def pspk(n, k):
	i = 1
	psp_n = psp(n)
	posible_prime = psp_n[1]
	list_b = []
	while( i in range(1,k) and posible_prime):
		psp_n = psp(n)
		posible_prime = psp_n[1]
		i = i+1
		list_b.append(psp_n[0])

	if(posible_prime):
		print "Es posible que ", n, " sea primo"
		psp_n[0] = list_b
	return psp_n

def epsp(n):
	if( n%2 == 0):
		print n, " es par"
		return [2, False]
	else:
		b = randint(3, n-1)
		d = gcdex(b, n)[2]
		if( d != 1):
			print "gcd(",b,",",n,")=", d , " es divisor de", n
			return [b, False]
		else:
			return [b, pow(b, (n-1)/2, n) == jacobi_symbol(b,n)%n]

def epspk(n, k):
	i = 1
	epsp_n = epsp(n)
	posible_prime = epsp_n[1]
	list_b = []
	while( i in range(1,k) and posible_prime):
		epsp_n = epsp(n)
		posible_prime = epsp_n[1]
		i = i+1
		list_b.append(epsp_n[0])

	if( posible_prime ):
		print "Es posible que ", n, " sea primo"
		epsp_n[0] = list_b
	return epsp_n

def mpot(p,m):
	s = 0
	while( m%p == 0 ):
		m = m/p
		s = s + 1

	return [s, m]

def fpsp(n):
	if( n%2 == 0):
		return [2, False]
	else:
		[t,s] = mpot(n-1, 2)
		b = randint(2, n-1)
		d = gcdex(b, n)[2]
		if( d != 1):
			print "gcd(",b,",",n,")=", d , " es divisor de", n
			return [b, False]
		elif( pow(b,t,n) == 1 or pow(b,t,n) == n-1):
			return [b, True]
		else:
			for i in range(1,s):
				if( pow(b,t*2^i,n) == n-1):
					return [b, True, i]
			return [b, False]

def fpspk(n, k):
	i = 1
	fpsp_n = fpsp(n)
	posible_prime = fpsp_n[1]
	list_b = []
	while( i in range(1,k) and posible_prime):
		fpsp_n = fpsp(n)
		posible_prime = fpsp_n[1]
		i = i+1
		list_b.append(fpsp_n[0])

	if( posible_prime ):
		print "Es posible que ", n, " sea primo"
		fpsp_n[0] = list_b

	return fpsp_n

"""
		PRÁCTICA FACTOR BASE
"""

def abmod(x, n):
	mod = x%n
	if( mod <= n/2 ):
		return mod
	else:
		return mod-n

def mayorpot(x, p):
	assert x != 0 and p!=0, "No se puede tratar el 0 en este método"
	if( p == -1):
		if( x < 0 ):
			return 1
		else:
			return 0
	else:
		pot = 0
		while( x%p == 0 ):
			x = x/p
			pot += 1

		return pot

def bnumer(b, base, n):
	x = 1
	y = abmod(b**2, n)

	if( y!= 0):
		exps = [mayorpot(y,p) for p in base]
		for i in range(len(base)):
			x *= base[i]**exps[i]

	return (x==y)

def vec_alfa(b,base,n):
	if( bnumer(b,base,n) ):
		y = abmod(b**2, n)
		exps = [mayorpot(y,p) for p in base]
		return exps
	else:
		print str(b)+ " no es un B-número"

def parp(lista):
	son_par = True
	for exp in lista:
		son_par = son_par and (exp%2 == 0)

	return son_par

def ssuma(lista1, lista2):
	assert len(lista1)==len(lista2), "Los vectores no tienen la misma longitud"
	return [lista1[i]+lista2[i] for i in range(len(lista1))]

def suma_listas(lists):
	vector_sum = [0] * len(lists[0])
	for l in lists:
		vector_sum = ssuma(vector_sum,l)
	return vector_sum

def aux(k, r):
	return list(combinations(range(k),r))

def suma(lista, k):
	return [suma_listas([lista[j] 	for j in comb])
									for comb in aux(len(lista),k)]

def bi(n, k_max, i_max, base):
	l1 = [floor(sqrt(j*n)) for j in range(1,k_max+1)]
	l2 = [n_k + j for j in range(i_max) for n_k in l1]
	BN = list(set([int(i) for i in l2 if bnumer(i, base, n)]))
	return BN

def getSumajPar(alfavec, j, blength):
	while(j <= blength):
		sumaj = suma(alfavec, j)
		sumajpar = [i for i in sumaj if parp(i)]
		if(len(sumajpar) > 0):
			return[True,sumajpar,j,sumaj]
		j += 1

	return [False,[],blength,[]]

def soleq(n, h, k_max, i_max):
	base_factor = [-1]+[prime(i) for i in range(1,h+1)]
	BN = bi(n, k_max, i_max, base_factor)
	alfavec = [vec_alfa(b,base_factor,n) for b in BN]
	found = False
	j = 1
	while j <= h+1 and not found:
		sjp = getSumajPar(alfavec, j, h+1)
		if sjp[0]:
			possible_alphas = sjp[1]
			j = sjp[2]
			suma_j = sjp[3]
			print len(possible_alphas)
			for alpha in possible_alphas:
				eles_alpha = aux(len(BN),j)[suma_j.index(alpha)]
				t_factors = [BN[i] for i in eles_alpha]
				s_factors = [base_factor[i]**(alpha[i]/2) for i in range(h)]
				t = prod(t_factors)
				s = prod(s_factors)
				if( (t%n) != (s%n) and (t%n) != (-s)%n):
					print( str(t) + " y " + str(s) +
						" es una solución no trivial de la ecuación")
					d_1 = gcd(n,t-s)
					return([n/d_1, d_1])
			j += 1
			print j
		else:
			print("Todas las soluciones son triviales")
			return False

def sumDictionaries(dict_a,dict_b):
	return { k: dict_a.get(k, 0) + dict_b.get(k, 0)
       for k in set(dict_a) | set(dict_b) }

def fac(n, h, k_max, i_max):
	factors_dict = {}
	factorized = False

	while not factorized:
		sol = soleq(n,h,k_max,i_max)
		print sol
		if(sol == False):
			print("No se ha encontrado una factorización")
			return {n:1}
		else:
			factorized = True
			if isprime(sol[0]) or sol[0]==1:
				dict_a = {sol[0]: 1}
			else:
				dict_a = fac(sol[0],h,k_max, i_max)

			if isprime(sol[1]) or sol[1]==1:
				dict_b = {sol[1]: 1}
			else:
				dict_b = fac(sol[1],h,k_max, i_max)

			return sumDictionaries(dict_a,dict_b)

# print soleq(186,30,10,10)
# print soleq(32056356,30,10,10)
# print fac(32056356,30,10,10)

	# Elección de la base. Fracciones continuas

def flatten(x):
	flat = []
	for l in x:
		for item in l:
			flat.append(item)
	return flat

def getBase(factorized_list):
	base = [-1]

	# Conteo de las apariciones. Añadimos los factores que aparezcan más de una vez
	factor_list = [f for f in d for d in factorized_list
	count_appearances = dict((i,factor_list.count(i)) for i in factor_list)

	base += [i for i in count_appearances if count_appearances[i]>1]

	#Contamos las apariciones con exponente par. a
	even_factor = [f for f in d if d[f]%2==0 for d in factorized_list]
	count_appearances = dict((i,factor_list.count(i)) for i in factor_list)
	
