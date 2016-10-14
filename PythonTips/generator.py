'''
之前有同学提到说，想知道更多关于Generator的东西。

那么这一期就给大家讲一讲Generator是什么？为什么要使用Generator？
'''

# def example1():
# 	def generator_function():
# 		yield 1
# 		yield 2
# 		return 3
#
# 	generatorObject = generator_function()
# 	print(next(generatorObject))
# 	print(next(generatorObject))
#
# 	try:
# 		print(next(generatorObject))
# 	except:
# 		pass
#
# def example2():
# 	def generator_function(number):
# 		yield 1
# 		yield 2
# 		return 3
#
# 	generatorObject = generator_function()
# 	print(next(generatorObject))
# 	print(next(generatorObject))
#
# 	try:
# 		print(next(generatorObject))
# 	except:
# 		pass
#
#
# example1()
#
#
#
#
#
# # Event Loop
# class Event():
#
#
#
# loop = []
# for event in loop:
# 	handle(event)
#

import socket
import time

def get(path):
	s = socket.socket()
	s.connect(('localhost', 3000))

	request = 'GET %s HHTP/1.0\r\n\r\n' % path
	s.send(request.encode())

	chunks = []
	while True:
		chunk = s.recv(1000)
		if chunk:
			chunks.append(chunk)
		else:
			body = (b''.join(chunks)).decode()
			print('--------------------------------------')
			print(body)
			print('--------------------------------------\n\n')
			return

start = time.time()
get('/super-slow')
get('/super-slow')
get('/super-slow')
print('%.1f sec' % (time.time() - start))












#
