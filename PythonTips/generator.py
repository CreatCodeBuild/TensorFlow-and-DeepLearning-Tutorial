'''
之前有同学提到说，想知道更多关于Generator的东西。

那么这一期就给大家讲一讲Generator是什么？为什么要使用Generator？
'''

def example1():

	def generator_function():
		yield 1		# 中文里 有产出 或者 让步
		yield 2
		return 3

	generatorObject = generator_function()
	print(next(generatorObject))
	print(next(generatorObject))
	try:
		print(next(generatorObject))
	except:
		print('We have reached the end of iteration')

def example2():
	def generator_function(number):
		while number > 0:
			yield number
			number -= 1
	for number in generator_function(10):
		print(number)

def example3():
	def fun():
		return 'fun'
	def generator_function(number):
		if number > 0:
			yield number
			print(fun())
			yield from generator_function(number-1)
	for number in generator_function(10):
		print(number)

# example1()
# example2()
# example3()


'''
Generator
Async Operation -> I/O Bound
				-> Memory Efficiency
'''

#
# # Event Loop
# class Event():
# 	pass
#
#
# class EventLoop():
# 	pass
#
#
# class Future():
# 	pass
#
#
#
import socket
import time
from selectors import DefaultSelector, EVENT_WRITE, EVENT_READ

selector = DefaultSelector()
n_task = 0

class Future:
	def __init__(self):
		self.callbacks = []

	def resolve(self):
		for func in self.callbacks:
			func()

class Task:	# responsible for calling next() on generators
	def __init__(self, gen):
		self.gen = gen
		self.step()

	def step(self):
		try:
			f =	next(self.gen)
		except StopIteration:
			return
		f.callbacks.append(self.step)



def get(path):
	s = socket.socket()
	s.connect(('localhost', 3000))

	request = 'GET %s HHTP/1.0\r\n\r\n' % path
	s.send(request.encode())

	chunks = []
	while True:
		chunk = s.recv(1000) # 1000 bytes
		if chunk:
			chunks.append(chunk)
		else:
			body = (b''.join(chunks)).decode()
			print('--------------------------------------')
			print(body)
			print('--------------------------------------\n\n')
			return

def async_get(path):
	global n_task
	n_task += 1

	s = socket.socket()
	s.setblocking(False)
	try:
		s.connect(('localhost', 3000))
	except BlockingIOError as e:
		print(e)

	request = 'GET %s HHTP/1.0\r\n\r\n' % path

	f = Future()
	selector.register(s.fileno(), EVENT_WRITE, data=f)

	yield f	# halt function state until s is writable
	selector.unregister(s.fileno())
	# s is writable
	s.send(request.encode())

	chunks = []


	while True:
		f = Future()
		selector.register(s.fileno(), EVENT_READ, data=f)
		yield f
		# by now, s is readable
		selector.unregister(s.fileno())
		chunk = s.recv(1000) # 1000 bytes
		if chunk:
			chunks.append(chunk)
		else:
			body = (b''.join(chunks)).decode()
			print('--------------------------------------')
			print(body)
			print('--------------------------------------\n\n')
			n_task -= 1
			return


start = time.time()
# get('/slow')
# get('/super-slow')
# get('/slower')
Task(async_get('/slow'))
Task(async_get('/slow'))
Task(async_get('/slow'))

while n_task:
	events = selector.select()
	for event, mask in events:
		fut = event.data
		fut.resolve()

print('%.1f sec' % (time.time() - start))












#
