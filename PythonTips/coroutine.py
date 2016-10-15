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
Task(async_get('/slow'))
Task(async_get('/super-slow'))
Task(async_get('/slower'))

while n_task:
	events = selector.select()
	for event, mask in events:
		fut = event.data
		fut.resolve()

print('%.1f sec' % (time.time() - start))
