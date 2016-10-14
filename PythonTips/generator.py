'''
之前有同学提到说，想知道更多关于Generator的东西。

那么这一期就给大家讲一讲Generator是什么？为什么要使用Generator？
'''

def example1():
	def generator_function():
		yield 1
		yield 2
		return 3

	generatorObject = generator_function()
	print(next(generatorObject))
	print(next(generatorObject))

	try:
		print(next(generatorObject))
	except:
		pass

def example2():
	def generator_function(number):
		yield 1
		yield 2
		return 3

	generatorObject = generator_function()
	print(next(generatorObject))
	print(next(generatorObject))

	try:
		print(next(generatorObject))
	except:
		pass


example1()





# Event Loop
class Event():



loop = []
for event in loop:
	handle(event)















#
