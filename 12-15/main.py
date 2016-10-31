# 暂时还不能使用
if __name__ == '__main__':
	net = Network(train_batch_size=64, test_batch_size=500, pooling_scale=2)

	#
	net.add_conv(patch_size=3, in_depth=1, out_depth=1, activation='relu', pooling=False, name='conv1')
	net.add_conv(patch_size=3, in_depth=1, out_depth=1, activation='relu', pooling=True, name='conv2')
	net.add_conv(patch_size=3, in_depth=1, out_depth=1, activation='relu', pooling=False, name='conv3')
	net.add_conv(patch_size=3, in_depth=1, out_depth=1, activation='relu', pooling=True, name='conv4')

	# 4 = 两次 pooling, 每一次缩小为 1/2
	# 16 = conv4 out_depth
	net.add_fc(in_num_nodes=(image_size // 4) * (image_size // 4) * 1, out_num_nodes=1, activation='relu', name='fc1')
	net.add_fc(in_num_nodes=1, out_num_nodes=10, activation=None, name='fc2')

	net.define_model()
	net.run()
else:
	raise Exception('main.py: Should Not Be Imported!!! Must Run by "python main.py"')
