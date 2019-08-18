from tensorboard import program
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', 'logs/scalars/20190815-171720'])
url = tb.launch()
while True:
    pass
