from tensorboard import program
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', 'logs/scalars'])
url = tb.launch()
while True:
    pass
