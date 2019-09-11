from tensorboard import program
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', '../../logs/ResNet/scalars/20190818-191529'])
url = tb.launch()
while True:
    pass
