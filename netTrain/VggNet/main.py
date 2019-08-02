from netTrain.VggNet.net_model import VGG16

















def main():


    VGG16(include_top=include_top, input_tensor=input_tensor,
          input_shape=input_shape, pooling=pooling,
          weights=weights, classes=classes)
    pass


if __name__ == '__main__':
    main()