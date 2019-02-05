# Komputo

Komputo is an experimental library to provide 
a Kotlin idiomatic API to build, load and execute computation
graphs, initially using Tensorflow but potentially on other
backends. 

It relies on Tensorflow for Java and it 
contains an experimental DSL to build and execute 
computation graphs in a structured and type-safe manner:

```kotlin

val model = trainableModel {

    sequential(inputShape = I[28, 28]) {
        conv2d(
            filterCount = 32,
            kernelSize = I[3, 3],
            activation = RELU
        )
        conv2d(
            filterCount = 64,
            kernelSize = I[3, 3],
            activation = RELU
        )
        maxPool2D(windowSize = I[2, 2])
        flatten()
        dropout(0.25)
        dense(128, activation = RELU)
        dropout(0.5)
        dense(10)
    }

    training {
        loss = crossEntropyWithLogits
        optimizer = Adagrad()
    }

}

session(model) {

    train {

        dataset = trainDataset.mapLabels { it.toOneHot(10) }

        epochs = 5
        batchSize = 128

        verbose()

    }
    
}

```

Examples using the MNIST dataset are available on the test
folder. The MNIST dataset has to be available at the `data` 
directory, it can be downloaded from http://yann.lecun.com/exdb/mnist/.

To use GPU, compile Tensorflow for Java and set the native
library path with the `-Djava.library.path=./path_to_tensorflow` JVM
argument.