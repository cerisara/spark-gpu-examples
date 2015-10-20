package org.deeplearning4j;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.conf.override.ClassifierOverride;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.FileInputStream;
import java.io.InputStream;
import java.util.Collections;

/**
 * @author Adam Gibson
 */
public class LoadAndPredict {

    public static void main(String[] args) throws Exception {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .constrainGradientToUnitNorm(true)
                .iterations(5)
                .learningRate(1e-1f).batchSize(1000)
                .momentum(0.5).constrainGradientToUnitNorm(true)
                .momentumAfter(Collections.singletonMap(3, 0.9))
                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
                .list(4)
                .layer(0, new RBM.Builder()
                        .weightInit(WeightInit.XAVIER)
                        .learningRate(1e-1f)
                        .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                        //.activationFunction("identity")
                        .nIn(784)
                        .nOut(600)
                        .build())
                .layer(1, new RBM.Builder()
                        .weightInit(WeightInit.XAVIER)
                        .learningRate(1e-1f)
                        .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                        .nIn(600)
                        .nOut(250)
                        .build())
                .layer(2, new RBM.Builder()
                        .weightInit(WeightInit.XAVIER)
                        .learningRate(1e-1f)
                        .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                        .nIn(250)
                        .nOut(200)
                        .build())
                .layer(3, new OutputLayer.Builder()
                        .weightInit(WeightInit.XAVIER)
                        .nIn(200)
                        .nOut(10)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        InputStream is = new FileInputStream("params.txt");
        INDArray read  = Nd4j.read(is);
        model.setParameters(read);
        DataSetIterator iter = new MnistDataSetIterator(1000,60000);
        Evaluation eval = new Evaluation();
        while(iter.hasNext()) {
            DataSet next = iter.next();
            eval.eval(next.getLabels(),model.output(next.getFeatureMatrix(), true));
        }

        System.out.println(eval.stats());
    }


}
