package org.deeplearning4j;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;


/**
 * Created by agibsonccc on 9/11/14.
 */
public class DBNExample {

    private static Logger log = LoggerFactory.getLogger(DBNExample.class);


    public static void main(String[] args) throws Exception {

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
                .iterations(5)
                .list(4)
                .layer(0, new RBM.Builder()
                        .weightInit(WeightInit.VI)
                        .learningRate(1e-1f)
                        .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                        .nIn(784)
                        .nOut(600)
                        .build())
                .layer(1, new RBM.Builder()
                        .weightInit(WeightInit.VI)
                        .learningRate(1e-1f)
                        .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                        .nIn(600)
                        .nOut(500)
                        .build())
                .layer(2, new RBM.Builder()
                        .weightInit(WeightInit.VI)
                        .learningRate(1e-1f)
                        .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                        .nIn(500)
                        .nOut(400)
                        .build())
                .layer(3, new OutputLayer.Builder()
                        .weightInit(WeightInit.VI)
                        .nIn(400)
                        .nOut(10)
                        .build())
                .pretrain(true).backprop(false)
                .build();

        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        network.setListeners(Arrays.<IterationListener>asList(new ScoreIterationListener(1)));

        DataSetIterator iter = new MultipleEpochsIterator(10,new MnistDataSetIterator(1000,1000));
        network.fit(iter);

        iter.reset();
        Evaluation eval = new Evaluation();

        while(iter.hasNext()) {
            DataSet d2 = iter.next();
            INDArray predict2 = network.output(d2.getFeatureMatrix());
            eval.eval(d2.getLabels(), predict2);
        }
        log.info(eval.stats());
    }
}
