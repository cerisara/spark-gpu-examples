package org.deeplearning4j;

import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.spark.impl.layer.SparkDl4jLayer;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.deeplearning4j.nn.conf.layers.DenseLayer;

/**
 * @author sonali
 */
public class SparkGpuExample {

    public static void main(String[] args) throws Exception {
        // set to test mode
        SparkConf sparkConf = new SparkConf().set(SparkDl4jMultiLayer.AVERAGE_EACH_ITERATION,"false")
                .setMaster("local[*]")
                .setAppName("sparktest");

        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .constrainGradientToUnitNorm(true).useDropConnect(true)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(10)
                .learningRate(1e-1)
                .layer(new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder()
                        .weightInit(WeightInit.XAVIER)
                        .dropOut(0.5)
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                                //.activationFunction("softmax")
                        .nIn(4).nOut(3)
                        .build())
                .build();

        System.out.println("Initializing network");
        SparkDl4jLayer master = new SparkDl4jLayer(sc,conf);
        DataSet d = new IrisDataSetIterator(150,150).next();
        d.normalizeZeroMeanZeroUnitVariance();
        d.shuffle();
        List<DataSet> next = d.asList();

        JavaRDD<DataSet> data = sc.parallelize(next);

        OutputLayer network2 = (OutputLayer) master.fitDataSet(data);
        Evaluation evaluation = new Evaluation(3);
        evaluation.eval(d.getLabels(), network2.output(d.getFeatureMatrix()));
        System.out.println(evaluation.stats());
    }
}
