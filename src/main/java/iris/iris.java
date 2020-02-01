package iris;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;

public class iris {

    private static final int FEATURE_COUNT = 0;
    private static final int CLASSES_COUNT = 0;

    public static void main(String[] args){
        try (RecordReader recordReader = new CSVRecordReader(0, ',')){
            recordReader.initialize(new FileSplit(new File("C:\\Users\\Владислав\\Eyes","iris.data")));
            DataSetIterator iterator = new RecordReaderDataSetIterator(
                    recordReader, 150, FEATURE_COUNT, CLASSES_COUNT);
            DataSet allData = iterator.next();
            allData.shuffle(42);

            DataNormalization normalizer = new NormalizerStandardize();
            normalizer.fit(allData);
            normalizer.transform(allData);

            SplitTestAndTrain testAndTrain= allData.splitTestAndTrain(0.65);
            DataSet trainingData = testAndTrain.getTrain();
            DataSet testData = testAndTrain.getTest();

            MultiLayerConfiguration cfg = new NeuralNetConfiguration.Builder()
                    .activation(Activation.TANH)
                    .l2(0.0001)
                    .list()
                    .layer(0, new DenseLayer.Builder().nIn(FEATURE_COUNT).nOut(3).build())
                    .layer(1, new DenseLayer.Builder().nIn(3).nOut(3).build())
                    .layer(2, new OutputLayer.Builder(
                            LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                            .activation(Activation.SOFTMAX)
                            .nIn(3).nOut(CLASSES_COUNT).build())
                    .build();

            MultiLayerNetwork model = new MultiLayerNetwork(cfg);
            model.init();
            model.fit();

            INDArray output = model.output(testData.getFeatures());
            Evaluation eval = new Evaluation(3);
            eval.eval(testData.getLabels(), output);
            eval.stats();

        } catch (InterruptedException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        

    }
}
