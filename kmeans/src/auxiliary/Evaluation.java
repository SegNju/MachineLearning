package auxiliary;

import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author daq
 */
public class Evaluation {

    private String clsName;
    private DataSet dataset;
    private double clusterDist;

    public Evaluation() {
    }

    public Evaluation(DataSet dataset, String clsName) {
        this.dataset = dataset;
        this.clsName = clsName;
    }

    public void evaluateClustering(int K) {
        try {
            Kmeans c = (Kmeans) Class.forName("auxiliary." + clsName).newInstance();

            double[][] features = dataset.getFeatures();
            double[][] clusterCenters = new double[K][dataset.getNumAttributes()];
            int[] clusterIndex = new int[dataset.getNumInstnaces()];
            c.train(features, K, clusterCenters, clusterIndex);

            double sum = 0;
            for (int i = 0; i < dataset.getNumInstnaces(); i++) {
                double[] center = clusterCenters[clusterIndex[i]];

                double dist = 0;
                for (int j = 0; j < dataset.getNumAttributes(); j++) {
                    double v1 = Double.isNaN(features[i][j]) ? 0 : features[i][j];
                    double v2 = Double.isNaN(center[j]) ? 0 : center[j];
                    dist = dist + (v1 - v2) * (v1 - v2);
                }
                
                sum = sum + dist;
            }
            
            clusterDist = sum ;/// dataset.getNumInstnaces();
        } catch (ClassNotFoundException ex) {
            Logger.getLogger(Evaluation.class.getName()).log(Level.SEVERE, null, ex);
        } catch (InstantiationException ex) {
            Logger.getLogger(Evaluation.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IllegalAccessException ex) {
            Logger.getLogger(Evaluation.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public double getClusterDist() {
        return clusterDist;
    }
}
