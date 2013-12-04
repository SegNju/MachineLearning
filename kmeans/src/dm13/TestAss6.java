/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package dm13;

import auxiliary.DataSet;
import auxiliary.Evaluation;

/**
 *
 * @author daq
 */
public class TestAss6 {

    public static void main(String[] args) {
        int K = 5;
        
        String[] dataPaths = new String[]{"breast-cancer.data", "segment.data","housing.data", "meta.data"};
        for (String path : dataPaths) {
            DataSet dataset = new DataSet(path);

            Evaluation eva = new Evaluation(dataset, "Kmeans");
            eva.evaluateClustering(K);

            // print mean and standard deviation of accuracy
            System.out.println("Dataset:" + path + ", distance:" + eva.getClusterDist());
        }
    }
}
