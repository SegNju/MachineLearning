/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package auxiliary;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Random;

/**
 *
 * @author daq
 */
public class Kmeans {

	private static int seed = 5;
	public static final int MAX_ITERATION = 500;	// max iteration counts
	public static boolean NEED_MIN_MAX_NORMALIZATION = true;
	
    public Kmeans() {
    	seed++;
    }

    /**
     * Input double[numIns][numAtt] features, int K: num of clusters
     * Output double[K][numAtt] clusterCenters, int[numIns] clusterIndex
     * 
     * clusterCenters[k] should store the kth cluster center
     * clusterIndex[i] should store the cluster index which the ith sample belongs to
     */
    public void train(double[][] features, int K, double[][] clusterCenters, int[] clusterIndex) {
    	
    	/* count of iteration */
    	int iterationCount = 1;
    	
    	/* if the iteration has converged */
    	boolean converged = false;
    	
    	/* cluster centers */
    	ArrayList<ArrayList<Double>> cCenters = new ArrayList<ArrayList<Double>>();
    	
    	/* centers for all the items */
    	ArrayList<Integer> featureCenters = new ArrayList<Integer>();
    	
    	/* error sum */
    	double sumError = 0.0;
    	
    	
        /* num of items */
    	int numItems = features.length;
    	
    	/* count of attributes */
    	int numAttributes = features[0].length;
    	
    	/* preprocess the features */
    	features = preprocess(features);
    	
    	/* min-max normalization of the features */
    	if(NEED_MIN_MAX_NORMALIZATION)
    		features = minMaxNormalize(features, numAttributes);
    	
    	/* randomly choose K initial cluster centers */
    	ArrayList<Integer> chosen = new ArrayList<Integer>();
    	Random rand = new Random(seed);		// different seed yield different results
    	randAssignKCenters(K, rand, cCenters, chosen, features, clusterCenters);
    	
    	
    	/* assign eacha item to its nearest center */
    	assignClusters(features, featureCenters, cCenters);	
    	
    	sumError = calculateErrorSum(features, featureCenters, cCenters);
    	
    	
    	
    	/* repeat until stable or max_iteration count is reached */
    	do{
    		
    		// refresh clusters
    		iterationCount++;
    		
    		cCenters = refreshClusters(features, cCenters, featureCenters);
    		converged = assignClusters(features, featureCenters, cCenters);
    		sumError = calculateErrorSum(features, featureCenters, cCenters);
    	}while(!converged && iterationCount < MAX_ITERATION);
    	
    	
    	
    	//printCluster(featureCenters);
    	
    	
    	
    	/* fill in clusterCenters according to the results */
    	for(int index = 0; index < cCenters.size(); index++){
    		for(int tmp = 0; tmp < numAttributes; tmp++){
    			clusterCenters[index][tmp] = cCenters.get(index).get(tmp);
    		}
    	}
    	
    	/* fill in cluster assignment for all the items */
    	for(int index = 0; index < numItems; index++){
    		clusterIndex[index] = featureCenters.get(index);
    	}
    	
    }

    /* min-max normalize the features, the input need contain no NaN */
    private double[][] minMaxNormalize(double[][] features, int numAttributes) {
		
    	for(int index = 0; index < numAttributes; index++){
    		double curMin = Double.MAX_VALUE;
    		double curMax = Double.MIN_VALUE;
    		
    		for(int itemNumber = 0; itemNumber < features.length; itemNumber++){
    			if(features[itemNumber][index] < curMin)
    				curMin = features[itemNumber][index];
    			
    			if(features[itemNumber][index] > curMax)
    				curMax = features[itemNumber][index];
    		}
    		
    		for(int itemNumber = 0; itemNumber < features.length; itemNumber++){
    			double newVal = (features[itemNumber][index] - curMin) / (curMax - curMin) * 1.0 + 0.0;
    			features[itemNumber][index] = newVal;
    		}
    	}
		return features;
	}

	/* print the clusters */
	private void printCluster(ArrayList<Integer> featureCenters) {
		// TODO Auto-generated method stub
		
		int total = featureCenters.size();
		int cnt0 = 0, cnt1 = 0, cnt2 = 0, cnt3 = 0, cnt4 = 0;
		for(Integer intgr:featureCenters){
			if(intgr == 0)
				cnt0++;
			else if(intgr == 1)
				cnt1++;
			else if(intgr == 2)
				cnt2++;
			else if(intgr == 3)
				cnt3++;
			else
				cnt4++;
		}
		
		System.out.println("Cluster 0: " + (double)cnt0);
		System.out.println("Cluster 1: " + (double)cnt1);
		System.out.println("Cluster 2: " + (double)cnt2);
		System.out.println("Cluster 3: " + (double)cnt3);
		System.out.println("Cluster 4: " + (double)cnt4);
		
	}

	/* randomly assign K cluster centers */
	private void randAssignKCenters(int K, Random rand,
			ArrayList<ArrayList<Double>> cCenters, ArrayList<Integer> chosen, double[][] features, double[][]clusterCenters) {
		cCenters.clear();
		
		int i = 0;
    	while(i != K){
    		int pos = rand.nextInt(features.length);
    		if(chosen.contains(pos))
    			continue;	// already chosen
    		else{
    			ArrayList<Double> toadd = new ArrayList<Double>();
    			for(int j = 0; j < clusterCenters[0].length; j++){
    				toadd.add(features[pos][j]);
    			}
    			i++;
    			cCenters.add(toadd);	// add a center to the cCenters
    			chosen.add(pos);
    			
    		}
    	}
    	
		/*
		ArrayList<Double> d1 = new ArrayList<Double>();
		d1.add(3.0);d1.add(2.0);d1.add(4.0);d1.add(0.0);d1.add(1.0);d1.add(1.0);d1.add(1.0);d1.add(0.0);d1.add(1.0);
		ArrayList<Double> d2 = new ArrayList<Double>();
		d2.add(3.0);d2.add(2.0);d2.add(6.0);d2.add(0.0);d2.add(0.0);d2.add(2.0);d2.add(1.0);d2.add(0.0);d2.add(1.0);
		ArrayList<Double> d3 = new ArrayList<Double>();
		d3.add(4.0);d3.add(2.0);d3.add(5.0);d3.add(0.0);d3.add(1.0);d3.add(0.0);d3.add(0.0);d3.add(1.0);d3.add(1.0);
		ArrayList<Double> d4 = new ArrayList<Double>();
		d4.add(4.0);d4.add(1.0);d4.add(6.0);d4.add(1.0);d4.add(0.0);d4.add(2.0);d4.add(0.0);d4.add(1.0);d4.add(0.0);
		ArrayList<Double> d5 = new ArrayList<Double>();
		d5.add(4.0);d5.add(1.0);d5.add(6.0);d5.add(0.0);d5.add(1.0);d5.add(0.0);d5.add(1.0);d5.add(2.0);d5.add(1.0);
		
		cCenters.add(d1);
		cCenters.add(d2);
		cCenters.add(d3);
		cCenters.add(d4);
		cCenters.add(d5);
		*/
		
	}

	/* re-calculate cluster centers */
	private ArrayList<ArrayList<Double>> refreshClusters(double[][] features,
			ArrayList<ArrayList<Double>> cCenters, ArrayList<Integer> featureCenters) {
		int attrCnt = features[0].length;
		ArrayList<ArrayList<Double>> newCenters = new ArrayList<ArrayList<Double>>();
		
		for(int i = 0; i < cCenters.size(); i++){
			ArrayList<Double> toadd = new ArrayList<Double>();
			
			
			double[] sumToadd = new double[attrCnt];
			for(int tmp = 0; tmp < attrCnt; tmp++)
				sumToadd[tmp] = 0.0;
			double cnt = 0.0;
			for(int j = 0; j < features.length; j++){
				if(featureCenters.get(j) == i){
					cnt += 1.0;
					for(int k = 0; k < features[j].length; k++){
						double val = Double.isNaN(features[j][k])? 0 : features[j][k];
						sumToadd[k] += val;
					}
				}
			}
			
			for(int tmp = 0; tmp < attrCnt; tmp++){
				toadd.add(sumToadd[tmp] / cnt);	// the new Center for pre-center i
			}
			
			
			
			
			/*
			for(int j = 0; j < cCenters.get(0).size(); j++){
				int attcnt = 0;
				int curAttCnt = 0;
				double val = 0.0;
				double curVal = 0.0;
				HashMap<Double, Integer> map = new HashMap<Double, Integer>();
				for(int k = 0; k < features.length; k++){
					if(featureCenters.get(k) == i){
						curVal = features[k][j];
						if(Double.isNaN(curVal))
							curVal = 0.0;
						
						if(!map.containsKey(curVal)){
							map.put(curVal, 1);
							
							if(1 > attcnt){
								attcnt = 1;
								val = curVal;
							}
						}else{
							int valCnt = map.get(curVal);
							valCnt++;
							map.put(curVal, valCnt);
							if(valCnt > attcnt){
								attcnt = valCnt;
								val = curVal;
							}
						}
					}
					
					
				}
				toadd.add(val);
			}
			*/
			
			newCenters.add(toadd);
		}
		return newCenters;
	}

	/* calculate the error sum */
	private double calculateErrorSum(double[][] features,
			ArrayList<Integer> featureCenters, ArrayList<ArrayList<Double>> cCenters) {
		// TODO Auto-generated method stub
		double sum = 0.0;
		for(int i = 0; i < features.length; i++){
			int clusterIndex = featureCenters.get(i);
			ArrayList<Double> center = cCenters.get(clusterIndex);
			double curSum = 0.0;
			for(int j = 0; j < center.size(); j++){
				double v1 = Double.isNaN(features[i][j]) ? 0 : features[i][j];
                double v2 = Double.isNaN(center.get(j)) ? 0 : center.get(j);
				curSum += ((v1 - v2) * (v1 - v2));
			}
			sum += curSum;
		}
		
		return sum;
	}
	
	
	/* preprocess, fill in NaN number with Median */
	private double[][] preprocess(double[][] features) {
		int attrCount = features[0].length;
		double[] midVals = new double[attrCount];	// mid values of attributes
		for(int i = 0; i < attrCount; i++){	// attain mid values of all attributes
			ArrayList<Double> list = new ArrayList<Double>();
			for(int j = 0; j < features.length; j++){
				if(!Double.isNaN(features[j][i])){
					list.add(features[j][i]);
				}
			}
			Collections.sort(list);
			if(list.isEmpty())
				midVals[i] = 0.0;
			else{
				double mid = list.get(list.size() / 2).doubleValue();
				midVals[i] = mid;
				//System.out.println("attribute " + i + " mid " + mid + " ,min " + list.get(0).doubleValue() + " ,max " + list.get(list.size() - 1).doubleValue());
			}
		}
		
		
		// fillin all NaN values with corresponding mid values
		for(int i = 0; i < features.length; i++)
			for(int j = 0; j < features[i].length; j++){
				if(Double.isNaN(features[i][j])){
					features[i][j] = midVals[j];
				}
			}
		return features;
	}

	
	/* assign cluster to each item according to the new cCenters 
	 * return true if no cluster change happened, otherwise return false
	 */
	private boolean assignClusters(double[][] features,
			ArrayList<Integer> featureCenters, ArrayList<ArrayList<Double>> cCenters) {
		
		
		boolean ret = true;
		ArrayList<Integer> oldfeatureCenters = new ArrayList<Integer>();
		for(Integer I:featureCenters)
			oldfeatureCenters.add(I);
		double errorSum = 0.0;
		featureCenters.clear();
		for(int i = 0; i < features.length; i++){
			double dist = Double.MAX_VALUE;
			int centerIndex = 0;
			
			for(int j = 0; j < cCenters.size(); j++){
				ArrayList<Double> curCenter = cCenters.get(j);
				// calculate distance
				double curDist = 0.0;
				
				for(int k = 0; k < curCenter.size(); k++){
					double v1 = Double.isNaN(features[i][k]) ? 0 : features[i][k];
                    double v2 = Double.isNaN(curCenter.get(k)) ? 0 : curCenter.get(k);
					curDist += ((v1 - v2) * (v1 - v2));
				}
				
				
				
				
				if(curDist < dist){
					dist = curDist;
					centerIndex = j;
				}
			}
			
			// assign cluster to features[i]
			//errorSum += dist;
			if(!oldfeatureCenters.isEmpty() && centerIndex != oldfeatureCenters.get(i))
				ret = false;
			featureCenters.add(centerIndex);
			
		}
	
		return ret;
	}
    
}
