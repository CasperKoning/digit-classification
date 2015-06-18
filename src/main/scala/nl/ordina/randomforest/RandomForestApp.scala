package nl.ordina.randomforest

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object RandomForestApp {


  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Scala App")
    val sc = new SparkContext(conf)
    val data = sc.textFile(args(0), 2)
    val parsedData = parseData(data)
    val (trainingData, testData) = splitDataToTrainingAndTestData(parsedData)
    val model = getRandomForestModel(trainingData)
    val labelAndPreds = makePredictions(model,testData)
    val testErr = computeError(labelAndPreds)
    println("Percentage of correctly predicted digits: %,.2f".format(100 * (1 - testErr)))
  }

  def parseData(data: RDD[String]): RDD[LabeledPoint] = {
    data.map { line =>
      val parts = line.split(',').map(_.toInt)
      LabeledPoint(parts(0), Vectors.dense(parts.tail.map(_.toDouble)))
    }
  }

  def splitDataToTrainingAndTestData(data: RDD[LabeledPoint]): (RDD[LabeledPoint], RDD[LabeledPoint]) = {
    val splits = data.randomSplit(Array(0.7, 0.3))
    (splits(0), splits(1))
  }

  def getRandomForestModel(data: RDD[LabeledPoint]): RandomForestModel = {
    data.cache() //Caching of data because it is reused throughout the training of the model.
    //If we did not cache the data, the data has to be recomputed from source through the DAG.

    RandomForest.trainClassifier(//Note: RandomForest, not DecisionTree
      input = data,
      numClasses = 10,
      categoricalFeaturesInfo = Map[Int, Int](),
      impurity = "gini",
      maxDepth = 15,
      maxBins = 5,
      numTrees = 200, //random forest specific configuration
      featureSubsetStrategy = "auto" //random forest specific configuration
    )
  }

  def makePredictions(model: RandomForestModel, testData: RDD[LabeledPoint]): RDD[(Double, Double)] = {
    testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
  }

  def computeError(labelAndPreds: RDD[(Double, Double)]): Double = {
    val correctPredictions = labelAndPreds.filter(r => r._1 != r._2)
                                          .count()
                                          .toDouble
    val totalAmount = labelAndPreds.count()

    correctPredictions/totalAmount
  }

}
