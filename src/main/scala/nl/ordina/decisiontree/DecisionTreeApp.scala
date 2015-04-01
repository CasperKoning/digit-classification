package nl.ordina.decisiontree

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.{SparkConf, SparkContext}

object DecisionTreeApp {

  def main(args: Array[String]) {
    //Voorbereiden spark context
    val conf = new SparkConf().setAppName("Scala App")
    val sc = new SparkContext(conf)

    //Inladen data
    val trainingSet = "D:\\dev\\datasets\\train.csv"
    val data = sc.textFile(trainingSet, 2)

    //Parsen van data
    val parsedData = data.map { line =>
        val parts = line.split(',').map(_.toInt)
        LabeledPoint(parts(0), Vectors.dense(parts.tail.map(_.toDouble)))
    }

    //Opdelen van data in training set en test set
    val splits = parsedData.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

    //Configuratie van decision tree algoritme
    val numberOfClasses = 10
    val categoricalFeaturesInfo = Map[Int, Int]()
    val impurity = "gini"
    val maxDepth = 10
    val maxBins = 32

    //Trainen van model
    val model = DecisionTree.trainClassifier(trainingData, numberOfClasses, categoricalFeaturesInfo,
      impurity, maxDepth, maxBins)

    //Voer model uit op testdata
    val labelAndPreds = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }

    //Bereken fout van model
    val testErr = labelAndPreds.filter(r => r._1 != r._2).count().toDouble / testData.count()

    //Print fout
    println("Correct voorspeld: %,.2f".format(100*(1-testErr)))
  }

}