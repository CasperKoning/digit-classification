package nl.ordina.randomforest

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.{SparkConf, SparkContext}

object RandomForestApp {


  def main(args: Array[String]) {
    //Voorbereiden spark context
    val conf = new SparkConf().setAppName("Scala App")
    val sc = new SparkContext(conf)

    //Inladen data
    val projectRoot = "C:\\Users\\cko20685\\IdeaProjects\\DigitClassification" //TODO: pas aan naar geschikte directory
    val trainingSet = projectRoot + "\\src\\main\\resources\\train.csv"
    val data = sc.textFile(trainingSet, 2).cache()

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
    val maxDepth = 15
    val maxBins = 5
    val numberOfTrees = 200                       //random forest specifieke configuratie
    val featureSubsetStrategy = "auto"            //random forest specifieke configuratie

    //Trainen van model
    val model = RandomForest.trainClassifier(trainingData,numberOfClasses,categoricalFeaturesInfo,
          numberOfTrees,featureSubsetStrategy,impurity, maxDepth, maxBins)

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
