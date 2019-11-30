
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.sql.functions._

object spam {

  def main(args: Array[String]): Unit = {

    // start spark session
    val spark = SparkSession
      .builder
      .master("local")
      .appName("Classification")
      .getOrCreate()

    // define input paths
    val spam_train = ""
    val spam_test = ""
    val nospam_train = ""
    val nospam_test = ""


    // load data as spark-datasets
    val spam_training = spark.read.textFile(spam_train)
    val spam_testing = spark.read.textFile(spam_test)
    val nospam_training = spark.read.text(nospam_train)
    val nospam_testing = spark.read.text(nospam_test)

    // implement: convert datasets to either rdds or dataframes (your choice) and build your pipeline

    // training data
    val spam_training_df = spam_training.toDF("feature")
    val no_spam_training_df = nospam_training.toDF("feature")
    
    // skip this step if the data is already labeled
    val spam_training_labeled = spam_training_df.withColumn("label", lit(1.0))
    val no_spam_training_labeled = no_spam_training_df.withColumn("label", lit(0.0))

    val training_data = (spam_training_labeled.union(no_spam_training_labeled)).orderBy(rand())
    //training_data.collect.foreach(println)

    // testing data
    val spam_testing_df = spam_testing.toDF("feature")
    val no_spam_testing_df = nospam_testing.toDF("feature")

    // skip this step if the data is already labeled
    val spam_testing_labeled = spam_testing_df.withColumn("label", lit(1.0))
    val no_spam_testing_labeled = no_spam_testing_df.withColumn("Label", lit(0.0))

    val testing_data = (spam_testing_labeled.union(no_spam_testing_labeled)).orderBy(rand())

    // bag of words feature with IDF
    val tokenizer = new Tokenizer().setInputCol("feature").setOutputCol("words")
    val wordsData = tokenizer.transform(training_data)
    val wordsTest = tokenizer.transform(testing_data)
    // TF IDF
    val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(20)

    val featurizedData = hashingTF.transform(wordsData)
    val featurizedTraining = hashingTF.transform(wordsTest)

    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)
    val idfTesting = idf.fit(featurizedTraining)

    val rescaledData = idfModel.transform(featurizedData)
    val rescaledTraining = idfTesting.transform(featurizedTraining)
    rescaledData.select("label", "features").show()

    // training
    val model = new NaiveBayes().fit(rescaledData.select("label", "features"))

    // testing
    val predictions = model.transform(rescaledData.select("label", "features"))
    predictions.show()

    //test error
    val eval = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
    val accuracy = eval.evaluate(predictions)

    println("Test accuracy: " + accuracy)


  }
}
