package com.csye7200

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.ml.feature.{StringIndexer, OneHotEncoder, VectorAssembler}
import org.apache.spark.ml.classification.{RandomForestClassifier, LogisticRegression, GBTClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.log4j.{Level, Logger}


object TitanicAnalysis {
  // 设置根日志记录器的级别为 WARN
  Logger.getRootLogger.setLevel(Level.WARN)

  // 确保 org.apache 下的所有日志记录器都设置为 WARN
  Logger.getLogger("org").setLevel(Level.WARN)
  Logger.getLogger("org.apache").setLevel(Level.WARN)
  Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
  Logger.getLogger("org.spark-project").setLevel(Level.WARN)


  // 定义特征工程函数，分别应用于训练集和测试集
  def featureEngineering(df: DataFrame, isTrain: Boolean): DataFrame = {
    var result = df

    // 1. 从Name中提取Title(称谓)
    result = result.withColumn("Title", regexp_extract(col("Name"), "([A-Za-z]+)\\.", 1))

    // 简化称谓类别
    result = result.withColumn("Title",
      when(col("Title") === "Mr", "Mr")
        .when(col("Title").isin("Miss", "Ms", "Mlle"), "Miss")
        .when(col("Title").isin("Mrs", "Mme"), "Mrs")
        .when(col("Title") === "Master", "Master")
        .otherwise("Other")
    )

    // 2. 创建家庭大小特征
    result = result.withColumn("FamilySize", col("SibSp") + col("Parch") + 1)

    // 3. 创建是否独自旅行的特征
    result = result.withColumn("IsAlone", when(col("FamilySize") === 1, 1).otherwise(0))

    // 4. 创建年龄组特征
    result = result.withColumn(
      "AgeGroup",
      when(col("Age").isNull, "Unknown")
        .when(col("Age") <= 12, "Child")
        .when(col("Age") <= 18, "Teenager")
        .when(col("Age") <= 30, "YoungAdult")
        .when(col("Age") <= 50, "Adult")
        .otherwise("Senior")
    )

    // 5. 创建票价组特征
    result = result.withColumn(
      "FareGroup",
      when(col("Fare").isNull, "Unknown")
        .when(col("Fare") <= 7.91, "Low")
        .when(col("Fare") <= 14.454, "LowMedium")
        .when(col("Fare") <= 31, "Medium")
        .otherwise("High")
    )

    // 6. 处理缺失值
    // 计算年龄中位数
    val medianAge = result.stat.approxQuantile("Age", Array(0.5), 0.01)(0)

    // 用中位数填充年龄缺失值
    result = result.withColumn("Age",
      when(col("Age").isNull, medianAge).otherwise(col("Age"))
    )

    // 填充Embarked缺失值
    val embarkedCounts = result.groupBy("Embarked").count().orderBy(col("count").desc)
    if (!embarkedCounts.isEmpty) {
      val mostCommonEmbarked = embarkedCounts.first().getAs[String]("Embarked")
      result = result.withColumn("Embarked",
        when(col("Embarked").isNull || col("Embarked") === "", mostCommonEmbarked).otherwise(col("Embarked"))
      )
    }

    // 计算票价中位数
    val medianFare = result.stat.approxQuantile("Fare", Array(0.5), 0.01)(0)

    // 用中位数填充票价缺失值
    result = result.withColumn("Fare",
      when(col("Fare").isNull, medianFare).otherwise(col("Fare"))
    )

    // 7. 从名称中提取姓氏
    result = result.withColumn("Surname", regexp_extract(col("Name"), "^([^,]+),", 1))

    // 8. 创建仓位(Cabin)首字母特征
    result = result.withColumn("CabinLetter",
      when(col("Cabin").isNull, "U").otherwise(substring(col("Cabin"), 1, 1))
    )

    return result
  }

  def main(args: Array[String]): Unit = {
    // 创建SparkSession
    val spark = SparkSession.builder()
      .appName("TitanicPrediction")
      .master("local[*]")
      .config("spark.ui.showConsoleProgress", "false")
      .config("spark.log.level", "WARN")
      .getOrCreate()

    // 导入隐式转换
    import spark.implicits._

    // 读取训练和测试数据集
    val trainDF = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("src/main/resources/train.csv")

    val testDF = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("src/main/resources/test.csv")

    // 显示数据集基本信息
    println("Training dataset structure:")
    trainDF.printSchema()
    println(s"Training dataset record count: ${trainDF.count()}")

    println("\nTest dataset structure:")
    testDF.printSchema()
    println(s"Test dataset record count: ${testDF.count()}")

    println("\nFirst 5 records from training dataset:")
    trainDF.show(5)

    // ================ 第一部分: 探索性数据分析 ================
    println("\n================ Part 1: Exploratory Data Analysis ================")

    // 1. 检查缺失值
    println("\n1. Missing values analysis:")

    // 定义统计缺失值的函数
    def countNulls(df: DataFrame): Array[(String, Long, Double)] = {
      val totalCount = df.count()
      df.columns.map { colName =>
        val nullCount = df.filter(col(colName).isNull || isnan(col(colName))).count()
        val nullPercentage = nullCount.toDouble / totalCount * 100
        (colName, nullCount, nullPercentage)
      }
    }

    // 分析训练集缺失值
    val trainNulls = countNulls(trainDF)
    println("Training dataset missing values:")
    trainNulls.foreach { case (colName, nullCount, nullPercentage) =>
      println(f"$colName: $nullCount ($nullPercentage%.2f%%)")
    }

    // 分析测试集缺失值
    val testNulls = countNulls(testDF)
    println("\nTest dataset missing values:")
    testNulls.foreach { case (colName, nullCount, nullPercentage) =>
      println(f"$colName: $nullCount ($nullPercentage%.2f%%)")
    }

    // 2. 继续前一个作业的统计分析
    // 2.1 船票等级和存活率的关系
    println("\n2.1 Relationship between ticket class and survival rate:")
    val pclassSurvival = trainDF.groupBy("Pclass")
      .agg(
        count("PassengerId").alias("Total passengers"),
        sum("Survived").alias("Survivors"),
        round(sum("Survived") / count("PassengerId") * 100, 2).alias("Survival rate(%)")
      )
      .orderBy("Pclass")

    pclassSurvival.show()

    // 2.2 性别与存活率的关系
    println("\n2.2 Relationship between gender and survival rate:")
    val sexSurvival = trainDF.groupBy("Sex")
      .agg(
        count("PassengerId").alias("Total passengers"),
        sum("Survived").alias("Survivors"),
        round(sum("Survived") / count("PassengerId") * 100, 2).alias("Survival rate(%)")
      )
      .orderBy("Sex")

    sexSurvival.show()

    // 2.3 年龄分布统计
    println("\n2.3 Age distribution statistics:")
    val ageStats = trainDF.select(
      min("Age").alias("Minimum age"),
      max("Age").alias("Maximum age"),
      avg("Age").alias("Average age"),
      stddev("Age").alias("Age standard deviation")
    )

    ageStats.show()

    // 2.4 票价统计
    println("\n2.4 Fare statistics:")
    val fareStats = trainDF.select(
      min("Fare").alias("Minimum fare"),
      max("Fare").alias("Maximum fare"),
      avg("Fare").alias("Average fare"),
      stddev("Fare").alias("Fare standard deviation")
    )

    fareStats.show()

    // 2.5 家庭大小与存活率关系
    println("\n2.5 Relationship between family size and survival rate:")
    val trainWithFamily = trainDF.withColumn("FamilySize", col("SibSp") + col("Parch") + 1)

    val familySurvival = trainWithFamily.groupBy("FamilySize")
      .agg(
        count("PassengerId").alias("Total passengers"),
        sum("Survived").alias("Survivors"),
        round(sum("Survived") / count("PassengerId") * 100, 2).alias("Survival rate(%)")
      )
      .orderBy("FamilySize")

    familySurvival.show()

    // 2.6 登船港口与存活率关系
    println("\n2.6 Relationship between embarkation port and survival rate:")
    val embarkedSurvival = trainDF.groupBy("Embarked")
      .agg(
        count("PassengerId").alias("Total passengers"),
        sum("Survived").alias("Survivors"),
        round(sum("Survived") / count("PassengerId") * 100, 2).alias("Survival rate(%)")
      )
      .orderBy("Embarked")

    embarkedSurvival.show()

    // ================ 第二部分: 特征工程 ================
    println("\n================ Part 2: Feature Engineering ================")

    // 对训练集和测试集分别进行特征工程
    println("\nProcessing training dataset...")
    val processedTrain = featureEngineering(trainDF, true)

    println("\nProcessing test dataset...")
    val processedTest = featureEngineering(testDF, false)

    // 显示处理后的数据结构
    println("\nProcessed training dataset structure:")
    processedTrain.printSchema()

    // 显示处理后的数据示例
    println("\nProcessed training dataset examples:")
    processedTrain.select(
      "PassengerId", "Survived", "Pclass", "Sex", "Age", "AgeGroup", "FamilySize",
      "IsAlone", "Title", "Fare", "FareGroup", "Embarked", "CabinLetter"
    ).show(5)

    // ================ 第三部分: 预测模型 ================
    println("\n================ Part 3: Prediction Model ================")

    // 1. 选择模型特征
    val selectedFeatures = Array(
      "Pclass", "Sex", "Age", "Fare", "Embarked",
      "FamilySize", "IsAlone", "Title", "CabinLetter"
    )

    // 2. 构建ML管道
    // 将分类变量转换为数值型
    val stringIndexers = Array("Sex", "Embarked", "Title", "CabinLetter").map { colName =>
      new StringIndexer()
        .setInputCol(colName)
        .setOutputCol(s"${colName}_idx")
        .setHandleInvalid("keep")
    }

    // One-Hot编码处理
    val oneHotEncoders = Array("Sex", "Embarked", "Title", "CabinLetter").map { colName =>
      new OneHotEncoder()
        .setInputCol(s"${colName}_idx")
        .setOutputCol(s"${colName}_vec")
    }

    // 组装特征向量
    val numericFeatures = Array("Pclass", "Age", "Fare", "FamilySize", "IsAlone")
    val categoricalFeatures = Array("Sex_vec", "Embarked_vec", "Title_vec", "CabinLetter_vec")

    val assembler = new VectorAssembler()
      .setInputCols(numericFeatures ++ categoricalFeatures)
      .setOutputCol("features")
      .setHandleInvalid("keep")

    // 3. 创建不同的分类器
    // 随机森林分类器
    val rf = new RandomForestClassifier()
      .setLabelCol("Survived")
      .setFeaturesCol("features")
      .setNumTrees(100)
      .setMaxDepth(5)
      .setSeed(42)

    // 逻辑回归
    val lr = new LogisticRegression()
      .setLabelCol("Survived")
      .setFeaturesCol("features")
      .setMaxIter(10)
      .setRegParam(0.01)

    // 梯度提升树
    val gbt = new GBTClassifier()
      .setLabelCol("Survived")
      .setFeaturesCol("features")
      .setMaxIter(10)
      .setMaxDepth(5)
      .setSeed(42)

    // 创建不同的管道
    val rfPipeline = new Pipeline().setStages(stringIndexers ++ oneHotEncoders ++ Array(assembler, rf))
    val lrPipeline = new Pipeline().setStages(stringIndexers ++ oneHotEncoders ++ Array(assembler, lr))
    val gbtPipeline = new Pipeline().setStages(stringIndexers ++ oneHotEncoders ++ Array(assembler, gbt))

    // 训练模型
    println("\nTraining Random Forest model...")
    val rfModel = rfPipeline.fit(processedTrain)

    println("\nTraining Logistic Regression model...")
    val lrModel = lrPipeline.fit(processedTrain)

    println("\nTraining Gradient Boosting Tree model...")
    val gbtModel = gbtPipeline.fit(processedTrain)

    // 4. 模型评估 - 使用交叉验证进行
    // 使用交叉验证优化随机森林模型
    val paramGrid = new ParamGridBuilder()
      .addGrid(rf.numTrees, Array(50, 100))
      .addGrid(rf.maxDepth, Array(3, 5, 7))
      .build()

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("Survived")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    println("\nPerforming cross-validation to optimize Random Forest model...")
    val crossval = new CrossValidator()
      .setEstimator(rfPipeline)
      .setEstimatorParamMaps(paramGrid)
      .setEvaluator(evaluator)
      .setNumFolds(5)

    val cvModel = crossval.fit(processedTrain)
    val bestRfModel = cvModel.bestModel

    // 5. 在交叉验证集上评估模型
    // 预测并评估随机森林模型
    val rfPredictions = rfModel.transform(processedTrain)
    val lrPredictions = lrModel.transform(processedTrain)
    val gbtPredictions = gbtModel.transform(processedTrain)
    val cvPredictions = bestRfModel.transform(processedTrain)

    // 计算准确率
    val rfAccuracy = evaluator.evaluate(rfPredictions)
    val lrAccuracy = evaluator.evaluate(lrPredictions)
    val gbtAccuracy = evaluator.evaluate(gbtPredictions)
    val cvAccuracy = evaluator.evaluate(cvPredictions)

    println("\nModel performance evaluation:")
    println(f"Random Forest accuracy: $rfAccuracy%.4f")
    println(f"Logistic Regression accuracy: $lrAccuracy%.4f")
    println(f"Gradient Boosting Tree accuracy: $gbtAccuracy%.4f")
    println(f"Cross-validated Random Forest accuracy: $cvAccuracy%.4f")

    // 选择最佳模型用于测试集预测
    val (bestModel, modelName): (PipelineModel, String) = {
      val maxAccuracy = Seq(rfAccuracy, lrAccuracy, gbtAccuracy, cvAccuracy).max
      if (maxAccuracy == rfAccuracy) (rfModel, "Random Forest")
      else if (maxAccuracy == lrAccuracy) (lrModel, "Logistic Regression")
      else if (maxAccuracy == gbtAccuracy) (gbtModel, "Gradient Boosting Tree")
      else (bestRfModel.asInstanceOf[PipelineModel], "Cross-validated Random Forest")
    }

    println(s"\nSelected ${modelName} model as the final model")

    // 6. 对测试集进行预测
    println("\nPredicting on test dataset...")
    val testPredictions = bestModel.transform(processedTest)

    // 7. 生成提交文件
    val submissionDF = testPredictions.select(
      col("PassengerId"),
      col("prediction").cast("integer").alias("Survived")
    )

    println("\nPrediction results example:")
    submissionDF.show(10)

    println(s"\nGenerated submission file contains ${submissionDF.count()} prediction results")

    // 将结果保存为CSV文件
    submissionDF.write.mode("overwrite").option("header", "true").csv("titanic_submission")

    println("\nPrediction complete! Submission file has been saved.")

    // 关闭SparkSession
    spark.stop()
  }
}