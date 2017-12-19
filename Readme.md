### 生成模型

建立`train`文件夹并在其中放入文件名以cat开头猫的图片，或者以dog开头狗的图片，并执行

```shell
python classifier.py
```

之后生成模型文件 model_final.tflearn*

### 进行预测

建立`test`文件夹并在其中放入狗或猫的图片，并执行

```shell
python predict.py
```

结果保存在submission.csv中，其中每行第一列代表文件名，第二列的为预测值（0或1，其中0表示狗，1表示猫）