## Project Introduction to Machine Learning
โปรเจคนี้เป็นส่วนหนึ่งในรายวิชา Introductino to Machine Learning 01418362
จัดทำโดย นาย พีรสิษฐ์ พลอยอร่าม 6410451237
<br>

## Water Quality DataSet
เป็น DataSet วัดคุณภาพของนำ้ว่ามีความปลอดภัยหรือไม่ โดยประกอบไปด้วย 20 features ที่ใช้ในการจำแนกคุณภาพของน้ำ และ label เฉลยที่บอกว่าน้ำมีความปลอดภัยหรือไม่

#### features
- aluminium - dangerous if greater than 2.8
- ammonia - dangerous if greater than 32.5
- arsenic - dangerous if greater than 0.01
- barium - dangerous if greater than 2
- cadmium - dangerous if greater than 0.005
- chloramine - dangerous if greater than 4
- chromium - dangerous if greater than 0.1
- copper - dangerous if greater than 1.3
- flouride - dangerous if greater than 1.5
- cteria - dangerous if greater than 0
- viruses - dangerous if greater than 0
- lead - dangerous if greater than 0.015
- nitrates - dangerous if greater than 10
- nitrites - dangerous if greater than 1
- mercury - dangerous if greater than 0.002
- perchlorate - dangerous if greater than 56
- radium - dangerous if greater than 5
- selenium - dangerous if greater than 0.5
- silver - dangerous if greater than 0.1
- uranium - dangerous if greater than 0.3
#### Y label
- is_safe - class attribute {0 - not safe, 1 - safe}
#### Reference
- แหล่งที่มาของ Water Quality DataSet: https://www.kaggle.com/datasets/mssmartypants/water-quality/data


## Directory
โครงสร้างของ file ประกอบด้วยดังนี้
Directory 
- data -> ที่อยุ่ของ DataSet
    - waterQuality1.csv
- src  -> ไฟล์หลัก
    - main_water_Quality1.ipynb -> ไฟล์ที่ใช้ในการทดอง
    - model_selection.py -> Libray ที่เขียนขึ้นเพื่อใช้ในในการทดลอง
    - model.py -> Model Classifier ที่ใช้ในการเรียนรู้โดยใช้ Linear SVM แบบ soft margin


## Support Vector Machine: SVM