# DVK-uNet – Neuronales Netz zur Schätzung von Dosis-Voxel-Kernen
Für die Berechnung der absorbierten Strahlungsdosis in der Dosimetrie werden Monte-Carlo Simulationen auf Gewebedichten aus einer CT-Bildgebung vollführt, um die deponierte Energie zu berechnen.
Der Transfer von Massendichten in die sog. Dosis-Voxel-Kernel entspricht einem Bild zu Bild Transfer. Dieser Transfer kann mit Hilfe von Verfahren aus der Bildrekonstruktion und Bildsegmentierung gelernt werden. Ich folgenden wird der Algorithmus vorgestellt, mit dessen Hilfe eine alternative und schnelle Lösung für die Berechnung von DVKs zur Verfügung gestellt wird.

## Simulation der DVKs
![DVK Illustration](https://github.com/karhunenloeve/DeepLearningCNN/blob/master/img/dvk_illu.jpg)
