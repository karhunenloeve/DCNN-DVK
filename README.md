# Neuronales Netz zur Schätzung von Dosis-Voxel-Kernen
Für die Berechnung der absorbierten Strahlungsdosis in der Dosimetrie werden Monte-Carlo Simulationen auf Gewebedichten aus einer CT-Bildgebung vollführt, um die deponierte Energie zu berechnen.

Der Transfer von Massendichten in die sog. Dosis-Voxel-Kernel entspricht einem Bild zu Bild Transfer. Dieser Transfer kann mit Hilfe von Verfahren aus der Bildrekonstruktion und Bildsegmentierung gelernt werden. Dosis-Voxel-Kerne zu simulieren ist äußerst zeitintensiv und benötigt große Ressourcen.

Im Folgenden wird der Algorithmus vorgestellt, mit dessen Hilfe eine alternative und schnelle Lösung für die Berechnung von DVKs zur Verfügung gestellt wird.

---

## Simulation der DVKs

<center><img src="https://github.com/karhunenloeve/DeepLearningCNN/blob/master/img/dvk_illu.jpg" alt="DVK Illustration" width="400px"></center>

Aus der CT-Bildgebung werden Ausschnitte entnommen, die man als Massenkernel bezeichnet. Das gesamte CT-Bild lässt sich in diese Massenkernel zerlegen. Anschließend wird auf den Massenkernel die Monte-Carlo Simulation durchgeführt. Es wird die Strahlungsenergie eines Isotops im Zentrum des Massenkernels simuliert. Als Ergebnis erhält man die Dosis-Voxel-Kerne. Diese werden anschließend zur Faltung einer SPECT-Bildgebung verwendet, um die Verteilung der absorbierten Strahlungsdosis zu erhalten.

Der Transfer von Massenkernen zu Dosis-Voxel-Kernen wird durch das neuronale Netz ersetzt.

---

## Benutzte Werkzeuge

* [Keras](https://keras.io/) - Keras Tensorflow Wrapper
* [Scikit-Learn](http://scikit-learn.org/stable/index.html/) – Some preprocessing
* [Tensorflow](https://www.tensorflow.org/) – Tensorflow as backend

---

## Zitation
Luciano Melodia: Deep Learning zur Schätzung absorbierter Strahlungsdosis für die nuklearmedizinische Diagnostik. University of Regensburg (2018) [pdf](https://arxiv.org/abs/1805.09108).

    @inproceedings{LuMe2018DVKCNN,
     author    = "Luciano Melodia",
     title     = "Deep Learning zur Schätzung absorbierter Strahlungsdosis für die nuklearmedizinische Diagnostik",
     year      = "2018",
     month     = "April",
     publisher = "University of Regensburg",
     url       = "https://arxiv.org/abs/1805.09108"
    }
